import numpy as np
import cv2
from ultralytics import YOLO
import torch 
import os
from pit.YOLO.utils import TrafficLight,Obstacle,MASK_COLORS_RGB
import requests
from tqdm import tqdm

class YOLOv8():

    def __init__(
        self,
        imageWidth = 640,
        imageHeight = 480,
        modelPath = None,
        ):

        self.defaultPath = os.path.normpath(os.path.join(
                        os.path.dirname(__file__), 
                        '../../../resources/pretrained_models/yolov8s-seg.engine'))
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.modelPath = self.__check_path(modelPath)
        self.img=np.empty((480,640,3),dtype=np.uint8)
        self.objectsDetected=None
        self.processedResults=[]
        self._calc_distence = False
        self.net=YOLO(self.modelPath,task='segment')
        print('YOLOv8 model loaded')   

    def pre_process(self,inputImg):
        inputImgClone=inputImg.copy()
        if inputImgClone.shape[:2] != (self.imageHeight,self.imageWidth):
            inputImgClone=cv2.resize(inputImgClone,
                                     (self.imageWidth,self.imageHeight))
        self.img[:,:,:]=inputImgClone[:,:,:]
        return self.img
    
    def predict(self, inputImg, classes = [2,9,11], confidence = 0.3, verbose = False, half = False):
        self.predictions = self.net.predict(inputImg,
                                            verbose = verbose,
                                            imgsz = (self.imageHeight,
                                                     self.imageWidth),
                                            classes = classes,
                                            conf = confidence,
                                            half = half
                                            )
        self.objectsDetected=self.predictions[0].boxes.cls.cpu().numpy()
        self.FPS=1000/self.predictions[0].speed['inference']
        return self.predictions[0]
    
    def render(self):
        
        annotatedImg = self.predictions[0].plot()

        return annotatedImg

    def post_processing(self,alignedDepth=None,clippingDistance=5):
        '''
        a depth image aligned to the rgb input is needed for computing the 
        distnace of the detected obstacle
        '''
        self.processedResults = []
        if len (self.objectsDetected) == 0:
            return self.processedResults
        self.bounding = self.predictions[0].boxes.xyxy.cpu().numpy().astype(int)
        if alignedDepth is not None:
            depth3D = np.dstack((alignedDepth,alignedDepth,alignedDepth))
            bgRemoved = np.where((depth3D > clippingDistance)| 
                                 (depth3D <= 0), 0, depth3D)
            self._calc_distence = True
            self.depthTensor=torch.as_tensor(bgRemoved,device="cuda:0")
        for i in range(len(self.objectsDetected)):
            if self.objectsDetected[i]==9:
                trafficBox = self.bounding[i]
                traficLightColor = self.check_traffic_light(trafficBox,self.img)
                result=TrafficLight(color=traficLightColor)
                result.name+=(' ('+traficLightColor+')')
            else:
                name=self.predictions[0].names[self.objectsDetected[i]]
                result=Obstacle(name=name)
            if alignedDepth is not None:
                mask=self.predictions[0].masks.data.cuda()[i]
                distance=self.check_distance(mask,self.depthTensor[:,:,:1])
                result.distance=distance.cpu().numpy().round(3)
            points=self.predictions[0].boxes.xyxy.cpu()[i]
            x=int(points.numpy()[0])
            y=int(points.numpy()[1])
            result.x=x
            result.y=y
            self.processedResults.append(result)
        return self.processedResults

    def post_process_render(self, showFPS = False, bbox_thickness = 4):
        if not self.processedResults:
            return self.img
        colors=[]
        masks = self.predictions[0].masks.data.cuda()
        boxes = self.predictions[0].boxes.xyxy.cpu().numpy().astype(int)
        imgClone=self.img.copy()
        for i in range(len(self.objectsDetected)):
            colors.append(MASK_COLORS_RGB[self.objectsDetected[i].astype(int)])
            name=self.processedResults[i].name
            x=self.processedResults[i].x
            y=self.processedResults[i].y
            distance=self.processedResults[i].distance
            cv2.rectangle(imgClone,(boxes[i,:2]),(boxes[i,2:4]),colors[i],bbox_thickness)
            cv2.putText(imgClone, name, 
                        (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        colors[i], 2)
            if self._calc_distence:
                cv2.putText(imgClone,str(distance) + " m",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            colors[i], 2)
        if showFPS:
            cv2.putText(imgClone, 'FPS: '+str(round(self.FPS)), 
                        (565,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 2)
        imgTensor=torch.from_numpy(imgClone).to("cuda:0")
        imgMask=self.mask_color(masks, imgTensor,colors)
        return imgMask

    @staticmethod
    def convert_to_trt(path,
                       imageWidth = 640,
                       imageHeight = 480,
                       half = True,
                       dynamic = False,
                       batch = 1,
                       int8 = False,
                       simplify = True
                       ):
        print('Converting to teneorRT engine')
        model = YOLO(path)
        model.export(format="engine",
             imgsz=(imageHeight,imageWidth),
             half=half,
             dynamic=dynamic,
             batch=batch,
             int8=int8,
             simplify=simplify)
        enginePath = os.path.splitext(path)[0]+'.engine'
        return enginePath
        
    @staticmethod
    def mask_color(masks, im_gpu,colors, alpha=0.5):
        colors = torch.tensor(colors, device="cuda:0", dtype=torch.float32) / 255.0 
        colors = colors[:, None, None]
        masks = masks.unsqueeze(3)
        masks_color = masks * (colors * alpha)
        inv_alpha_masks = (1 - masks * alpha).cumprod(0) 
        mcs = masks_color.max(dim=0).values  
        im_gpu = im_gpu/255
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs 
        im_mask = im_gpu * 255
        im_mask_np = im_mask.squeeze().byte().cpu().numpy()
        return im_mask_np

    @staticmethod
    def check_traffic_light(traffic_box,im_cpu):
        mask = np.zeros((480,640),dtype='uint8')
        x1,y1,x2,y2=(traffic_box[0], traffic_box[1], traffic_box[2], traffic_box[3])
        d = 0.3*(x2-x1)
        R_center=(int(x1/2+x2/2),int(3*y1/4+y2/4))
        Y_center=(int(x1/2+x2/2),int(y1/2+y2/2))
        G_center=(int(x1/2+x2/2),int(y1/4+3*y2/4))
        maskR=cv2.circle(mask.copy(),R_center,int(d/2),1,-1)
        maskY=cv2.circle(mask.copy(),Y_center,int(d/2),1,-1)
        maskG=cv2.circle(mask.copy(),G_center,int(d/2),1,-1)
        maskR_gpu=torch.tensor(maskR,device="cuda:0").unsqueeze(2)
        maskY_gpu=torch.tensor(maskY,device="cuda:0").unsqueeze(2)
        maskG_gpu=torch.tensor(maskG,device="cuda:0").unsqueeze(2)
        im_hsv=cv2.cvtColor(im_cpu, cv2.COLOR_RGB2HSV)
        im_hsv_gpu = torch.tensor(im_hsv,device="cuda:0")
        masked_red = im_hsv_gpu*maskR_gpu
        masked_yellow = im_hsv_gpu*maskY_gpu
        masked_green = im_hsv_gpu*maskG_gpu
        value_R=torch.sum(masked_red[:,:,2])/torch.count_nonzero(masked_red[:,:,2])
        value_Y=torch.sum(masked_yellow[:,:,2])/torch.count_nonzero(masked_yellow[:,:,2])
        value_G=torch.sum(masked_green[:,:,2])/torch.count_nonzero(masked_green[:,:,2])
        mean = (value_R+value_Y+value_G)/3
        threshold_perc=0.25
        min= torch.min(torch.tensor([value_R,value_Y,value_G]))
        max= torch.max(torch.tensor([value_R,value_Y,value_G]))
        if (max-min)<30:
            return 'idle'
        threshold=(max-min)*threshold_perc
        # print('red',value_R,'yellow',value_Y,'green',value_G,mean,threshold)
        redOn=(value_R>mean) and (value_R-mean)>threshold
        yellowOn=(value_Y>mean) and (value_Y-mean)>threshold
        greenOn=(value_G>mean) and (value_G-mean)>threshold
        traffic_light_status=[redOn.cpu().numpy(),yellowOn.cpu().numpy(),greenOn.cpu().numpy()]
        colors=['red','yellow','green']
        traffic_light_color=''
        for i in range(len(traffic_light_status)):
            if traffic_light_status[i]:
                traffic_light_color+=colors[i]
                traffic_light_color+=' '
        return traffic_light_color

    @staticmethod
    def check_distance(mask,depth_gpu):
        mask=mask.unsqueeze(2)
        isolated_depth = mask*depth_gpu
        # distance = torch.sum(isolated_depth)/torch.count_nonzero(isolated_depth)
        distance = torch.median(isolated_depth[isolated_depth.nonzero(as_tuple=True)])
        return distance
    
    @staticmethod
    def reshape_for_matlab_server(frame):
        frame=frame.copy()[:,:,[2,1,0]]
        flatten = frame.flatten(order='F').copy() 
        return flatten.reshape(frame.shape,order='C')
    
    def __download_model(self):
        url = 'https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt'
        filepath = os.path.splitext(self.defaultPath)[0]+'.pt'
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        print('Downloading yolov8s-seg.pt')
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

    def __check_path(self, modelPath):
        
        if modelPath:

            enginePath = os.path.splitext(modelPath)[0]+'.engine'
            if os.path.exists(enginePath):
                return enginePath
            
            if os.path.splitext(modelPath)[1] != '.engine':
                try:
                    enginePath = self.convert_to_trt(path = modelPath,
                                                     imageWidth = self.imageWidth,
                                                     imageHeight = self.imageHeight)
                except:
                    errorMsg = modelPath + ' does not exist, or is in unsupported formats.'
                    raise SystemExit(errorMsg) 
            else:
                enginePath = modelPath
        else: 
            if not os.path.exists(self.defaultPath):
                ptPath = os.path.splitext(self.defaultPath)[0]+'.pt'
                if not os.path.exists(ptPath):
                    self.__download_model()
                enginePath = self.convert_to_trt(path = ptPath,
                                                 imageWidth = self.imageWidth,
                                                 imageHeight = self.imageHeight)
            else:
                enginePath = self.defaultPath
        return enginePath
        # if modelPath:
        #     self.modelPath=modelPath
        # else: 
        #     self.modelPath=os.path.normpath(os.path.join(
        #     os.path.dirname(__file__), '../../../resources/pretrained_models/yolov8s-seg.engine'))
        # if not os.path.exists(self.modelPath):
        #     self.convert_to_trt(imageWidth=self.imageWidth,imageHeight=self.imageHeight)