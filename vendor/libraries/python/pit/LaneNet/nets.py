import numpy as np
import cv2
import torch 
import os
import torchvision.transforms as transforms
import tensorrt as trt
import requests
from tqdm import tqdm
from pit.LaneNet.utils import NP_TO_TORCH_DICT
import time 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pit.YOLO.utils import MASK_COLORS_RGB
import copy

class LaneNet():

    def __init__(
        self,
        modelPath = None,
        imageHeight = 480,
        imageWidth = 640,
        rowUpperBound = 0
        ):

        self.defaultPath = os.path.normpath(os.path.join(
                              os.path.dirname(__file__), 
                              '../../../resources/pretrained_models/lanenet.engine'))
        self.modelPath = self.__check_path(modelPath)
        self.rowUpperBound = rowUpperBound
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.imgTransforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # use the same nomalization as the training set
            ])
        self.__allocate_buffers()
        self.engine = self.__load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream(device='cuda:0')
        print('LaneNet loaded')
    
    def pre_process(self, inputImg):
        if inputImg.shape[:2] != (self.imageHeight,self.imageWidth):
            inputImg = cv2.resize(inputImg, 
                           (self.imageWidth, self.imageHeight), 
                           interpolation = cv2.INTER_LINEAR)
        self.imgClone = inputImg
        self.imgTensor = self.imgTransforms(self.imgClone[self.rowUpperBound:,:,:])
        return self.imgTensor
    
    def predict(self, inputImg):
        if not inputImg.dtype == torch.float32:
            raise SystemExit('input image data type error, need to be torch.float32')
        self.inputBuffer[:] = inputImg.flatten()[:]
        bindings = [self.inputBuffer.data_ptr()] +\
                   [self.binaryLogitsBuffer.data_ptr()] +\
                   [self.binaryBuffer.data_ptr()] +\
                   [self.instanceBuffer.data_ptr()]
        start = time.time()
        self.context.execute_async_v2(bindings=bindings, 
                                      stream_handle=self.stream.cuda_stream)
        end=time.time()
        self.stream.synchronize()
        self.FPS = 1/(end-start)
        self.binaryPred = (self.binaryBuffer.cpu().numpy().reshape((256,512))*255).astype(np.uint8)
        self.instancePred = self.instanceBuffer.cpu().numpy().reshape((3,256,512)).transpose((1, 2, 0))
        return (self.binaryPred,self.instancePred)
    
    def render(self,showFPS = True):
        binary3d = np.dstack((self.binaryPred,self.binaryPred,self.binaryPred))
        # instanceVisual = (self.instancePred*255).clip(max=255).astype(np.uint8)
        instanceVisual = (self.instancePred*255).astype(np.uint8)
        lanes= cv2.bitwise_and(instanceVisual,binary3d)
        overlaid=cv2.addWeighted(lanes,
                                 1,
                                 self.imgTensor.numpy().transpose((1, 2, 0))[:,:,[2,1,0]],
                                 1,
                                 0,
                                 dtype=cv2.CV_32F)
        resized=cv2.resize(overlaid, 
                           (self.imageWidth, self.imageHeight - self.rowUpperBound), 
                           interpolation = cv2.INTER_LINEAR)
        resized=(resized*255).clip(max=255).astype(np.uint8)[:,:,[2,1,0]]
        annotatedImg = copy.deepcopy(self.imgClone)
        annotatedImg[self.rowUpperBound:,:,:]=resized
        if showFPS:
            cv2.putText(annotatedImg, 'FPS: '+str(round(self.FPS)), 
                        (565,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 2)
        return annotatedImg
    
    def post_process(self, eps=0.5, min_samples=250, min_area=100):
        
        self.lane_mask = np.zeros((256,512,3),dtype=np.uint8)

        ### remove blobs smaller than min_area from binary lane prediction
        CCA_ret = cv2.connectedComponentsWithStats(self.binaryPred, 
                                                   connectivity=8, 
                                                   ltype=cv2.CV_32S)
        CCA_labels = CCA_ret[1]
        CCA_stats = CCA_ret[2]
        for index, stat in enumerate(CCA_stats):
            if stat[4]<=min_area:
                idx = np.where(CCA_labels == index)
                self.binaryPred[idx]=0

        ### apply cluster to lane embeddings
        land_idx = np.where(self.binaryPred == 255)
        lane_embedding = self.instancePred[land_idx]
        lane_coord = np.vstack((land_idx[1], land_idx[0])).T
        db = DBSCAN(eps=eps, min_samples=min_samples)
        try:
            lane_embedding = StandardScaler().fit_transform(lane_embedding)
            db.fit(lane_embedding)
        except Exception as err:
            print(err)
            return self.lane_mask
        cluster_labels=db.labels_
        if cluster_labels is None:
            return self.lane_mask
        cluster_unique_labels=np.unique(cluster_labels)
        # cluster_centers=db.components_
        for index,label in enumerate(cluster_unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(cluster_labels ==label)
            pix_coord_idx = tuple((lane_coord[idx][:,1],lane_coord[idx][:,0]))
            self.lane_mask [pix_coord_idx]= MASK_COLORS_RGB[index]
        
        return self.lane_mask
    
    def post_process_render(self):
        annotatedImg=copy.deepcopy(self.imgClone)
        rbgRaw=(self.imgTensor.numpy().transpose((1, 2, 0))*255).astype(np.uint8)
        overlaid=cv2.addWeighted(self.lane_mask,
                                 1,
                                 rbgRaw,
                                 1,
                                 0,)
                                #  dtype=cv2.CV_32F)
        resized=cv2.resize(overlaid, 
                           (self.imageWidth, self.imageHeight - self.rowUpperBound), 
                           interpolation = cv2.INTER_LINEAR)
        annotatedImg[self.rowUpperBound:,:,:] = resized
        return annotatedImg

    @staticmethod
    def convert_to_trt(path):
        print('Converting to teneorRT engine')

        #convert to onnx format
        model = torch.load(path)
        model.eval()
        dummy_input = torch.rand((1,3,256,512)).cuda()
        onnx_path=os.path.join(os.path.split(path)[0],'lanenet.onnx')
        torch.onnx.export(model, dummy_input, onnx_path)
        
        #convert to tensorrt engine
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        EXPLICIT_BATCH = []
        if trt.__version__[0] >= '7':
            EXPLICIT_BATCH.append(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(*EXPLICIT_BATCH)
        parser= trt.OnnxParser(network, logger)

        success = parser.parse_from_file(onnx_path)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not success:
            print('Parser read failed')

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 22) # 1 MiB
        config.set_flag(trt.BuilderFlag.FP16)
        serialized_engine = builder.build_serialized_network(network, config)
        enginePath=os.path.join(os.path.split(path)[0],'lanenet.engine')
        with open(enginePath, 'wb') as f:
            f.write(serialized_engine)
        return enginePath

    def __check_path(self,modelPath):
    
        if modelPath:
            enginePath = os.path.splitext(modelPath)[0]+'.engine'
            if os.path.exists(enginePath):
                return enginePath
            if os.path.splitext(modelPath)[1] != '.engine':
                try:
                    enginePath = self.convert_to_trt(modelPath)
                except:
                    errorMsg = modelPath + ' does not exist, or is in unsupported format, please ensure the model is a .pt file.'
                    raise SystemExit(errorMsg) 
            else:
                enginePath = modelPath
        else: 
            if not os.path.exists(self.defaultPath):
                self.__download_model()
                ptPath = os.path.splitext(self.defaultPath)[0]+'.pt'
                enginePath = self.convert_to_trt(ptPath)
            else:
                enginePath = self.defaultPath
        return enginePath

    def __load_engine(self):
        self.logger = trt.Logger()
        if not os.path.isfile(self.modelPath):
            raise SystemExit('ERROR: file (%s) not found!' % self.modelPath)
        with open(self.modelPath,'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def __allocate_buffers(self):
        self.inputBuffer          = torch.empty((512*256*3),
                                             dtype=torch.float32,
                                             device='cuda:0')
        self.binaryLogitsBuffer         = torch.empty((512*256*2),
                                             dtype=torch.float32,
                                             device='cuda:0')
        self.binaryBuffer   = torch.empty((512*256*1),
                                             dtype=torch.int32,
                                             device='cuda:0')
        self.instanceBuffer       = torch.empty((512*256*3),
                                             dtype=torch.float32,
                                             device='cuda:0')
        
    def __download_model(self):
        url = 'https://quanserinc.box.com/shared/static/c19pjultyikcgzlbzu6vs8tu5vuqhl2n.pt'
        filepath = os.path.splitext(self.defaultPath)[0]+'.pt'
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        print('Downloading lanenet.pt')
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")