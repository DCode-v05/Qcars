import numpy as np
import math


class MadgwickFilter:

    def __init__(self, sample_rate_hz=100.0, beta=0.1):
        self.dt   = 1.0 / sample_rate_hz
        self.beta = beta
        self.q    = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, gyro, accel, mag):
        q1, q2, q3, q4 = self.q

        norm = np.linalg.norm(accel)
        if norm < 1e-6:
            return
        ax, ay, az = accel / norm

        norm = np.linalg.norm(mag)
        if norm < 1e-10:
            self._update_6dof(gyro, accel)
            return
        mx, my, mz = mag / norm

        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q4 = 2.0 * q4

        hx = (mx*(0.5-q3*q3-q4*q4) +
              my*(q2*q3-q1*q4) +
              mz*(q2*q4+q1*q3))
        hy = (mx*(q2*q3+q1*q4) +
              my*(0.5-q2*q2-q4*q4) +
              mz*(q3*q4-q1*q2))

        bx = 2.0 * math.sqrt(hx*hx + hy*hy)
        bz = 2.0 * (mx*(q2*q4-q1*q3) +
                    my*(q3*q4+q1*q2) +
                    mz*(0.5-q2*q2-q3*q3))

        s1 = (-_2q3*(2*(q2*q4-q1*q3)-ax) +
               _2q2*(2*(q1*q2+q3*q4)-ay) +
              -4*bz*q3*(bz*(0.5-q3*q3-q4*q4)-mz) +
               (-_2q1*bz+_2q4*bx)*(bz*(q2*q4-q1*q3)-mx*0.5+my*(q1*q2+q3*q4)))

        s2 = (_2q4*(2*(q2*q4-q1*q3)-ax) +
              _2q1*(2*(q1*q2+q3*q4)-ay) +
             -4*q2*(1-2*q2*q2-2*q3*q3-az) +
              _2q4*bx*(bx*(0.5-q3*q3-q4*q4)+bz*(q2*q4-q1*q3)-mx) +
              (_2q1*bx+_2q3*bz)*(bx*(q2*q3-q1*q4)+bz*(q1*q2+q3*q4)-my) +
              _2q4*bz*(bz*(0.5-q2*q2-q3*q3)-mz))

        s3 = (-_2q1*(2*(q2*q4-q1*q3)-ax) +
               _2q4*(2*(q1*q2+q3*q4)-ay) +
              (-4*q3)*(1-2*q2*q2-2*q3*q3-az) +
              (-_2q2*bx-_2q4*bz)*(bx*(0.5-q3*q3-q4*q4)+bz*(q2*q4-q1*q3)-mx) +
              (_2q1*bx+_2q3*bz)*(bx*(q2*q3-q1*q4)+bz*(q1*q2+q3*q4)-my) +
              (2*bx*q3-_2q1*bz)*(bz*(0.5-q2*q2-q3*q3)-mz))

        s4 = (_2q2*(2*(q2*q4-q1*q3)-ax) +
              (-_2q1)*(2*(q1*q2+q3*q4)-ay) +
              (-_2q2*bz+_2q4*bx)*(bx*(0.5-q3*q3-q4*q4)+bz*(q2*q4-q1*q3)-mx) +
              (-_2q1*bx+_2q2*bz)*(bx*(q2*q3-q1*q4)+bz*(q1*q2+q3*q4)-my) +
              _2q2*bz*(bz*(0.5-q2*q2-q3*q3)-mz))

        norm = math.sqrt(s1*s1+s2*s2+s3*s3+s4*s4)
        if norm < 1e-10:
            norm = 1.0
        s1/=norm; s2/=norm; s3/=norm; s4/=norm

        gx, gy, gz = gyro
        qDot1 = 0.5*(-q2*gx-q3*gy-q4*gz) - self.beta*s1
        qDot2 = 0.5*( q1*gx+q3*gz-q4*gy) - self.beta*s2
        qDot3 = 0.5*( q1*gy-q2*gz+q4*gx) - self.beta*s3
        qDot4 = 0.5*( q1*gz+q2*gy-q3*gx) - self.beta*s4

        q1 += qDot1*self.dt
        q2 += qDot2*self.dt
        q3 += qDot3*self.dt
        q4 += qDot4*self.dt

        norm = math.sqrt(q1*q1+q2*q2+q3*q3+q4*q4)
        self.q = np.array([q1,q2,q3,q4]) / norm

    def _update_6dof(self, gyro, accel):
        q1,q2,q3,q4 = self.q
        norm = np.linalg.norm(accel)
        if norm < 1e-6:
            return
        ax,ay,az = accel/norm
        s1 = -2*q3*(2*q2*q4-2*q1*q3-ax)+2*q2*(2*q1*q2+2*q3*q4-ay)
        s2 =  2*q4*(2*q2*q4-2*q1*q3-ax)+2*q1*(2*q1*q2+2*q3*q4-ay)-4*q2*(1-2*q2**2-2*q3**2-az)
        s3 = -2*q1*(2*q2*q4-2*q1*q3-ax)+2*q4*(2*q1*q2+2*q3*q4-ay)-4*q3*(1-2*q2**2-2*q3**2-az)
        s4 =  2*q2*(2*q2*q4-2*q1*q3-ax)-2*q3*(2*q1*q2+2*q3*q4-ay)
        norm = math.sqrt(s1**2+s2**2+s3**2+s4**2) or 1.0
        gx,gy,gz = gyro
        self.q += self.dt*np.array([
            0.5*(-q2*gx-q3*gy-q4*gz)-self.beta*s1/norm,
            0.5*(q1*gx+q3*gz-q4*gy) -self.beta*s2/norm,
            0.5*(q1*gy-q2*gz+q4*gx) -self.beta*s3/norm,
            0.5*(q1*gz+q2*gy-q3*gx) -self.beta*s4/norm
        ])
        self.q /= np.linalg.norm(self.q)

    def get_euler_angles(self):
        w,x,y,z = self.q
        sinr_cosp = 2*(w*x+y*z)
        cosr_cosp = 1-2*(x*x+y*y)
        roll  = math.atan2(sinr_cosp, cosr_cosp)
        sinp  = max(-1.0, min(1.0, 2*(w*y-z*x)))
        pitch = math.asin(sinp)
        siny_cosp = 2*(w*z+x*y)
        cosy_cosp = 1-2*(y*y+z*z)
        yaw   = math.atan2(siny_cosp, cosy_cosp)
        return {
            "roll_deg":  math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg":   math.degrees(yaw)
        }
