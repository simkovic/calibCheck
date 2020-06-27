import tobii_research as tr
from time import sleep,time
import numpy as np
from threading import Thread,Lock
from Constants import *

VMONSIZECM=(81.8,55.7)
VMONSIZEPIX=(2560,1440)

c=np.cos(25/180*np.pi)
s=np.sin(25/180*np.pi)
R25=np.array([[c,-s],[s,c]])

def etxyz2roomxyz(xyz):
    xyz=np.array(xyz)
    if np.all(xyz==0):return [np.nan,np.nan,np.nan]
    xy=xyz[[2,1]]
    res=np.inner(R25,xy)
    return [xyz[0],res[1],res[0]]


class Eyetracker(Thread):
    def __init__(self,monDistanceCm,fcallback=None,calibf=None,monOffsetDeg=(0,0)):
        Thread.__init__(self)
        self.monOffsetDeg=np.array(monOffsetDeg)
        self.monDistanceCm=monDistanceCm
        self.fcallback=fcallback
        if calibf is None:
            self.calibrated=False
            self.calib=None
        else: 
            try: 
                self.calib=np.loadtxt(calibf)
                self.calibrated=True
            except: 
                print('Calibration file not found')
                self.calibrated=False
                self.calib=None
        self.etinfo=None
        self.ett=None
        self.imgTM=None
        self.imgEI=None
        self.gaze=None
        self.refreshRate=None
        #print('Eye-tracker ready')
        self.gazeLock=Lock()
        self.gqueue=[]
        
    def getInfoHeader(self):
        return self.etinfo
    def getLatestGaze(self,units='deg'):
        return self.gaze
    def getTime(self):
        return tr.get_system_time_stamp()
    def popAllGaze(self):
        self.gazeLock.acquire(True)
        out=self.gqueue
        self.gqueue=[]
        self.gazeLock.release()
        return out
    def getLatestTM(self):
        return self.imgTM
    def getLatestEI(self):
        return self.imgEI
        
    def deg2norm(self,xy):
        temp=xy
        if self.calibrated: temp-=self.monOffsetDeg
        temp=temp/180.*np.pi*float(self.monDistanceCm)/np.array(VMONSIZECM)
        temp[Y]= -temp[Y]
        return temp+0.5
        
    def norm2deg(self,xy):
        ''' transform from normalized unit coordinates on the virtual monitor to
            to degrees of visual angle
        ''' 
        temp=xy
        temp-=0.5 # center at origin
        temp[Y]= -temp[Y] # invert y axis
        temp*=np.array(VMONSIZECM)
        temp=temp/float(self.monDistanceCm)/np.pi*180 # small angle approximation
        return temp 
        
    def __sampleCallback(self,gd):
            if self.fcallback is None: f=-1
            else: f=self.fcallback()
            self.dat=[time(),gd.device_time_stamp,f]
            for eye in [gd.left_eye,gd.right_eye]:
                self.dat.extend(eye.gaze_point.position_on_display_area)
            for eye in [gd.left_eye,gd.right_eye]:
                self.dat.append(eye.pupil.diameter)
            for eye in [gd.left_eye,gd.right_eye]:
                self.dat.extend(eye.gaze_origin.position_in_user_coordinates)
            for eye in [gd.left_eye,gd.right_eye]:
                self.dat.append(eye.gaze_point.validity)
            for eye in [gd.left_eye,gd.right_eye]:
                self.dat.append(eye.pupil.validity)

    def run(self):
        #self.et=tr.EyeTracker('tet-tcp://169.254.151.56')
        ets=tr.find_all_eyetrackers()
        if not len(ets)==1: return None
        self.et=ets[0]
        self.etinfo=("## FW Version: " + self.et.firmware_version+ '\n'+
            "## SDK Version: " + tr.__version__ + '\n'+
                "## Eyetracker Sampling Rate: " + str(self.et.get_gaze_output_frequency())+'\n'+
                '## Calib Matrix: '+str(self.calib).replace('\n',',')+'\n')
        #TODO add other info

        self.refreshRate=self.et.get_gaze_output_frequency()
        sd=None;ts=None
        self.finished=False
        self.dat=None
        self.et.subscribe_to(tr.EYETRACKER_GAZE_DATA,self.__sampleCallback)
        #TODO add eye images
        while not self.finished:
            if self.dat is None:
                sleep(0.001)
                continue
            dat=np.copy(self.dat)
            if len(self.gqueue)>0 and self.gqueue[-1][1]==dat[1]:
                sleep(0.001)
                continue
            
            if dat[-4]==0: dat[[LX,LY]]=np.nan
            else: dat[[LX,LY]]=self.norm2deg(dat[[LX,LY]])
            if dat[-3]==0: dat[[RX,RY]]=np.nan
            else: dat[[RX,RY]]=self.norm2deg(dat[[RX,RY]])
            if dat[-2]==0: dat[LDIAM]=np.nan
            if dat[-1]==0: dat[RDIAM]=np.nan

            if self.calibrated:
                dat[[LX,LY]]=self.calib[:2,0]+self.calib[:2,1]*dat[[LX,LY]]+self.monOffsetDeg
                dat[[RX,RY]]=self.calib[2:,0]+self.calib[2:,1]*dat[[RX,RY]]+self.monOffsetDeg
            
            dat[RDIST]=etxyz2roomxyz(dat[RDIST])
            dat[RDIST]=etxyz2roomxyz(dat[RDIST])
            
            self.gazeLock.acquire(True)
            self.gaze=dat
            self.gqueue.append(dat)
            self.gazeLock.release()
            sleep(0.001)

    def terminate(self):
        self.finished=True
        self.et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA,self.__sampleCallback)
        if self.isAlive(): self.join()
    
    
if __name__=='__main__': 
    et=Eyetracker(60)
    et.start()
    t0=time()
    while time()-t0<20:
        sleep(0.001)
    print(et.getLatestGaze())
    et.terminate()
