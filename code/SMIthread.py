import iViewX    
from time import sleep,time
import numpy as np
from threading import Thread,Lock
from Constants import *

__all__=['Eyetracker']

VMONSIZECM=(81.8,55.7)
VMONSIZEPIX=(2560,1440)

c=np.cos(np.pi/9)
s=np.sin(np.pi/9)
R20=np.array([[c,-s],[s,c]])

def etxyz2roomxyz(xyz):
    xyz=np.array(xyz)
    if np.all(xyz==0):return [np.nan,np.nan,np.nan]
    xy=xyz[[2,1]]
    res=np.inner(R20,xy)
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
        
    #def startRecording(self,calibrated=True):
    #    self.calibrated=calibrated
    #    self.paused=True
    #    self.gaze=[]
    #def stopRecording(self):
    #    self.paused=False 

    
    def vpix2deg(self,xy):
        ''' transform from pix on the virtual monitor to
            to degrees of visual angle
        '''
        if xy[X]==0 and xy[Y]==0:
            return np.array([np.nan,np.nan]) 
        temp=np.array(xy)/np.array(VMONSIZEPIX)#unit coords
        temp-=0.5 # center at origin
        #print xy, temp
        temp[Y]= -temp[Y]
        temp*=np.array(VMONSIZECM)
        temp=temp/float(self.monDistanceCm)/np.pi*180 #small angle approximation
        return temp 

    def deg2norm(self,xy):
        temp=xy
        if self.calibrated: temp-=self.monOffsetDeg
        temp=temp/180.*np.pi*float(self.monDistanceCm)/np.array(VMONSIZECM)
        temp[Y]= -temp[Y]
        return temp+0.5
    def getInfoHeader(self):
        return self.etinfo
    def getLatestGaze(self,units='deg'):
        return self.gaze
    def getTime(self):
        return self.ett
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
    
    def run(self):
        iViewX.connect()
        info = iViewX.getSystemInfo()
        geom=iViewX.getGeometry()
        if info is None: self.etinfo= '## Eye-tracker Info not available' 
        else:
            self.etinfo=("## iViewX Version: " + info[1]+'\n'+
                "## iViewX API Version: " + info[2]+'\n'+
                "## Eye tracking device: " + info[3]+'\n'+
                "## Eyetracker Sampling Rate: " + str(info[0])+'\n'+
                "## Tracking mode: " + info[4]+'\n'+
                '## Calib Matrix: '+str(self.calib).replace('\n',',')+'\n')
        
        if geom is None: self.etinfo+='## Eye-tracker Geometry not available'
        else: self.etinfo+= '## '+str(geom)[1:-1].replace('\'','')+'\n'
        self.geometryName=geom['setupName']
        self.refreshRate=info[0]
        sd=None;ts=None
        self.finished=False
        self.paused=False
        while not self.finished:
            if sd is None: oldt=np.nan
            else: oldt=sd[0]
            self.ett=iViewX.getCurrentTimestamp()
            sd=iViewX.getSample()
            ts=iViewX.getTrackingStatus()
            if not sd is None and not ts is None and oldt!=sd[0]:
                le=self.vpix2deg(sd[1:3])
                re=self.vpix2deg(sd[3:5])
                if self.calibrated:
                    le=self.calib[:2,0]+self.calib[:2,1]*le+self.monOffsetDeg
                    re=self.calib[2:,0]+self.calib[2:,1]*re+self.monOffsetDeg
                if self.fcallback is None: f=-1
                else: f=self.fcallback()
                temp=[time(),sd[0],f, le[X],le[Y],re[X],re[Y]]
                # pupil
                temp.extend(sd[5:7])
                # eye pos
                temp.extend(etxyz2roomxyz(sd[7:10]))
                temp.extend(etxyz2roomxyz(sd[10:13]))
                temp.extend(ts)
                self.gazeLock.acquire(True)
                self.gaze=temp
                self.gqueue.append(temp)
                self.gazeLock.release()
            self.imgTM=iViewX.getTrackingMonitor()
            #self.imgEI=iViewX.getEyeImage()
            sleep(0.001)
        iViewX.disconnect()
    
    def terminate(self):
        self.finished=True
        if self.isAlive(): self.join()

if __name__=='__main__': 
    et=Eyetracker(60)
    et.start()
    t0=time()
    while time()-t0<20:
        sleep(0.001)
    print(et.getLatestGaze())
    et.terminate()
