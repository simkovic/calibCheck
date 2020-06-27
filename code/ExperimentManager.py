# -*- coding: utf-8 -*-
import numpy as np
import cv2,os
from time import sleep
from psychopy.core import getTime
from psychopy import prefs,visual,monitors
prefs.general['audioLib']=['pygame']
from psychopy.sound import Sound
from threading import Thread
from Constants import *

__all__=['asusMG279', 'ExperimentManager','QinitDisplay']

# define a couple of monitors available in our lab    
asusMG279=monitors.Monitor('asusMG279', width=59.6, distance=70)
asusMG279.setSizePix((2560,1440))
asusVS229=monitors.Monitor('asusVS229', width=47.6, distance=50)
asusVS229.setSizePix((1920,1080))
eizo=monitors.Monitor('eizo', width=34, distance=40)
eizo.setSizePix((1280,1024))


def QinitDisplay(Q):
    '''create ad return display based on the Q class '''
    if Q.fullscr:ws=Q.monitor.getSizePix()
    else: ws=Q.winSize
    wind=visual.Window(monitor=Q.monitor,fullscr=Q.fullscr,
        size=ws,units='deg',color=Q.bckgCLR,pos=Q.winPos,
        winType='pyglet',screen=Q.screen,waitBlanking=True)
    return wind

def Qsave(Q,fn):
    '''save the Q class'''
    f=open(Q.outputPath+fn+'.q','wb')
    try: 
        s=str(dict(vars(Q))).encode()
        f.write(s)
        f.close()
    except: f.close(); raise
def Qload(filepath):
    '''load the Q class '''
    pass
    #TODO
        
class AttentionCatcher():
    def __init__(self,win,pos,sound,monHz=60):
        self.win=win
        self.cMaxRadius=1.5
        self.cHz=3
        self.monHz=monHz
        clra=[[1,-1,1],[-1,1,1],[1,1,-1]][np.random.randint(3)]
        clrb=[[-1,-1,1],[-1,1,-1],[1,-1,-1]][np.random.randint(3)]
        self.ci=visual.Circle(win,radius=0.1,fillColor='red',lineWidth=0)
        #self.co=visual.Circle(win,radius=self.cMaxRadius,fillColor='white',lineWidth=0)
        
        mask=np.load(Q.dataPath+os.path.sep+'spiral.npy')
        im=np.zeros((mask.shape[0],mask.shape[1],3))
        for i in range(3):im[mask==1,i]=clra[i]
        for i in range(3):im[mask==0,i]=clrb[i]
        self.co=visual.ImageStim(win,im,mask='circle',size=self.cMaxRadius*2,units='deg',interpolate=True)
        self.inc= -self.cMaxRadius/float(self.monHz)*self.cHz
        self.soundStim=sound
        #self.soundStim.setLoops(100)
        self.co.setPos(pos)
        self.ci.setPos(pos)
        self.soundOn=False
        self.dur=-1
    def draw(self,sound=True):
        if sound and not self.soundOn: 
            self.soundStim.play(loops=-1)
            self.soundOn=True
        self.co.setSize(self.co.size[0]+self.inc*2)
        if self.co.size[0]<0 or self.co.size[0]>2*self.cMaxRadius:
            self.inc*=-1
        self.co.draw();self.ci.draw()

    def getPos(self):
        return self.co.pos
    def setPos(self,pos):
        self.co.setPos(pos)
        self.ci.setPos(pos)
    def stop(self):
        self.soundStim.stop()
        self.soundOn=False
    def flip(self):
        if self.dur!=-1 and getTime()-self.t0 < self.dur:
            self.draw()
            self.co.win.flip()
            return 0
        else:
            if self.soundOn: self.stop()
            self.co.win.flip()
            return 1
    def show(self,dur):
        self.dur=dur
        self.t0=getTime()
        
class Camera(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.status=-1
        self.frame=None   
    def run(self):
        self.finished=False
        self.paused=False
        while not self.finished:
            if self.cap.isOpened() and not self.paused:
                self.status,self.frame=self.cap.read()
            else:
                self.status=-1
                self.frame=None
            sleep(0.001)
        self.cap.release()
        
    def getLatestSample(self):
        return self.status,self.frame
    def startRecording(self):
        self.paused=False
        
    def stopRecording(self):
        self.paused=True 
        
    def terminate(self):
        self.finished=True
        if self.isAlive(): self.join()

def infoboxBaby(pos,fn=None):
    '''pos - list with coordinates at which the gui is centered
       fn - write meta data to this file
       returns participant id, age cohort, eye tracking device, output suffix'''
    import datetime
    from psychopy import gui          
    myDlg = gui.Dlg(title='VP Info',pos=pos)    
    myDlg.addField('VP ID:',0)# subject id
    today=datetime.date.today()
    myDlg.addField('Kohorte',choices=['4M','7M','10M'],initial='7M')
    myDlg.addField('ET:',choices=('smi','tob'))
    if not fn is None:
        myDlg.addField('Geschlecht:',choices=('mann','frau')) 
        myDlg.addText('Geburtsdatum')
        myDlg.addField('Tag',choices=range(1,32),initial=15)
        myDlg.addField('Monat',choices=range(1,13),initial=6)
        myDlg.addField('Jahr',choices=[2018,2019])
        myDlg.addText('Datenverwaltung')
        myDlg.addField('Versuchsleiter',choices=['MS','SR','MV','SW','FK','KD','SMW'])
        myDlg.addField('Datasharing',initial=False)
        #myDlg.addField('FunLT Bedingung',initial=-1)
        #myDlg.addField('Pursuit Bedingung',initial=-1)
    myDlg.show()#show dialog and wait for OK or Cancel
    if myDlg.OK:
        import numpy as np
        d=myDlg.data 
        if not fn is None and not d[0]==0:
            age=today-datetime.date(d[6],d[5],d[4])
            vlid=['MS','SR','MV','SW','FK','KD','SMW'].index(d[7])
            vpinfo=[d[0],int(d[1][:-1]),int(d[3]=='mann'),age.days,vlid,np.int32(d[8])]
            strout= ('\n'+'{},'*(len(vpinfo)-1)+'{}').format(*(vpinfo))
            with open(fn,'a') as f: f.write(strout)
        #curd=str(datetime.datetime.today())
        #curd='-'+curd[:10]+'-'+curd[11:19];curd=curd.replace(':','-')
        return d[0],d[1],['smi','tob'].index(d[2]),'Vp%dc%s'%(d[0],d[1])
    else:
        import sys
        sys.exit()
class Q():
    '''Experiment Manager Settings'''
    dataPath=os.path.dirname(os.path.realpath(__file__))+os.path.sep+'EMdata'+os.path.sep
    vpinfofn='vpinfo.res'
    WS=[[840,1920],#window
        [480,640],#camera
        [240,300],#tracking monitor
        [240,700],#eye image
        [626,920],# gaze on screen
        [300,300],#eye position lateral
        [300,300]#eye position sagital
        ]
    ofs=[[np.nan,np.nan],[],[],[],[np.nan,np.nan],[],[]]
    ofs[TM]=[0,WS[EI][W]]
    ofs[CAM]=[0,60]
    ofs[EI]=[WS[CAM][H],0]
    ofs[EP]=[WS[TM][H],WS[EI][W]]
    ofs[EP+1]=[WS[TM][H]+WS[EP][H],WS[EI][W]]
    ofs[MON]=[0,WS[EI][W]+WS[TM][W]]
    WP=[]
    for i in range(len(ofs)):
        WP.append([[ofs[i][H],ofs[i][H]+WS[i][H]],
            [ofs[i][W],ofs[i][W]+WS[i][W]]])
    
    fontclr=(255,255,255)
    eyeclr=[(0,255,0),(0,0,255)]
    winname='Ximkoview 2.0: Matus\'s Grossartiges Eyetracking-Programm'
    circleradius=5# pix
    exyslope=600 # in mm
    exyoffset=np.array([0,-100,-700])+exyslope/2.
    
    
class ExperimentManager(Thread):
    def __init__(self,fcallback,ecallback,Qexp,showCam=True,
        calibVp=-9999,loglevel=0,infobox=None):
        ''' fcallback - callback function, returns the window current frame count
             of Experiment, value is written to ET output
            ecallback - callback function, used for controlling Experiment 
                with commands from keyboard
            Qexp - setting of the Experiment class
            showCam - bool, if true displays images from camera that shows infant
            calibVp - loads calibration parameters from infants with id -9999 
            loglevel - 0 = no file output 
                     - 1 = output eye tracking file 
                     - 2 = ET file + additional file written by Experiment instance
                     the parameter is overriden to loglevel=0 if subject id is 0
            infobox - function called at the start of experiment to collect meta-data
        '''
        Thread.__init__(self)
        self.showCam=showCam
        self.ecallback=ecallback
        self.fcallback=fcallback
        self.loglevel=loglevel
        self.Qexp=Qexp
        if infobox is None: infobox=infoboxBaby
        #get info
        if self.loglevel>0:
            self.vp,self.cohort,slot,suffix=infobox((0,0),
                fn=Q.dataPath+Q.vpinfofn)
        else: 
            self.vp=0; self.cohort='';suffix=''
        if self.vp==0: self.loglevel=0
        self.fn=Qexp.expName+suffix
        #if self.loglevel>0: Qsave(Qexp,self.fn)
        # setup experiment
        self.winE=QinitDisplay(Qexp)
        if self.loglevel>1: self.outE=open(Qexp.outputPath+self.fn+'.res','w')
        if self.loglevel>0:
            self.outL=open(Qexp.outputPath+self.fn+'.log','a')
            import psychopy,sys
            info='## Python Version: {}.{}.{}\n'.format(*sys.version_info[:3])+\
                '## Psychopy Version:' +psychopy.__version__+'\n'
            self.outL.write(info)
        # eyetracker setup
        calibf=Q.dataPath+'calibData'+os.path.sep+str(calibVp)+'.res'
        if slot==0: from SMIthread import Eyetracker
        elif slot==1: from TobiiThread import Eyetracker
        self.ET=Eyetracker(Qexp.monitor.getDistance(),fcallback=fcallback,
             monOffsetDeg=Qexp.stimOffset,calibf=calibf)
        # sound setup
        self.sound=Sound(Q.dataPath+'attentionCatcher.ogg')
        # attention catcher
        self.AC= AttentionCatcher(self.winE,-np.array(Qexp.stimOffset),
            self.sound,Qexp.refreshRate)
        # camera setup
        self.VC = Camera()
        #TODO save camera stream?
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #self.outV = cv2.VideoWriter(Qexp.outputPath+self.fn+'.avi',
        #    fourcc, 60.0, (Q.WS[WIN][W],Q.WS[WIN][H]))
        self.VC.start()
        self.ET.start()
        hd=self.ET.getInfoHeader()
        
        while hd is None:
            sleep(0.01)
            hd=self.ET.getInfoHeader()
        if self.loglevel>0:
            self.outL.write(hd)
            self.outL.flush()
        #self.logLock=Lock()
        self.updateGui=True
        self.expIsWaiting=False
        self.showAC=False
        from PIL import Image
        self.ep0=np.array(Image.open(Q.dataPath+'ep0.png'))[:,:,:3]
        self.ep1=np.array(Image.open(Q.dataPath+'ep1.png'))[:,:,:3]
        self.monim=np.array(Image.open(Q.dataPath+'mon.png'))[:,:,:3]
    
    def run(self):
        '''use ExperimentManager.start() to run this code'''
        self.finished=False
        self.infoscreen=np.zeros((Q.WS[WIN][H],Q.WS[WIN][W],3),dtype=np.uint8)
        cv2.namedWindow(Q.winname)
        cv2.moveWindow(Q.winname,0,0)
        while not self.finished:
            if self.updateGui:
                #self.infoscreen=np.zeros(self.infoscreen.shape,np.uint8)
                gzs=self.ET.popAllGaze()
                for gz in gzs:
                    
                    if gz is None: continue
                    if not self.expIsWaiting:
                        s=("{};"*(len(gz)-1)+'{}\n').format(*gz)
                        if self.loglevel>0: self.outL.write(s)
                ret,frame=self.VC.getLatestSample()#show camera
                if ret==1: 
                    self.infoscreen[Q.WP[CAM][H][S]:Q.WP[CAM][H][E],
                        Q.WP[CAM][W][S]:Q.WP[CAM][W][E],:]=np.flipud(np.fliplr(frame))
                tm=self.ET.getLatestTM()#show tracking monitor
                if not tm is None:
                    self.infoscreen[Q.WP[TM][H][S]:Q.WP[TM][H][E],
                        Q.WP[TM][W][S]:Q.WP[TM][W][E],:]=tm
                ei=self.ET.getLatestEI()#show eye image
                if not ei is None:
                    for kk in range(3):
                        self.infoscreen[Q.WP[EI][H][S]:Q.WP[EI][H][E],
                            Q.WP[EI][W][S]:Q.WP[EI][W][E],kk]=ei
                #show Eye position
                lgz=self.ET.getLatestGaze()
                self.infoscreen[Q.WP[EP][H][S]:Q.WP[EP][H][E],
                        Q.WP[EP][W][S]:Q.WP[EP][W][E],:]=self.ep0
                self.infoscreen[Q.WP[EP+1][H][S]:Q.WP[EP+1][H][E],
                        Q.WP[EP+1][W][S]:Q.WP[EP+1][W][E],:]=self.ep1
                self.infoscreen[Q.WP[MON][H][S]:Q.WP[MON][H][E],
                        Q.WP[MON][W][S]:Q.WP[MON][W][E],:]=self.monim
                if not lgz is None:
                    
                    lgz=np.copy(lgz)
                    for i in range(2):
                        if np.isnan(lgz[LX+2*i]): continue
                        c=self.ET.deg2norm(lgz[[LX+2*i,LY+2*i]])
                        woffset=np.array([Q.WP[MON][W][S],Q.WP[MON][H][S]])
                        wsize=np.array([Q.WS[MON][W],Q.WS[MON][H]])
                        c=tuple(np.int32(np.round(wsize*c+woffset)))
                        cv2.circle(self.infoscreen,c,Q.circleradius,Q.eyeclr[i],-1)
                        temp=lgz[RDIAM+1+i*3:RDIAM+4+i*3]
                        temp= (temp+Q.exyoffset)/Q.exyslope
                        temp[1]=1-temp[1]
                        if np.all(temp>0) and np.all(temp<1):
                            for j in range(2):
                                if j==0: c=[temp[0],temp[1]]
                                elif j==1: c=[temp[0],temp[2]]
                                #elif j==2: c=[1-temp[2],temp[1]]
                                woffset=np.array([Q.WP[EP+j][W][S],Q.WP[EP+j][H][S]])
                                wsize=np.array([Q.WS[EP+j][W],Q.WS[EP+j][H]])
                                c=tuple(np.int32(np.round(wsize*c+woffset)))
                                cv2.circle(self.infoscreen,c,Q.circleradius,Q.eyeclr[i],-1) 
                self.infoscreen[700:900,0:400]=0
                txt='t=%.4f, f=%d'%(getTime(),self.fcallback())
                cv2.putText(self.infoscreen,txt,(0,800),cv2.FONT_HERSHEY_TRIPLEX,1,Q.fontclr)
                #self.outV.write(self.infoscreen)
            else:
                img=np.zeros(self.infoscreen.shape,np.uint8)
                self.infoscreen=cv2.putText(img,'Press g to start tracking',
                    (int(img.shape[1]/4),int(img.shape[0]/2)),cv2.FONT_HERSHEY_TRIPLEX,1,Q.fontclr)
            cv2.imshow(Q.winname,self.infoscreen)
            key=cv2.waitKey(1)
            if key in list(map(ord,'tue')):
                self.expIsWaiting=False
            if key==ord('u'):
                self.ecallback(1)
                print('Jump to Next Block')
                self.writeLog('Jump to Next Block')
            elif key==ord('a'):
                self.AC.show(5)
            elif key==ord('e'):
                self.ecallback(-1)
                print('Jump to End')
                self.writeLog('Jump to End')
            elif key==ord('g'):
                self.updateGui=not self.updateGui
            elif key==ord('t'):
                self.updateGui=True 
        self.VC.terminate()
        self.ET.terminate()
        #self.outV.release()
        if self.loglevel>0:self.outL.close()
        if self.loglevel>1: self.outE.close()
        self.winE.close()
        
        if not self.vp==0:
            print('Press any key to terminate')
            key=cv2.waitKey(0)
        cv2.destroyAllWindows()
    ############################################################    
    # the remaining methods should be called by the Experiment class
    ############################################################
            
    def terminate(self):
        self.finished=True
        if self.isAlive(): self.join()

    def calibrate(self,ncalibpoints):
        ''' shows calibration display
            ncalibpoints - number of calibration points
            
        '''
        scale=1
        w=11;h=11
        xys=np.array([[0,0],[-w,h],[w,h],[w,-h],[-w,-h],[w,0],[-w,0],[0,h],[0,-h]])
        dur=5
        durShift=0.5
        self.writeLog('START calibration')
        print('Start Calibration')
        self.waitForKey()
        for k in range(ncalibpoints):
            self.AC.setPos(xys[k,:]*scale)
            self.writeLog('AC at %f %f'%(xys[k,0]*scale,xys[k,1]*scale))
            self.AC.show(dur)
            while self.AC.flip()==0: pass
            if k+1<ncalibpoints:#translation motion
                incxy= scale*(xys[k+1,:]-xys[k,:])/float(self.Qexp.refreshRate)/float(durShift)
                t0=getTime()
                while getTime()-t0 <durShift:
                    pos=self.AC.getPos()
                    self.AC.setPos(pos+incxy)
                    self.AC.draw(sound=False)
                    self.winE.flip()
        self.writeLog('END calibration') 
        print('Calibration finished')
        self.winE.flip()
        
    def waitForKey(self):
        print('Paused: g - show gaze, t - continue, u - jump to next block, e - jump to end, a - attention catcher')
        self.updateGui=False
        self.expIsWaiting=True
        while self.expIsWaiting:
            self.AC.flip()
        print('Resume')
    def setQexp(self,Q):
        self.Qexp=Q
        Qsave(Q,self.fn)
    def startSound(self):
        self.sound.play(loops=0)
    def stopSound(self):
        self.sound.stop()
    def getExpWin(self):
        return self.winE
    def writeShortLog(self,msg):
        if self.loglevel>1:
            self.outE.write(msg)
            self.outE.flush()
    def writeLog(self,msg,f=-1):
        if self.loglevel>0:
            tc=getTime()
            ts=self.ET.getTime()
            self.outL.write('{};{};{};MSG;{}\n'.format(tc,ts,f,msg))
            self.outL.flush()

class TestExperiment():
    def __init__(self):
        class Qexp():
            expName='test'
            monitor = asusMG279
            refreshRate=120 # [hz]
            scale=1 #multiplier
            bckgCLR= [-0.22,-0.26,-0.24]# [1 -1]
            fullscr=True
            winPos=(0,0)
            screen=1
            stimOffset=np.array([0,0])
            outputPath=''
        self.EM=ExperimentManager(ecallback=self.controlCallback,
            fcallback=lambda: 0,Qexp=Qexp,loglevel=1)
        self.showAC=False
        self.jumpToEnd=False
        self.EM.start()
    def controlCallback(self,command):
        if command==-1: 
            self.jumpToEnd=True
        
    def run(self):
        self.win.flip()
        self.EM.waitForKey()
        self.EM.terminate()
        
if __name__=='__main__':
    exp=TestExperiment()
    exp.run()
