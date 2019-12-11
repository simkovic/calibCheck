# -*- coding: utf-8 -*-
from psychopy import visual,core
from psychopy.misc import pix2deg, deg2pix, cm2deg
from psychopy.event import getKeys
import os,warnings
import numpy as np
warnings.filterwarnings('ignore')
from matustools.ExperimentManager import *



class Q():
    monitor = asusMG279
    scale=1 #multiplier
    bckgCLR= [-0.22,-0.26,-0.24]# [1 -1]
    fullscr=True
    winPos=(0,0)
    ###############################################
    from os import getcwd as __getcwd
    __path = __getcwd()
    __path = __path.rstrip('code')
    from os.path import sep as __sep
    #path=path.replace('Users','Benutzer')
    inputPath=__path+"input"+__sep
    outputPath=__path+"output"+__sep
    

def infoboxCalib(pos,writeInfo=False):
    import datetime
    from psychopy import gui          
    myDlg = gui.Dlg(title='VP Info',pos=pos)    
    myDlg.addField('VP ID:',0)# subject id
    today=datetime.date.today()
    myDlg.addField('Kohorte',choices=['4M','7M','10M'],initial='4M')
    myDlg.addField('ET:',choices=('smi','tob')) 
    #myDlg.addField('Entfernung',choices=[50,60,70,80,90],initial=70)
    #myDlg.addField('Höhe',choices=[-10,0,10],initial=0)
    myDlg.show()#show dialog and wait for OK or Cancel
    if myDlg.OK:
        import numpy as np
        d=myDlg.data 
        suf='Vp%dc%s%s'%(d[0],d[1],d[2])
        d.append(suf)
        print('Eyetracker: %s'%(d[2]))
        d[2]=int(d[2]=='tob')
        return d
    else:
        import sys
        sys.exit()
        
class Experiment():
    def __init__(self):
        self.f=-1;title='calib'
        self.vp,cohort,self.slot,suffix=infoboxCalib((0,0))
        #self.infpath=Qpursuit01.inputPath+title+cohort+'.sched'
        Qsave(Q,title+suffix)
        self.win=QinitDisplay(Q)
        self.EM=ExperimentManager(ofpath=Q.outputPath+title+suffix,
            ecallback=self.controlCallback,fcallback=lambda: 0,slot=self.slot,
            winE=self.win,monDist=Q.monitor.getDistance())
        self.showAC=False
        self.jumpToEnd=False
        #init vars
        self.EM.start()
    def controlCallback(self,command):
        if command==2:
            self.showAC=True
        elif command==-1: 
            self.jumpToEnd=True
        
    def run(self):
        from PIL import Image
        #self.win.flip()
        dists=[55,45,65]
        for dist in dists:
            if self.jumpToEnd: break
            print('Eyetracker: '+['smi','tobii'][self.slot]+
                ', Entfernung: %d, Hohe: -10'%dist)
            self.EM.ep1=np.array(Image.open(Q.inputPath+'ep%d.png'%dist))[:,:,:3]
            self.EM.showAC()
            self.EM.calibrate(ncalibpoints=[5,9][dist==dists[0]])
        if self.EM.terminate() and self.vp>0:
            writeInfoFile(',-1')
        else:  writeInfoFile(',-1')
        self.win.close()
        core.quit()
          
            
if __name__ == '__main__':
    exp=Experiment()
    exp.run()
