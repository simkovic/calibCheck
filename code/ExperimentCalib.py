# -*- coding: utf-8 -*-
from psychopy import visual,core
from psychopy.misc import pix2deg, deg2pix, cm2deg
from psychopy.event import getKeys
import os,warnings
import numpy as np
warnings.filterwarnings('ignore')
from ExperimentManager import *



class Q():
    '''Experiment settings'''
    expName='calib'
    monitor = asusMG279
    scale=1 #multiplier
    bckgCLR= [-0.22,-0.26,-0.24]#gray
    fullscr=True
    winPos=(0,0)
    refreshRate=120 # hz
    screen=0 # screen 0 shows experiment manager
    stimOffset=np.array([0,0])
    ###############################################
    from os import getcwd as __getcwd
    __path = __getcwd()
    __path = __path.rstrip('code')
    from os.path import sep as __sep
    #path=path.replace('Users','Benutzer')
    inputPath=__path+"input"+__sep
    outputPath=__path+"output"+__sep
    

def infoboxCalib(pos,fn):
    '''Get metadata at the start of experiment'''
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
        self.EM=ExperimentManager(ecallback=lambda x: 0,fcallback=lambda: 0,
            Qexp=Q,loglevel=1,infobox=infoboxCalib)
        self.EM.start()
        
    def run(self):
        from PIL import Image
        self.EM.winE.flip()
        dists=[55,45,65]
        for dist in dists:
            print('Entfernung: %d, Hohe: -10'%dist)
            self.EM.ep1=np.array(Image.open(Q.inputPath+'ep%d.png'%dist))[:,:,:3]
            self.EM.calibrate(ncalibpoints=[5,9][dist==dists[0]])
        self.EM.terminate()
        core.quit()
          
            
if __name__ == '__main__':
    exp=Experiment()
    exp.run()
