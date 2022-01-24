import os,pystan,pickle
import numpy as np
import pylab as plt
from matusplotlib import *
from scipy.stats import scoreatpercentile as sap
MD=70 #monitor distance used to compute deg visual angle in the output files
#position of calibration points
CTRUE=np.array([[0,0],[-11,11],[11,11],[11,-11],[-11,-11],[11,0],[-11,0],[0,11],[0,-11]]) # true calibration locations in degrees
#CTRUE=CTRUEDEG/180*np.pi*MD # true calibartion locations in cm
SEED=5 # the seed of random number generator was fixed to make the analyses replicable
TC,TS,F,LX,LY,RX,RY,BX,BY,LD=range(10); RD=12;BD=15
DPI=500 #figure resolution
DPATH='data'+os.path.sep  
##########################################################################
# DATA LOADING ROUTINES
##########################################################################
def readCalibGaze(fn):
    ''' reads calibration gaze data
        fn - file with calibration gaze data
        returns list with raw eyetracking data for each calibration
            point, first row of each list element provides the
            position of the calibration point
    '''
    calib=[]
    f=open(fn,'r')
    decalib=True
    calibon=False
    decalibmat=np.array(4*[[0,1]])
    for line in f.readlines():
        if line[:2]=='##':
            pre,mat=line.rsplit(':')
            if pre=='## Python Version':decalib=False
            elif pre=='## Calib Matrix' and decalib:
                decalibmat=parseCalibMatrixString(mat)
            continue
        line=line.rstrip('\n')
        line=line.rstrip('\r')
        words=line.rsplit(';')
        words=list(filter(len,words))
        if len(words)<3: print(words)
        if words[3]=='MSG':
            if words[4][:6]=='AC at ':
                temp=words[4].rsplit(' ')
                calibon=True
                calib.append([[float(temp[2]),float(temp[3])]+13*[np.nan]])
            elif words[4][:3]=='END': calibon=False    
        elif calibon:
            vals=np.float64(words)[:15]
            if len(vals)<9: continue
            vals[3:7]=decalibmat[:,0]+vals[3:7]*decalibmat[:,1]
            #if decalib: 
            #    print(vals[3:7],decalibmat[:,0],decalibmat[:,1])
            #    bla

            calib[-1].append(vals)
    for i in range(len(calib)):
        if len(calib[i][-1])<9: calib[i].pop()
        calib[i]=np.array(calib[i])
        if np.ndim(calib[i])<2: bla
    f.close() 
    return calib
    
def getCMsingle(c,ctrue):
    ''' computes linear calibration 
        calibData - calibration data returned by readCalibGaze 
        omit - list contains label of
            calibration point to be ommited
            there were 9 calibration points, labeled from 
            left to right, top to bottom
        returns 
            - 1D array with calibration coefficients in format 
            [offset X, slope X, offset Y, slope Y]
            - residual
            - array with omitted calibration points
            - (median) gaze location on each of calibration points
    '''
    coef=[np.nan,np.nan,np.nan,np.nan]
    resid=np.nan
    assert(c.shape[0]==9)
    assert(c.shape[1]==2)
    assert(np.all(np.isnan(c[:,0])==np.isnan(c[:,1])))
    sel=~np.isnan(c[:,0])
    assert(sel.sum()>=3)
    temp=0
    for k in range(2):
        x=np.column_stack([np.ones(sel.sum()),ctrue[sel,k]])
        res =np.linalg.lstsq(x,c[sel,k],rcond=None)
        coef[k*2:(k+1)*2]=res[0]
        temp+=res[1][0]
    resid=temp**0.5/sel.sum()
    assert(np.all(np.isnan(coef))==np.any(np.isnan(coef)))
    assert(np.isnan(resid)==np.any(np.isnan(coef))) 
    return coef,resid,c

def plotCMsingle(calibData,eye,cm=None,ofn=None,lim=[30,20],c=[22,11.5]):
    ''' plots calibration data
        calibData - calibration data returned by readCalibGaze 
        cm - linear calibration
        ofn - path string, if provided saves the figure to ofn+'.png' 
    '''
    X=0;Y=1
    clrs=['0.25','r','m','c','y','b','g','k','0.5']
    grid=[[[0,0],[c[X],0]],[[0,0],[-c[X],0]],[[0,0],[0,-c[Y]]],[[0,0],[0,c[Y]]],
         [[c[X],c[Y]],[c[X],0]],[[c[X],c[Y]],[0,c[Y]]],
         [[-c[X],c[Y]],[-c[X],0]],[[-c[X],c[Y]],[0,c[Y]]],
         [[c[X],-c[Y]],[c[X],0]],[[c[X],-c[Y]],[0,-c[Y]]],
         [[-c[X],-c[Y]],[-c[X],0]],[[-c[X],-c[Y]],[0,-c[Y]]]]
    #print(calibMatrix)
    for b in range(len(calibData)):
        D=np.array(calibData[b])
        ax=plt.gca()
        plt.plot(D[:,LX+2*eye],D[:,LY+2*eye],'.',zorder=-2,color=clrs[b],mew=0)
        plt.plot(np.nanmedian(D[:,LX+2*eye]),np.nanmedian(D[:,LY+2*eye]),'d',
                 color=clrs[b],mec='0.75',mew=1,zorder=2)  
        if not cm is None:
            for line in grid:
                lout=np.zeros((2,2))
                for dd in range(2):
                    for cc in range(2):
                        lout[dd,cc]=(-cm[0][2*cc]+line[dd][cc])/cm[0][2*cc+1]
                #print(line,lout)
                plt.plot(lout[:,0],lout[:,1],'k',alpha=0.1)
        plt.xlim([-lim[X],lim[X]])
        plt.ylim([-lim[Y],lim[Y]])
        ax.set_aspect(1)
    if not ofn==None:plt.savefig(ofn+'.png',dpi=DPI)
       
def checkFiles(plot=False):
    '''checks the congruency between meta-data and log files'''
    opath=os.getcwd()[:-4]+'output'+os.path.sep+'anon'+os.path.sep
    vpinfo=np.int32(np.loadtxt(opath+'vpinfo.res',delimiter=','))

    #assert(np.all(vpinfo[:,4]>=30*3.5))
    #assert(np.all(info[:,4]<=30*11))
    if plot:
        plt.hist(vpinfo[:,4],bins=np.linspace(30*3.5,30*11,17))
        plt.gca().set_xticks(np.linspace(30*4,30*11,9))
        plt.xlabel('age in days');
    print('Checking for surplus files missing from metadata file')
    fns=list(filter(lambda x: x[-4:]=='.log',os.listdir(opath)))
    fns=np.sort(fns)
    vpns=vpinfo[:,[0,1]]
    for fn in fns:
        temp=fn.rsplit('.')[0].rsplit('Vp')[1].rsplit('c')
        vp=int(temp[0])
        m=int(temp[1][:-4])
        temp=(np.logical_and(vpns[:,0]==vp, vpns[:,1]==m)).nonzero()[0]
        if not temp.size: print(fn+' surplus file')         
    print('Checking for missing files')
    fns=[]
    for i in range(vpinfo.shape[0]):
        fn='calibVp%03dc%dM'%(vpinfo[i,0],vpinfo[i,1])
        if not os.path.isfile(opath+fn+'tob.log') and not os.path.isfile(opath+fn+'smi.log'): 
            print(opath+fn)
        else: fns.append(fn)
    return fns
  
def loadCalibrationData(fns):
    '''loads all data into a single object '''
    opath=os.getcwd()[:-4]+'output'+os.path.sep+'anon'+os.path.sep
    vpinfo=np.int32(np.loadtxt(opath+'vpinfo.res',delimiter=','))
    D=[]
    for k in range(len(fns)):
        D.append([vpinfo[k,:]])
        for s in range(2):
            suf=['tob','smi'][s]
            try: 
                #print(fns[k]+suf+'.log')
                d=readCalibGaze(opath+fns[k]+suf+'.log')
                D[-1].append(d)
                assert(len(d)<20)
            except FileNotFoundError: D[-1].append([]) 
    # check location of calibration points
    for s in range(2):
        for n in range(len(D)):
            for m in range(3):
                d=[D[n][s+1][:9],D[n][s+1][9:14],D[n][s+1][14:19]][m]
                for i in range(len(d)):
                    assert(np.allclose(d[i][0,:2],CTRUE[i,:]))
            for p in range(len(D[n][s+1])):
                D[n][s+1][p]= D[n][s+1][p][1:,:]
    # select discard data columns which are not needed
    for s in range(2):
        for n in range(len(D)):
            for p in range(len(D[n][s+1])):
                temp=np.zeros((D[n][s+1][p].shape[0],18))*np.nan
                temp[:,:7]=D[n][s+1][p][:,:7] #TC,TS,F,LX,LY,RX,RY
                temp[:,BX]=np.nanmean(temp[:,[LX,RX]],axis=1)
                temp[:,BY]=np.nanmean(temp[:,[LY,RY]],axis=1)
                temp[:,LD:(LD+3)]=D[n][s+1][p][:,9:12]
                temp[:,RD:(RD+3)]=D[n][s+1][p][:,12:15]
                a=np.array([temp[:,LD:(LD+3)],temp[:,RD:(RD+3)]])
                temp[:,BD:(BD+3)]=np.nanmean(a,axis=0)
                D[n][s+1][p]=temp
    age=[]
    for i in range(len(D)):
        age.append(D[i][0][4])
    np.save(DPATH+'age',age)
    return D
    
#########################################################################################
# DATA PRE-PROCESSING ROUTINES
#########################################################################################

from scipy.interpolate import interp1d
from scipy.stats import norm


def computeState(isX,t,minGapDur=0,minDur=0,maxDur=np.inf):
    ''' generic function that determines event start and end
        isX - 1d array, time series with one element for each
            gaze data point, 1 indicates the event is on, 0 - off
        md - minimum event duration
        returns
            list with tuples with start and end for each
                event (values in frames)
            timeseries analogue to isFix but the values
                correspond to the list
    '''
    if isX.sum()==0: return np.int32(isX),[]
    s = (isX & ~np.roll(isX,1)).nonzero()[0].tolist()
    e= (np.roll(isX,1) & ~isX).nonzero()[0].tolist()
    if len(s)==0 and len(e)==0: s=[0];e=[isX.size-1]
    if s[-1]>e[-1]:e.append(isX.shape[0]-1)
    if s[0]>e[0]:s.insert(0,0)
    if len(s)!=len(e): print ('invalid on/off');raise TypeError
    f=1
    while f<len(s):
        if t[s[f]]-t[e[f-1]]<minGapDur:
            isX[e[f-1]:(s[f]+1)]=True
            e[f-1]=e.pop(f);s.pop(f)
        else: f+=1
    states=[]
    for f in range(len(s)):    
        if  t[e[f]]-t[s[f]]<minDur or t[e[f]]-t[s[f]]>maxDur:
            isX[s[f]:e[f]]=False
        else: states.append([s[f],e[f]-1])
    #fixations=np.array(fixations)
    return isX,states

def filterGaussian(told,xold,tnew,sigma=0.05,filterInterval=None,ignoreNanInterval=0):
    ''' told and tnew - time series with time in seconds
        sigma - standard deviation of the gaussian filter in seconds
        filterInterval - width of the filter interval in seconds, default 8 times sigma
    '''
    if filterInterval is None: filterInterval=sigma*6
    assert(filterInterval>0 and ignoreNanInterval>=0 and sigma>0)
    xnew=np.zeros(tnew.shape)
    nns=np.isnan(xold)
    for i in range(tnew.size):
        tc=told-tnew[i]
        sel=np.abs(tc)<filterInterval/2
        if ignoreNanInterval>0: 
            outside=np.abs(tc)>ignoreNanInterval/2
            nanoutside=np.logical_and(outside, nns)
            sel=np.logical_and(sel, ~nanoutside)
        if sel.sum()>1:
            w= norm.pdf(tc[sel],0,sigma)
            xnew[i]=w.dot(xold[sel])/w.sum()
        else: xnew[i]=np.nan
    return xnew

def intersection(intv,intvs):
    intvsout=[]
    for i in intvs:
        if i[0]<intv[1] and i[1]>intv[0]:
            intvsout.append([max(i[0],intv[0]), min(i[1],intv[1])])
    return intvsout
    
def eucldunits(ctrue,chat,dsd,dva):
    ''' translate to cm compute euclidean distance and translate back 
        to units 
        ctrue - calib. target position
        chat - predicted position
        dsd - screen-to-eye distance
        dva - type of unit computation'''
    AX=np.newaxis
    if dva<=0 or dva==4: 
        tempcm=np.sqrt(np.sum(np.square(ctrue-chat),axis=2))
        if dva<0: return tempcm
        ctruecm=ctrue+dsd[:,:,:2];chatcm=chat+dsd[:,:,:2];
        dctrue2=np.sum(np.square(ctruecm),axis=2)+np.square(dsd[:,:,2])
        dchat2=np.sum(np.square(chatcm),axis=2)+np.square(dsd[:,:,2])
        temp=np.arccos((dctrue2+dchat2-np.square(tempcm))/2/
            np.sqrt(dctrue2*dchat2))/np.pi*180
    elif dva==1 or dva==2 or dva==3 or dva>=5:
        if dva==1:ddd=np.array([0,0,57.5])[AX,AX,AX,:] 
        elif dva==2:ddd=np.nanmean(dsd,1)[:,AX,AX,:]
        elif dva==3:ddd=dsd[:,:,AX,:]
        elif dva>=5: ddd=np.array([0,0,[57.5,47.5,67.5][m+1]])[AX,AX,AX,:] 
        
        chatcm=np.tan(chat/180*np.pi)*ddd[:,:,:,2]+ddd[:,:,0,:2]
        ctruecm=np.tan(ctrue/180*np.pi)*ddd[:,:,:,2]+ddd[:,:,0,:2]
        tempcm2=np.sum(np.square(ctruecm-chatcm),axis=2)
        dctrue2=np.sum(np.square(ctruecm),axis=2)+np.square(ddd[:,:,0,2])
        dchat2=np.sum(np.square(chatcm),axis=2)+np.square(ddd[:,:,0,2])
        temp=np.arccos((dctrue2+dchat2-tempcm2)/2/np.sqrt(dctrue2*dchat2))
        temp=temp/np.pi*180
        #elif units=='cm': temp=np.sqrt(tempcm2)
    return temp

def chunits(inn,dva,hvd=None,frd=None):
    ''' hvd - horizontal or vertical screen-to-eye distance 
        frd - frontal screen-to-eye distance
    '''
    cm=inn/180*np.pi*MD
    if dva==0: x=cm 
    elif dva==1: x=np.arctan(cm/57.5)/np.pi*180
    elif dva==2: 
        x=np.arctan((cm-np.nanmean(hvd,0))/
            np.nanmean(frd,0))/np.pi*180    
    elif dva==3: 
        x=np.arctan((cm-hvd)/frd)/np.pi*180
    elif dva==4: 
        x=57.5*(cm-hvd)/frd
    elif dva==5 or dva==6 or dva==7:
        x=np.arctan(cm/[57.5,47.5,67.5][dva-5])/np.pi*180
    return x    
    
def extractFixations(inG,eye,thvel=10,hz=60,minDur=0.3,dva=0): 
    AX=np.newaxis
    G=np.concatenate(inG,0)
    C=[]
    for i in range(len(inG)):
        C.append(np.ones((inG[i].shape[0],2))*np.array(CTRUE[i]))
    C=np.concatenate(C,0)
    res=np.ones((9,7))*np.nan
    if G.shape[0]<3: return res 
    t=np.arange(G[0,0],G[-1,0],1/hz)[1:-1]
    d=[t];
    G[:,LD+3*eye:LD+3+3*eye]/=10 # mm to cm
    G[:,LD+1+3*eye]-=16.45;G[:,LD+2+3*eye]+=2.5 # origin at screen center
    for i in range(2):
        ii=[LX+2*eye,LY+2*eye][i]
        sel,discard=computeState(np.isnan(G[:,ii]),G[:,0],maxDur=0.05)
        x=chunits(G[~sel,ii],dva,hvd=G[~sel,LD+i+3*eye],frd=G[~sel,LD+2+3*eye])
        C[~sel,i]=chunits(C[~sel,i],dva,hvd=G[~sel,LD+i+3*eye],frd=G[~sel,LD+2+3*eye])
        C[sel,i]=np.nan
        d.append(filterGaussian(G[~sel,0],x,t,sigma=0.1))
        #out=filterGaussian(G[~sel,0],G[~sel,i],t,sigma=0.1,ignoreNanInterval=0.05)
    d=np.array(d).T
    #np.save('e',d)
    vel=np.sqrt(np.square(np.diff(d[:,1:3],axis=0)).sum(1))*hz
    temp=d[1:,0]/2+d[:-1,0]/2
    a,fix=computeState(vel<thvel,d[:,0],minDur=minDur,minGapDur=0.03)
    #np.save('f',fix)
    if len(fix)==0: return res
    for p in range(len(inG)):
        if inG[p].shape[0]==0:continue
        else: intvs=intersection(inG[p][[1,-1],0],t[np.array(fix)])
        if len(intvs)>0:
            se=intvs[np.argmax(np.diff(intvs,1))]
            s=np.logical_and(se[0]<=t[1:],se[0]>=t[:-1]).nonzero()[0][0]+1
            #print(G[-1,0]-G[0,0],se[0]-G[0,0],se[1]-G[0,0]) 
            e=np.logical_and(se[1]<=t[1:],se[1]>=t[:-1]).nonzero()[-1][0]+1 
            if e-s>0.1*hz: 
                res[p,:2]=np.nanmean(d[s:e,1:],0)
                ss=np.logical_and(se[0]<=G[1:,0],se[0]>=G[:-1,0]).nonzero()[0][0]+1
                ee=np.logical_and(se[1]<=G[1:,0],se[1]>=G[:-1,0]).nonzero()[-1][0]+1 
                res[p,4:]=np.nanmean(G[ss:ee,LD+3*eye:RD+3*eye],0)
                res[p,2:4]=np.nanmean(C[ss:ee,:],0)
    #np.save('g',res)
    return res  
    

def dpworker(G,thacc,thvel,minDur,dva):
    '''
       G - gaze data
       minDur - minimum fixation duration
       thvel - velocity threshold for fixation identification
       thacc - accuracy threshold for discarding calibration location with poor data
       dva - method for computation of degrees of visual angle, 
            0- cm; 
            1- visual angle based on constant head distance of 57.5; 
            2- visual angle based on empirical head distance: session mean
            3- visual angle based on empirical head distance: calibration location mean
            4- cm computed from angle as in (3) and constant distance 57.5
            5,6,7 - visual angle based on nominal starting distance (5=57.5,6=47.5,7=67.5)
       retuns - gaze data at 60 Hz as ndarray with dimensions: 1. eye-tracking device 
            2. calibration session 3. eye (L,R,B) 4. calibration location (see CTRUE)
            5. horizontal gaze, vertical gaze, eye distance
            - information about exclusion of calibration locations as ndarray
    '''
    MINVALIDCL=[4,3,3]
    R=np.zeros((2,3,3,9,7))*np.nan
    included=np.zeros((2,3,3,3,7),dtype=np.int32)
    coh=[7,7,7,7,0,7,7,1,7,7,2][G[0][1]]
    c=np.zeros((9,3))*np.nan
    
    for s in range(2):
        for m in range(3):
            d=[G[1+s][:9],G[1+s][9:14],G[1+s][14:19]][m]              
            for i in range(3):
                included[s,coh,m,i,6]=1
                if len(d)>2:
                    excl=False
                    check=np.zeros(9,dtype=np.bool)
                    for p in range(min(check.size,len(d))):
                        check[p]=np.any(~np.isnan(d[p][:,[LX+2*i,LY+2*i]]))
                    if check.sum()>=MINVALIDCL[m]: 
                        included[s,coh,m,i,1]=1
                        included[s,coh,m,i,0]=check.sum()
                    #np.save('d',d)
                    if dva==5: dva2=dva+m
                    else:dva2=dva
                    c=extractFixations(d,i,thvel=thvel,minDur=minDur,dva=dva2)
                    if np.isnan(c[:,0]).sum()<=(9-MINVALIDCL[m]):
                        included[s,coh,m,i,3]=1
                        included[s,coh,m,i,2]=(~np.isnan(c[:,0])).sum()
                        cm=list(getCMsingle(c[:,:2].copy(),c[:,2:4].copy()))
                        co=cm.copy()
                        while np.isnan(co[2][:,0]).sum()<(9-MINVALIDCL[m]) and (co[1]>thacc):
                            ci=co.copy()
                            for k in range(9):
                                if np.isnan(co[2][k,0]): continue
                                cg=co[2].copy();cg[k,:]=np.nan
                                res=list(getCMsingle(cg,c[:,2:4].copy()))
                                if res[1]<ci[1]:ci=res
                            if ci[1]<co[1]: co=ci
                            else: break
                        if co[1]<cm[1]:cm=co
                    else: excl=True
                else: excl=True
                if excl or cm[1]>thacc or cm[0][1]<0 or cm[0][3]<0:
                    cm=[np.nan*np.ones(4),np.nan,np.zeros((9,2))*np.nan]
                if not np.isnan(cm[1]): 
                    included[s,coh,m,i,5]=1
                    included[s,coh,m,i,4]=(~np.isnan(cm[2][:,0])).sum()
                R[s,m,i,:,:2]=cm[2]
                R[s,m,i,:,2:]=c[:,2:]
    return R,included 
    
def dataPreprocessing(D,fn,thacc=0.5,thvel=10,minDur=0.3,dva=0,verbose=False,ncpu=8):  
    ''' the processing code is in dpworker(), this is just a wrapper
        for parallel application of dpworker
    '''              
    from multiprocessing import Pool
    pool=Pool(ncpu)
    res=[]
    for n in range(len(D)):
        temp=pool.apply_async(dpworker,[D[n],thacc,thvel,minDur,dva])
        res.append(temp)
    pool.close()  
    from time import time,sleep
    from matusplotlib import printProgress
    tot=len(D)
    t0=time();prevdone=-1
    while True:
        done=0
        for r in res:
            done+=int(r.ready())
        if done>prevdone:
            printProgress(done,tot,time()-t0,' running simulation\t')
            prevdone=done
        sleep(1)
        if done==tot: break
    ds=np.zeros((2,len(D),3,3,9,7))*np.nan
    included=np.zeros((2,3,3,3,7),dtype=np.int32)
    for n in range(len(D)):
        #print(res[n].get())
        ds[:,n,:,:,:,:],incl=res[n].get()
        included+=incl
    assert(not np.any(np.isnan(ds[:,:,:,:,:,0]).sum(4)==7))
    assert(not np.any(np.isnan(ds[:,:,:,:,:,0]).sum(4)==8))
    print(included)
    np.save(DPATH+fn+'incl',included)
    np.save(DPATH+fn,ds)
##########################################################################

def tableSample(fn):
    ''' prints code for latex table to console
        table includes sample exclusion description '''
    I=np.load(DPATH+fn+'.npy')
    left=[]
    for a in [4,7,10]:
        for e in ['Tob','Smi']:
            for d in [45,55,65]:
                if e=='Tob' and d==45: aa=a
                #elif e=='Tob' and d==55: aa='$%d$'%[86,65,53][int((a-4)/3)]
                else: aa=''
                if d==45: ee=e
                else: ee=''
                left.append([aa,ee,d])
    left=list(np.array(left,dtype=object).T)
    res=[]
    for g in range(2):
        for e in range(3):
            for i in [0,2,4]:
                if i==0 and e==0 and g==1: 
                    for l in left: 
                        res.append(l)
                temp=[]
                for coh in range(3):
                    for dev in range(2):
                        for dist in [1,0,2]:
                            if g==1: temp.append((I[dev,coh,dist,e,i]/I[dev,coh,dist,e,i+1])/[9,5,5][dist]*100)
                            elif g==0: temp.append(I[dev,coh,dist,e,i+1]/I[dev,coh,dist,e,-1]*100)
                res.append(np.array(temp,dtype=object))
    res=np.array(res,dtype=object).T
    top=[]
    for p in range(2):
        for e in ['L','R','B']:
            if p==1 and e=='L': top.extend(['$m$','ET','$d$'])    
            for i in range(3): top.append(e+str(i+1))
    res= np.array([top]+list(res),dtype=object)
    ndarray2latextable(res,decim=0,hline=[0,3,6,9,12,15],
        nl=0,vline=[2,5,8,9,10,11,14,17]) 
def figureSample(fn,dev=0):
    I=np.load(DPATH+fn+'.npy')
    
    CLRS=['0.3','0.5','0.7']#['k','gray','w']
    xtcs=[]
    figure(size=3,aspect=0.6,dpi=DPI)
    for tp in range(2):
        for coh in range(3):
            ax=subplot(2,3,3*tp+coh+1)
            ax.set_axisbelow(True)
            plt.grid(True,axis='y')
            for dist in range(3):
                plt.text(dist*4+1.7, 5,['55','45','65'][dist],horizontalalignment='center',size=10)
                for e in range(3):
                    xtcs.append(dist*4+e+0.5)
                    for i in range(3):
                        if tp==0: y=I[dev,coh,dist,e,2*i+1]
                        else: y=I[dev,coh,dist,e,2*i]/I[dev,coh,dist,e,2*i+1]/[9,5,5][dist]*100
                        plt.bar(dist*4+e,y,width=0.9-i*0.2,color=CLRS[i],align='edge')
            ax.set_xticks(xtcs)
            if tp==0:plt.ylim([0,90])
            else: plt.ylim([0,100])
            if coh: ax.set_yticklabels([])
            else: plt.ylabel(['Number of included\nparticipants','Percentage of included\ncalibration locations'][tp])
            if tp==0:plt.title(['4 months','7 months','10 months'][coh])
            ax.set_xticklabels(['L','R','B']*int(len(xtcs)/3))
            
    #plt.show()
    plt.savefig('../publication/figs/%s%d.png'%(fn,dev),bbox_inches='tight',dpi=DPI)       
                
def sampleDescr(dva):
    ds=np.load(DPATH+f'dsFixTh1_0dva{dva}.npy')
    R=np.zeros((3,4))*np.nan
    for dev in range(2):
        m=0
        for eye in range(2):
            maxcp=5#[9,5][int(m>0)]
            y=ds[dev,:,m,eye,:maxcp,:2]
            c=ds[dev,:,m,eye,:maxcp,2:4]
            sel=~(np.isnan(y[:,:,0]).sum(1)>2)
            r=eucldunits(c[sel,:],y[sel,:,:],ds[dev,sel,m,eye,:maxcp,4:7],dva)
            R[m,dev*2+eye]=np.nanmean(r)
    ndarray2latextable(R,decim=1,hline=[1],nl=1);
            
 
def trainLC(suf,m,dev,eye=2,docompile=True,short=False):
    ''' train linear corection, estimate the offset and slope pars
        suf - suffix of the target ds, the file provides the data
        m - which session to use for training, i-th session is m=i-1
        dev - device: 0=Tobii, 1=SMI
        eye - 0= left, 1=right, 2=binocular average
        docompile - if true compiles the stan model
        short - if true only ,,2s'' version is evaluated
     '''
    models=[]
    data='''data {
        int<lower=0> N;
        vector[2] y[N,5];
        vector[2] c[N,5];
        real age[N];
        real dist[N,5];'''
    for k in range(4):
        for i in range(4):
            r=['','corr_matrix[2] ry;'][int(i/2)]
            yvar=['diag_matrix(sy)','quad_form_diag(ry,sy)'][int(i/2)]
            odim=[3,4][i%2]
            os=['','[N]'][int(k>0)]
            pars=f'''
            }}parameters{{
                vector[{odim}] o{os};
                vector<lower=0,upper=100>[2] sy;
                {r}
                '''
            temp=f'''vector[{odim}] mo;
            vector<lower=0,upper=100>[{odim}] so;
            corr_matrix[{odim}] ro;'''
            temp2=f'''
            vector<lower=-100,upper=100>[{odim}] aay;
            vector<lower=-100,upper=100>[2] ady;
            '''
            if k==2: pars+=temp
            elif k==3: pars+= temp+temp2
            os=['','[n]'][int(k>0)]
            yslope=[f'o{os}[3]', f'tail(o{os},2).'][i%2]
            pag=['','+aay*age[n]'][int(k==3)]
            pred=['','+ady*dist[n,p]'][int(k==3)]
            temp=['',f'o[n]~multi_normal(mo{pag},quad_form_diag(ro,so));'][int(k>1)]
            so=['','so~cauchy(0,20);'][int(k>1)]
            model=f'''
        }} model {{
            sy~cauchy(0,20);
            {so}
            for (n in 1:N){{ {temp}
            for (p in 1:5){{
                if (! is_nan(y[n,p][1]))
                    c[n,p]~multi_normal(head(o{os},2)+{yslope}*y[n,p]{pred},{yvar});       
        }}}}}}'''
            models.append(data+pars+model)
    sms=[]
    print('Compiling models')
    assert(len(models)==16)
    iss=range(len(models))
    if docompile:
        for i in iss:
            sms.append(pystan.StanModel(model_code=models[i]))
            with open(DPATH+f'sm{i}.pkl', 'wb') as f: pickle.dump(sms[-1], f)
    sms=[]
    for i in range(len(models)):
        with open(DPATH+f'sm{i}.pkl', 'rb') as f: temp=pickle.load(f)
        sms.append(temp)
    ds=np.load(DPATH+f'ds{suf}.npy')
    print('Compilation Finished, Fitting models')
    y=ds[dev,:,m,eye,:5,:2]
    c=ds[dev,:,m,eye,:5,2:4]
    assert(np.all(np.isnan(y[:,:,0]).sum(1)<6))
    sel=~(np.isnan(y[:,:,0]).sum(1)>2)
    age=np.load(DPATH+'age.npy')[sel]
    age= age/30-7
    dist=ds[dev,:,m,eye,:5,6]
    dist=(dist[sel,:]-57.5)/10
    #c=ds[dev,:,m,eye,:5,2:4]
    dat={'y':y[sel,:,:],'N':sel.sum(),'c':c[sel,:],'age':age,'dist':dist}
    if short: doi=[1,5,9]
    else: doi=range(12) 
    for i in doi:
        print(dev,eye,m,i)
        fit = sms[i].sampling(data=dat,iter=10000,
            chains=6,thin=10,warmup=5000,n_jobs=6,seed=SEED)
        saveStanFit(fit,DPATH+f'd{dev}e{eye}m{m}i{i}{suf}')   
                  
def validateLC(fn,mcal=1,mval=0,dev=0,novelLocations=False,dva=0,units='deg',
    plot=0,pref='a',legend=False):
    ''' prints code for latex table to console
        table shows accuracy estimates
        fn, m and dev determine the input file - see output of computePA
        dva and units - determine the unit type
        plot - 1 plot 2s only, 2 plots all versions
        pref - prefix for the figure output file
        legend and ylim - figure parameters
    '''   
    qntls=[50,2.5,97.5,25,75]
    left=[]
    for i in range(4): # eye
        ii=['L','R','DALR','PALR'][i]
        ee=['Tobii','Smi'][dev]
        left.append([ee,ii])

    left=list(np.array(left).T)
    top=['device','eye']
    for i in ['p','u','h','a']:
        for j in ['1s','2s','1sc','2sc']:
            top.append(i+j)
    
    for i in range(len(top)-2):
        left.append(['']*len(left[0]))
    res=np.array([top]+list(np.array(left).T),dtype='U256')
    resout=np.array(np.nan*np.zeros((list(res.shape)+[7])),dtype=object)
    resout[:,:,0]=res
    ds=np.load(DPATH+f'ds{fn}dva{dva}.npy')
    AX=np.newaxis
    lracc=np.zeros((2,203))*np.nan
    for i in range(16):
        lrchat=np.zeros((2,203,9,7))*np.nan
        for eye in range(4):
            if eye<3:
                #with open(DPATH+f'sm{i}.pkl','rb') as f: sm=pickle.load(f)
                try:w=loadStanFit(DPATH+f'd{dev}e{eye}m{mcal}i{i}{fn}dva{dva}')
                except: continue
                inds=(w['rhat'][0,:-1]>1.1).nonzero()[0]
                if len(inds)>0:
                    print(dev,eye,top[2+i],'FAILED',w['nms'][inds],w['rhat'][0,:-1][inds])
                    if eye==0: lchat=np.nan
                    elif eye==1: rchat=np.nan
                    res[1+eye,2+i]='-';continue
                else: print(dev,eye,i,top[2+i],'CONVERGED') 
                #print(f'd{dev}e{eye}m{m}i{i}',len(inds))
                o=np.mean(w['o'],axis=0)
                if o.ndim==1: o=o[AX,:]
                slope= o[:,AX,2:] 
                if slope.ndim==2: slope=slope[:,:,AX]
                #sel=~np.all(np.isnan(ds[dev,:,m+1,eye,:5,0]),axis=1)
                sel=~(np.isnan(ds[dev,:,mcal,eye,:5,0]).sum(1)>2)
                age=np.load(DPATH+'age.npy')[sel]
                age= age/30-7
                y=ds[dev,sel,mval,eye,:,:2]
                ctrue=ds[dev,sel,mval,eye,:,2:4]
                dist=ds[dev,:,mval,eye,:,6]
                dist=(dist[sel,:]-57.5)/10
                chat= o[:,AX,:2]+slope*y
                if 'ady' in w.keys(): 
                    ady=np.mean(w['ady'],0)
                    chat+= (ady[AX,AX,:]* dist[:,:,AX])
                if eye<2: 
                    lrchat[eye,sel,:,:2]=chat
                    lrchat[eye,sel,:,2:4]=ctrue
                    lrchat[eye,sel,:,4:]=ds[dev,sel,mval,eye,:,4:]
            elif eye==3: 
                chat=np.nanmean(lrchat[:,:,:,:2],axis=0)
                sel=~np.all(np.isnan(chat[:,:,0]),axis=1)
                chat=chat[sel,:,:]
                ctrue=np.nanmean(lrchat[:,sel,:,2:4],axis=0)
            if np.all(np.isnan(chat)):
                res[1+eye,2+i]='-';continue
            if eye<3: dsd=ds[dev,sel,mval,eye,:,4:7]
            elif eye==3: dsd=np.nanmean(lrchat[:,sel,:,4:],axis=0)
            temp=eucldunits(ctrue,chat,dsd,dva)
            if novelLocations: temp=np.nanmean(temp[:,5:],axis=1)
            else: temp=np.nanmean(temp[:,:5],axis=1)
            if eye<2: lracc[eye,sel]=temp
            sel=np.isnan(temp) 
            mm=np.nanmean(temp);se=np.sqrt(np.nanvar(temp)/(~np.isnan(temp)).sum())
            res[1+eye,2+i]='\\textbf{%.2f} (%.2f,%.2f)'%(mm,mm-1.96*se,mm+1.96*se)
            resout[1+eye,2+i,5]=mm; resout[1+eye,2+i,6]=se 
            resout[1+eye,2+i,:5]=list(map(lambda x: sap(temp[~sel],x),qntls))
        #np.save('lracc',lracc)
        temp=lracc[1,:]-lracc[0,:]
        sel=np.isnan(temp) 
        mm=np.nanmean(temp);se=np.sqrt(np.nanvar(temp)/(~np.isnan(temp)).sum())
        #print(top[2+i],'acc right eye - acc left eye: m= %.3f, 95p CI (%.3f,%.3f), r=%.3f'%(mm,mm-1.96*se,mm+1.96*se, np.corrcoef(lracc[1,~sel],lracc[0,~sel])[0,1]))      
    sel=resout==''
    resout[sel]=np.nan
    if plot>0:
        suf=['','N'][int(novelLocations)]
        ffn=pref+['Tob','Smi'][dev]+f'{mcal}{mval}Dva{dva}{units}{suf}'
        plt.close('all')
        if plot==2: figure(size=2,aspect=0.8,dpi=DPI)
        else: figure(size=4,aspect=0.5,dpi=DPI)
        figureAccuracy(resout,short=plot==2,legend=legend,dev=dev)
        plt.savefig('../publication/figs/%s.png'%ffn,bbox_inches='tight',dpi=DPI) 
    else: 
        ndarray2latextable(res.T,decim=0,hline=[1,5,9,13],nl=1); 
        return resout

def figureAcc():
    plt.close('all')
    figure(size=3,aspect=1.1,dpi=DPI)
    mps=['complete pooling','no pooling','hierarchical', 'hier. with predictors']
    for i in range(2):
        subplot(2,1,1+i)
        res=validateLC('FixTh1_0',mcal=0,mval=2,dva=0,dev=i,plot=0)
        
        for k in range(3):
            plt.plot([k*2.8+2.6,k*2.8+2.6],[0,7],'k',lw=.5)
            plt.text(k*2.8+1.2,0.15,mps[k],ha='center',size=7)
        figureAccuracy(res,dev=i,ylim=[0,[5,6][i]])
        subplotAnnotate()
        plt.xlabel(None)
    plt.xlabel('LC version')   
    plt.savefig('../publication/figs/acc.png',bbox_inches='tight',dpi=DPI) 

    

def figureAccuracy(res,short=False,legend=False,dev=0,ylim=[0,5]):
    ''' plots accuracy of estimates'''
    if short: res=res[:,[0,1,3,7,11,15],:]
    clrs=['g','c','b','y']
    xs= np.arange(res.shape[1]-2)*0.7
    lbls=['prediction avg.','data avg.','right eye','left eye']
    k=0
    #for k in range(2):
    #ax=plt.subplot(1,2,1+k)
    handles=[]
    for col in range(1,5)[::-1]:
        ofs=(col-1.5)/1.5*0.2
        x=np.array([xs+ofs,xs+ofs])
        out=plt.plot(x,res[col+k*4,2:,1:3].T,color=clrs[col-1],alpha=0.7)
        plt.plot(x,res[col+k*4,2:,3:5].T,color=clrs[col-1],lw=3,solid_capstyle='round')
        plt.plot(xs+ofs,res[col+k*4,2:,0],mfc=clrs[col-1],
            mec=clrs[col-1],marker='d',lw=0) 
        #if short:
        plt.plot(xs+ofs-0.15,res[col+k*4,2:,5],mfc=clrs[col-1],
            mec=clrs[col-1],marker='_',lw=0) 
        mm=se=res[col+k*4,2:,5];se=res[col+k*4,2:,6]
        plt.plot(x-0.15,[mm-1.96*se,mm+1.96*se],color=clrs[col-1],alpha=0.7)     
        handles.append(out[0])
        plt.plot()
    ax=plt.gca()
    ax.set_xticks(xs+0.1);
    mps=['complete pooling','no pooling','hierarchical', 'hier. with predictors'][:3]
    if not short: 
        ax.set_xticklabels(['1s','2s','1sc','2sc']*4)
        plt.xlabel(([13,11][int(short)]*'   ').join(mps))
    else: ax.set_xticklabels(mps);
    plt.yticks(np.arange(0,7,0.5))
    plt.ylim(ylim)
    plt.xlim([-0.5,xs[-1]-0.7*3.5])
    plt.grid(True,axis='y')
    #if not k: plt.xlabel('LC model')
    plt.ylabel(['Tobii X3 120','SMI Redn'][dev]+'\nAccuracy in degrees')
    if legend:plt.legend(handles[::-1],lbls[::-1],loc=1)
    #plt.title(['Tobii X3 120','SMI Redn'][k])  

def figurePreproc():
    '''plot illustration of the preprocessing steps'''
    figure(size=3,aspect=0.4,dpi=DPI)
    d=np.load(DPATH+'d.npy',allow_pickle=True)
    raw=np.concatenate(d,0)
    subplot(1,2,1)
    plt.plot((raw[:,1]-raw[0,1])/1000000,raw[:,3])
    plt.plot((raw[:,1]-raw[0,1])/1000000,raw[:,4])
    plt.ylim([-30,50])
    e=np.load(DPATH+'e.npy',allow_pickle=True)
    e[:,0]-=e[0,0]
    plt.xlabel('Time in seconds')
    plt.ylabel('Location in degrees')
    #plt.legend(['horizontal','vertical'],loc=1)
    subplot(1,2,2)
    plt.plot(e[:,0],e[:,1])
    plt.plot(e[:,0],e[:,2])
    #plt.ylabel('Location in degrees')
    plt.xlabel('Time in seconds')
    plt.ylim([-30,50])
    f=np.load(DPATH+'f.npy')
    k=0;h=0
    val=[1,5,7,14,16,19,21]
    for ff in f:
        if e[ff[1],0]-e[ff[0],0]>0.1:
            plt.plot([e[ff[0],0],e[ff[1],0]],[50-1*k,50-1*k],['r','g'][int(h in set(val))]);
            k+=1
        h+=1
    plt.savefig('../publication/figs/preproc.png',bbox_inches='tight',dpi=DPI)

def computeVarAll(fn,doCompile=True,dva=0,transform=0,predictors=0,quick=False,eye=2):
    ''' compute accuracy estimates with the three-level model
        fn - suffix of the ds file with input data
        dev - device: 0=Tobii, 1=SMI
        eye - 0= left, 1=right, 2=binocular average
        docompile - if true compiles the stan model
        dva - which unit to use
        predictors - include accuracy predictors with the model
        transform - 0: accuracy predictors are linear w.r.t. standard deviation
            - 1: accuracy predictors are linear w.r.t. variance
            - 2: accuracy predictors are log-linear w.r.t. standard deviation
        quick - use 1000 MC iterations of each chain
    '''
    
    trns=['','sqrt','exp'][transform]
    pn=['','+nas*age[n]'][predictors]
    pm=['','+mms*(m-1)+mas[(m>3)+1]*age[n]'][predictors]
    po=['','+ods[(m>3)+1]*dist[n,m,p]+oms*(m-1)+oas[(m>3)+1]*age[n]'][predictors]
    lbs=['0','-10'][int(transform==2)]
    model=f'''
    data {{
        int<lower=0> N; //nr subjects
        vector[2] y[N,6,9];
        vector[2] c[N,6,9];
        real age[N];
        real dist[N,6,9];
    }}parameters{{
        vector<lower=-100,upper=100>[2] o[N,6];
        real<lower=-10,upper=10> r[2,N];
        vector<lower={lbs},upper=10>[2] sy[2];
        vector<lower={lbs},upper=10>[2] so[2];
        vector<lower=-100,upper=100>[2] mo[2,N];
        vector<lower={lbs},upper=10>[2] sm[2];
        vector<lower=-100,upper=100>[2] mm[2];
        real<lower=0,upper=10> sr[2];
        real<lower=-10,upper=10> mr[2];'''+['','''
        vector<lower=-10,upper=10>[2] nas;
        vector<lower=-10,upper=10>[2] ods[2];
        vector<lower=-10,upper=10>[2] mms;
        vector<lower=-10,upper=10>[2] mas[2];
        vector<lower=-10,upper=10>[2] oms;
        vector<lower=-10,upper=10>[2] oas[2];'''][predictors]+f'''
    }} model {{
        for (n in 1:N){{
            mo[1][n]~normal(mm[1],{trns}(sm[1]{pn}));
            mo[2][n]~normal(mm[2],{trns}(sm[2]{pn}));
            r[1][n]~normal(mr[1],sr[1]);
            r[2][n]~normal(mr[2],sr[2]);
        for (m in 1:6){{
            o[n,m]~normal(mo[(m>3)+1][n],{trns}(so[(m>3)+1]{pm}));
        for (p in 1:9){{
            if (! is_nan(y[n,m,p][1]))
                c[n,m,p]~normal(o[n,m]+r[(m>3)+1][n]*y[n,m,p],
                    {trns}(sy[(m>3)+1]{po}));}}}}}}}}'''
    print(model) 
    sm = pystan.StanModel(model_code=model)                
    #with open(DPATH+f'smHADER{transform}.pkl', 'wb') as f: pickle.dump(sm, f)
    #with open(DPATH+f'smHADER{transform}.pkl', 'rb') as f: sm=pickle.load(f)
    #with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    ds=np.load(DPATH+f'ds{fn}dva{dva}.npy')
    ds=np.hstack((ds[0,:],ds[1,:]))
    #print(ds.shape);bla

    sel=~np.all(np.isnan(ds[:,:,eye,:,0]),axis=(1,2))
    #sel[30:]=False
    age=np.load(DPATH+'age.npy')[sel]
    age= age/30-7
    dist=(ds[sel,:,eye,:,6]-57.5)/10
    y=ds[sel,:,eye,:,:2]
    c=ds[sel,:,eye,:,2:4]
    dat={'y':y,'N':y.shape[0],'c':c,'age':age,'dist':dist}
    mlt=[10,1][int(quick)]
    fit = sm.sampling(data=dat,iter=1000*mlt,chains=6,thin=10,warmup=500*mlt,n_jobs=6,seed=SEED,init=0)
    saveStanFit(fit,DPATH+f'smHADER{transform}{eye}{predictors}dva{dva}')
def _avgDist(h,v,N=10000000,lim=None):
    ''' return average expected distance E[sqrt(x^2+y^2)]
        with [x,y]~normal([0,0],S) where S is the covariance matrix 
            S=diag([h^2,v^2])
        h, v - standard deviations of the normal distribution that generates x and y
        N - number of samples, the default is sufficient for two-decimals precision
        lim - int parameter for adjusting the required memory capacity 0<lim<N
    '''
    na=np.newaxis
    if lim is None: lim=N
    if np.isscalar(h):h=np.array([h])
    if np.isscalar(v):v=np.array([v])
    M=N/lim
    res=0
    for m in range(int(M)):
        a=np.square(np.random.randn(lim)[:,na,na]*h[na,:])
        b=np.square(np.random.randn(lim)[:,na,na]*v[na,:])
        res+=np.sqrt(a+b).mean(0)/M
    return np.squeeze(res)
def plotVarAll(trns,prediction,f=None,w=None,saveFig=True):
    ''' plots accuracy at each level for an 7M infant seat 60 cm from eye tracker
        w - stanfit data which are plotted, if None prediction 
            and trns determine the input
        prediction - include accuracy predictors with the model
        trns - 0: accuracy predictors are linear w.r.t. standard deviation
            - 1: accuracy predictors are linear w.r.t. variance
            - 2: accuracy predictors are log-linear w.r.t. standard deviation
        f- function necessary to transform the Stan parameters to accuracy
        saveFig - if true the figure is saved to hard drive
    '''
    if f is None: f=[lambda x:x,np.sqrt,lambda x: np.exp(x)-1][trns]
    if w is None: w=loadStanFit(f'data/smHADER{trns}2{prediction}dva0')
    assert(np.max(w['rhat'][0,:-1])<1.1)# assume convergence when R^hat<1.1
    na=np.newaxis
    wds=['sy','so','sm']
    for wd in wds:
        #print(wd)
        tmp=_avgDist(f(w[wd][...,0]),f(w[wd][...,1]),lim=50000)
        w[wd]= np.concatenate((f(w[wd]),tmp[...,na]),axis=2)
    #plt.figure(figsize=(10,5))
    figure(size=3,aspect=0.5)
    for dev in range(2):
        #axx=plt.subplot(1,2,dev+1)
        axx=subplot(1,2,dev+1)
        sm=[0,0]
        for ax in range(3):
            #print(['hor','ver','both'][ax])
            for i in range(len(wds)): 
                if ax==2: 
                    if wds[i][1]!='m': sm[0]+=np.square(w[wds[i]][:,dev,ax])
                    sm[1]+=np.square(w[wds[i]][:,dev,ax])
                errorbar(w[wds[i]][:,dev,ax],x=[3*i+ax],clr='gcy'[ax])

            #plt.plot([3*ax+i+.5,3*ax+i+.5],[0,10],'k',lw=0.5)
        et=['Tobii X3 120','SMI REDn'][dev]
        print(et+' NP acc.: ',np.round(errorbar(np.sqrt(sm[0]),x=[999])[0],2))
        print(et+' CP acc.: ',np.round(errorbar(np.sqrt(sm[1]),x=[999])[0],2))
        print(et+' PE: ',np.round(errorbar(sm[0]/sm[1],x=[999])[0],2))
        plt.grid(True,axis='y')
        plt.xlim([-.5,3*ax+i+0.5])
        axx.set_xticks(range(3*ax+i+1))
        axx.set_xticklabels(['H','V','C']*3)
        plt.ylim([0,[2.5,2][prediction]])
        subplotAnnotate()
        if dev==0: plt.ylabel('Accuracy in degrees')
        plt.title(et+'\nlocation   session   population')
        if saveFig: plt.savefig(f'../publication/figs/var{trns}{prediction}.png',bbox_inches='tight',dpi=DPI)  
        
def plotSlopeAll(trns):
    ''' plots the preditor values and prints hypothetic manipulations
        trns - determines which Stan model/file is plotted
            - 0: accuracy predictors are linear w.r.t. standard deviation
            - 1: accuracy predictors are linear w.r.t. variance
            - 2: accuracy predictors are log-linear w.r.t. standard deviation
    '''
    w=loadStanFit(f'data/smHADER{trns}21dva0')
    wds=['oas','mas','nas','oms','mms','ods']
    figure(size=3,aspect=0.5)

    k=0
    devlbl=['Tobii','SMI'];levlbl={'o':'L','m':'S','n':'P'}
    vlbl={'a':'Age','m':'Session','d':'Distance'}
    for i in range(len(wds)): 
        for dev in range(1+int(w[wds[i]].ndim==3)):
            for ax in range(2):
                if w[wds[i]].ndim==3: tmp=w[wds[i]][:,dev,ax]
                else:tmp=w[wds[i]][:,ax]
                out=errorbar(tmp,x=[k],clr='gcy'[ax]);k+=1
                #if wds[i]=='ods' and ax==0 and dev==1: print(out)
            ll=['',':'+devlbl[dev]][int(w[wds[i]].ndim==3)]
            s=f'{vlbl[wds[i][1]]}\n{levlbl[wds[i][0]]}{ll}'
            plt.text(k-1.5,0.41,s,ha='center',size=8)
            if i==3 or i==5:plt.plot([2*(i+1)+1.5,2*(i+1)+1.5],[-1,1],'k',lw=0.5)
    plt.grid(True,axis='y')
    plt.xlim([-.5,k-.5])
    plt.gca().set_xticks([])
    #axx.set_xticklabels(['H','V']*len(wds))
    plt.ylim([-.4,.4])
    plt.ylabel('Accuracy in degrees')
    plt.savefig(f'../publication/figs/slope{trns}.png',bbox_inches='tight',dpi=DPI) 
    def perturb(ww,ax,wp,val,dev=0,w=None):
        ''' compute and print total accuracy before and after reg coef wp
            is added val times to variable wp
            dev - eye-tracking device
            ax - axis on which the addtion is performed 0 - hor, 1-ver, 2-both
            w - stan parameter estimatesd
        '''
        devs=['TOB','SMI']
        if w is None: w=loadStanFit(f'data/smHADER021dva0')
        aa=errorbar(np.sqrt(np.array(list(map(lambda x: np.square(_avgDist(
            np.median(w[x][:,dev,0]),np.median(w[x][:,dev,1]))),['sy','so']))).sum(0)))[0][0]
        if ax==2:
            if w[wp].ndim==2:w[ww][:,dev,:]+=val*w[wp]
            else:w[ww][:,dev,:]+=val*w[wp][:,dev,:]
        else: 
            if w[wp].ndim==2:w[ww][:,dev,ax]+=val*w[wp][:,ax,na]
            else: w[ww][:,dev,ax]+=val*w[wp][:,dev,ax]
        bb=errorbar(np.sqrt(np.array(list(map(lambda x: np.square(_avgDist(
            np.median(w[x][:,dev,0]),np.median(w[x][:,dev,1]))),['sy','so']))).sum(0)))[0][0]
        print(f'{devs[dev]}: {ww} {aa:.3f} -> {wp} {bb:.3f}, {bb-aa:.3f}')
        plt.close()
        return w
    perturb('so',2,'mas',3,dev=0)
    perturb('so',2,'mas',-3,dev=1)
    w=perturb('so',2,'mms',4,dev=0)
    w=perturb('sy',2,'oms',4,dev=0,w=w)
    
    w=perturb('so',2,'mms',4,dev=1,w=w)
    w=perturb('so',2,'oms',4,dev=1,w=w)
    plotVarAll(0,1,w=w,saveFig=False)
    perturb('sy',2,'ods',1,dev=1)
     
if __name__=='__main__':
    figureSample(f'dsFixTh1_0dva0incl',dev=0) 
    figureSample(f'dsFixTh2Vel20minDur0_1dva0incl')
    figureSample(f'dsFixTh1_0dva0incl',dev=1)
    validateLC('FixTh2Vel20minDur0_1',mcal=1,mval=0,dev=0,plot=1,pref='aP')
    plotVarAll(trns=0,prediction=0)
    plotVarAll(trns=1,prediction=1)

    figureAcc() 
    plotVarAll(trns=0,prediction=1)
    plotSlopeAll(0);stop

    # loading and preprocessing
    fns=checkFiles()             
    D=loadCalibrationData(fns)
    with open(DPATH+'D.out','wb') as f: pickle.dump(D,f)
    with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    dataPreprocessing(D,f'dsFixTh1_0dva0',thacc=1,dva=i)
    dataPreprocessing(D,'dsFixTh2Vel20minDur0_1dva0',thacc=2,
        thvel=20,dva=0,minDur=0.1)
    #compute results
    trainLC('FixTh1_0dva0',m=0,dev=0,docompile=False) 
    computeVarAll('FixTh1_0',transform=0,predictors=1)
    #compute results of supplementary analyses
    trainLC('FixTh1_0dva0',m=0,dev=1,docompile=False)
    trainLC('FixTh2Vel20minDur0_1dva0',m=0,dev=0,docompile=False)
    computeVarAll('FixTh1_0',transform=1,predictors=1)
    computeVarAll('FixTh1_0',transform=0,predictors=0)
    #plot figures
    figureAcc() 
    plotVarAll(trns=0,prediction=1)
    plotSlopeAll(0)
    #plot supplementory figures
    sampleDescr(4)
    figurePreproc()
    figureSample(f'dsFixTh1_0dva0incl',dev=0) 
    figureSample(f'dsFixTh2Vel20minDur0_1dva0incl')
    figureSample(f'dsFixTh1_0dva0incl',dev=1)
    validateLC('FixTh2Vel20minDur0_1',mcal=1,mval=0,dev=0,plot=1,pref='aP')
    plotVarAll(trns=0,prediction=0)
    plotVarAll(trns=1,prediction=1)
    
   

    

    
    




