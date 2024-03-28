import os,stan,pickle
import numpy as np
import pylab as plt
from matusplotlib import saveStanFit,loadStanFit
from scipy.stats import scoreatpercentile as sap
MD=70 #monitor distance used to compute deg visual angle in the output files
#position of calibration points
CTRUE=np.array([[0,0],[-11,11],[11,11],[11,-11],[-11,-11],[11,0],[-11,0],[0,11],[0,-11]]) # true calibration locations in degrees
#CTRUE=CTRUEDEG/180*np.pi*MD # true calibartion locations in cm
SEED=8 # the seed of random number generator was fixed to make the analyses replicable
TC,TS,F,LX,LY,RX,RY,BX,BY,LD=range(10); RD=12;BD=15
DPI=500 #figure resolution
DPATH='data'+os.path.sep  
##########################################################################
# DATA LOADING ROUTINES
##########################################################################

def printRhat(w):
    from arviz import summary
    print('checking convergence')
    azsm=summary(w)
    nms=azsm.axes[0].to_numpy()
    rhat = azsm.to_numpy()[:,-1]
    nms=nms[np.argsort(rhat)]
    rhat=np.sort(rhat)
    stuff=np.array([nms,rhat])[:,::-1]
    print(stuff[:,:10].T)
    i=(rhat>1.1).nonzero()[0]
    nms=nms.tolist()
    nms.append('__lp')
    nms=np.array(nms)[np.newaxis,:]
    rhat=rhat.tolist()
    rhat.append(-1)
    rhat=np.array(rhat)[np.newaxis,:]
    return i.size>0,nms,rhat
    
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
    
def getCM(c,ctrue,normEnt=0,onepar=False,residVec=False):
    ''' computes linear calibration 
        calibData - calibration data returned by readCalibGaze 
        omit - list contains label of
            calibration point to be ommited
            there were 9 calibration points, labeled from 
            left to right, top to bottom
        onepar - does not work, needs fixing
        returns 
            - 1D array with calibration coefficients in format 
            [offset X, slope X, offset Y, slope Y]
            - residual
            - array with omitted calibration points
            - (median) gaze location on each of calibration points
    '''
    coef=[np.nan,np.nan,np.nan,np.nan]
    resid=np.nan
    #assert(c.shape[0]==9)
    assert(c.shape[1]==2)
    assert(np.all(np.isnan(c[:,0])==np.isnan(c[:,1])))
    sel=~np.isnan(c[:,0])
    if sel.sum()<[3,2][int(onepar)]: return coef,resid,c
    temp=np.zeros(sel.sum())
    for k in range(2):
        if onepar: x=np.ones((sel.sum(),1))
        else:x=np.column_stack([np.ones(sel.sum()),c[sel,k]])
        res =list(np.linalg.lstsq(x,ctrue[sel,k],rcond=None))
        if len(res[1])==0: return [np.nan,np.nan,np.nan,np.nan],np.nan,c
        if onepar: coef[k*2:(k+1)*2]=np.array([res[0][0],0])
        else: coef[k*2:(k+1)*2]=res[0]
        temp+=np.square(ctrue[sel,k]-x.dot(res[0]))#res[1][0]
    #resid=np.max(np.sqrt(temp))
    w=np.power(temp,normEnt)
    w=w/w.sum()
    if residVec: 
        resid=np.zeros(c.shape[0])*np.nan
        resid[~np.isnan(c[:,0])]=np.sqrt(temp)
    else: 
        resid=np.sqrt((w*temp).sum())#*(sel.sum()**normEnt)/(sel.sum()**normEnt+(~sel).sum()**normEnt)*2
        assert(np.any(np.isnan(resid))==np.any(np.isnan(coef))) 
    assert(np.all(np.isnan(coef))==np.any(np.isnan(coef)))
    
    return coef,resid,c
def cmPredict(y,coef):
    '''coef - 1D array with calibration coefficients in format 
            [offset X, slope X, offset Y, slope Y]
       returns ndarray with the predicted coordinates'''
    assert(y.shape[1]==2)
    return np.array([np.array([np.ones(y.shape[0]),y[:,0]]).T.dot(coef[:2]), np.array([np.ones(y.shape[0]),y[:,1]]).T.dot(coef[2:])]).T


def selCPiter(c,ctrue,minNrCL,th,normEnt,onepar):
    cm=list(getCM(c,ctrue,normEnt=normEnt,onepar=onepar))
    co=cm.copy()
    while np.isnan(co[2][:,0]).sum()<(c.shape[0]-minNrCL) and (co[1]>th):
        ci=co.copy()
        for k in range(c.shape[0]):
            if np.isnan(co[2][k,0]): continue
            cg=co[2].copy();cg[k,:]=np.nan
            res=list(getCM(cg,ctrue,normEnt=normEnt,onepar=onepar))
            if res[1]<ci[1]:ci=res
        if th==0: print(ci[1],np.int32(~np.isnan(cg[:,0])))
        if ci[1]<co[1]: co=ci
        else: break
    if co[1]<cm[1]:cm=co
    if cm[1]>th or cm[0][1]<0 or cm[0][3]<0 or not np.isnan(c[:,0]).sum()<=(c.shape[0]-minNrCL): 
        cm=[np.nan*np.ones(4),np.nan,np.zeros((c.shape[0],2))*np.nan]
    return cm


def selCPalgo(c,ctrue,algo,minNrCL,repeat=None,threshold=None,
    replacement=np.zeros((0,0,0)),normEnt=0,onepar=False):
    if minNrCL<=0:minNrCL=c.shape[0]+minNrCL
    def enoughCLs(res): return np.isnan(res[2][:,0]).sum()<=(c.shape[0]-minNrCL)
    assert(repeat in ('block','trial',None))
    if algo in ('local','avg'): assert(not threshold is None)
    res=[None,np.inf,c]
    for k in range(-1,replacement.shape[2]):
        if k>=0:
            sel=np.isnan(res[2][:,0])
            res[2][sel,:]=replacement[sel,:,k]
        if algo=='local' or algo=='valid': 
            res=list(getCM(np.copy(res[2]),np.copy(ctrue),normEnt=normEnt, onepar=onepar,residVec=True))
            if algo=='local':res[2][res[1]>threshold,:]=np.nan
        elif algo=='iter':
            res=selCPiter(np.copy(res[2]),np.copy(ctrue),minNrCL=minNrCL,normEnt=normEnt,onepar=onepar,th=threshold)
        elif algo=='avg':
            res=list(getCM(np.copy(res[2]),np.copy(ctrue),normEnt=normEnt, onepar=onepar))
            if repeat in ('block',None) and (res[1]>threshold):res[2][:,:]=np.nan
        
        if not enoughCLs(res) and repeat in ('block',None): res[2][:,:]=np.nan
        if enoughCLs(res) and (np.all(res[1]<threshold) or algo=='valid'): break
        
        k+=1
    if res[0] is None or not enoughCLs(res): res=[np.nan*np.ones(4),np.nan,np.zeros((c.shape[0],2))*np.nan]
    res.append(k)
    return res
    #if not returnPars: return np.int32(~np.isnan(res[2][:,0]))


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
    #elif dva==8:  x=55*(cm-hvd)/frd
    return x    
    
def extractFixations(inG,eye,thvel=10,hz=60,minDur=0.06,dva=0): 
    AX=np.newaxis
    res=np.ones((9,7))*np.nan
    if len(inG)==0: return res
    elif len(inG)==1:
        G=np.array(inG)
        inG=[inG]
    else:G=np.concatenate(inG,0)
    if G.shape[0]<3: return res 
    C=[]
    for i in range(len(inG)):
        C.append(np.ones((inG[i].shape[0],2))*np.array(CTRUE[i]))
    C=np.concatenate(C,0)
    
    
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
    
def extractGaze(inG,eye,thvel=10,hz=60,minDur=0.5,dva=0):  
    AX=np.newaxis
    n=int(minDur*hz)
    res=np.ones((9,7))*np.nan
    C=[]
    for p in range(len(inG)):
        C.append(np.ones((inG[p].shape[0],2))*np.array(CTRUE[p]))
        inG[p][:,LD+3*eye:LD+3+3*eye]/=10 # mm to cm
        inG[p][:,LD+1+3*eye]-=16.45;inG[p][:,LD+2+3*eye]+=2.5 # origin at screen center
        if inG[p].shape[0]==0:continue
        for i in range(2):
            ii=[LX+2*eye,LY+2*eye][i]
            sel=~np.isnan(inG[p][:,ii])
            if sel.sum()<n:continue
            tmp=np.nonzero(sel)[0]
            sel[tmp[n-1]:]=False
            x=chunits(inG[p][sel,ii],dva,hvd=inG[p][sel,LD+i+3*eye],frd=inG[p][sel,LD+2+3*eye])
            C[p][sel,i]=chunits(C[p][sel,i],dva,hvd=inG[p][sel,LD+i+3*eye],frd=inG[p][sel,LD+2+3*eye])
            res[p,:2]=np.nanmedian(x,0)
            res[p,2:4]=np.nanmean(C[p][sel,i],0)
            res[p,4:]=np.nanmean(inG[p][sel,LD+3*eye:RD+3*eye],0)
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
    MINVALIDCL=[1,1,1]
    R=np.zeros((2,3,3,9,7))*np.nan
    included=np.zeros((2,3,3,3,7),dtype=np.int32)
    coh=[7,7,7,7,0,7,7,1,7,7,2][G[0][1]]
    c=np.zeros((9,3))*np.nan
    
    for s in range(2):
        for m in range(3):
            d=[G[1+s][:9],G[1+s][9:14],G[1+s][14:19]][m]              
            for i in range(3):
                
                included[s,coh,m,i,6]=1
                check=np.zeros(9,dtype=bool)
                for p in range(min(check.size,len(d))):
                    check[p]=np.any(~np.isnan(d[p][:,[LX+2*i,LY+2*i]]))
                if check.sum()>=MINVALIDCL[m]: 
                    included[s,coh,m,i,1]=1
                    included[s,coh,m,i,0]=check.sum()
                if dva==5: dva2=dva+m
                else:dva2=dva
                meth=[extractFixations,extractGaze][thvel is None]
                c=meth(d,i,thvel=thvel,minDur=minDur,dva=dva2)
                R[s,m,i,:,:]=c
                if np.isnan(c[:,0]).sum()<=(9-MINVALIDCL[m]):
                    included[s,coh,m,i,[3,5]]=1
                    included[s,coh,m,i,[2,4]]=(~np.isnan(c[:,0])).sum()
                else: R[s,m,i,:,:2]=np.nan 
    return R,included 

   
def dataPreprocessing(D,fn,thacc=0.5,thvel=10,minDur=0.06,dva=0,verbose=False,ncpu=8):  
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
    #assert(not np.any(np.isnan(ds[:,:,:,:,:,0]).sum(4)==7))
    #assert(not np.any(np.isnan(ds[:,:,:,:,:,0]).sum(4)==8))
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


    

def computeVarAll(fn,addSMI=False,transform=0,predictors=0,ssmm=1,eye=2):
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
    mdata=f'''
    data {{
        int<lower=0> N; //nr subjects
        int<lower=0> M; //nr blocks
        real y[N,M,9,2];
        int P[7];
        int E;
        int mask[N,M];
        real c[N,M,9,2];
        vector[N] age;
        real dist[N,M,9];
    '''

    addSMI=int(addSMI)
    trns=['','sqrt','exp','abs'][transform]
    pn=['','+nas[k,e]*age'][predictors]  
    po=['','+oas[k,(m>3)+1]*age[n]+ods[k,1,(m>3)+1]*dist[n,m,p]+ods[k,2,(m>3)+1]*abs(dist[n,m,p])+oms[k,(m>3)+1]*(P[m]+p-10)+mms[k]*(m-2)+oqs[k,(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3))'][predictors]# 
    s2min=[5,1.6][int(transform==2)]
    lbs=['0','-10'][int(transform==2)]
    model=mdata+f'''
    }}parameters{{
        vector<lower=-100,upper=100>[N] o[2,E];
        vector<lower=-10,upper=10>[N] r[2,E];
        real<lower={lbs},upper=20> sy[2,E];
        real<lower={lbs},upper=10> sm[2,E];
        real<lower=-100,upper=100> mm[2,E];
        real<lower=0,upper=10> sr[2,E];
        real<lower=-10,upper=10> mr[2,E];
        real<lower=-10,upper=10> nam[2,E];
        real<lower=-10,upper=10> odm[2,2,E];
        real<lower=-10,upper=10> mmm[2];
        real<lower=-10,upper=10> omm[2,E];
        real<lower=-10,upper=10> oqm[2,E];
        real<lower=-10,upper=10> oam[2,E];
        '''+['','''
        real<lower=-10,upper=10> nas[2,E];
        real<lower=-10,upper=10> ods[2,2,E];
        real<lower=-10,upper=10> mms[2];
        real<lower=-10,upper=10> oms[2,E];
        real<lower=-10,upper=10> oqs[2,E];
        real<lower=-10,upper=10> oas[2,E];'''][predictors]+f'''
        real<lower={s2min}> s2[2,E];
        vector<lower=-100,upper=100>[N] t[M];
        real<lower=-100,upper=100> mt; real<lower=0,upper=100> st;
        real<lower=-100,upper=100> nat;
        real<lower=-100,upper=100> mmt;
        real<lower=-100,upper=100> omt;
        real<lower=-100,upper=100> qmt[E];
        real<lower=-100,upper=100> onan[E];
        real<lower=-100,upper=100> anan;
        real<lower=-100,upper=100> pnan;
        real<lower=-100,upper=100> mnan;
        real<lower=-100,upper=100> qnan[E];
        
    }} model {{
        real lps[2];
        real tmp;
        real tmpnan;
        for (k in 1:2){{
        for (e in 1:E){{ 
            r[k,e]~normal(mr[k,e],sr[k,e]);
            o[k,e]~normal(mm[k,e]+nam[k,e]*age,({trns}(0.1*(sm[k,e]{pn}))));}}}}
        t~normal(mt+nat*age,st);
        for (m in 1:M){{
        for (n in 1:N){{
        if (mask[n,m]==1){{
        for (p in 1:(P[m+1]-P[m])){{
            tmp=t[n]+mmt*(m-2)+omt*(p-10+P[m])+qmt[(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3));
            tmpnan=onan[(m>3)+1]+pnan*(p-10+P[m])+qnan[(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3))+mnan*(m-2)+anan*age[n];
            if ( (! is_nan(y[n,m,p,1])) ){{ //&& distance(to_vector(c[n,m,p]),to_vector(y[n,m,p]))<7
                lps[1]=log_inv_logit(tmp); 
                lps[2]=log1m_inv_logit(tmp);
                for (k in 1:2){{
                    lps[1]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n]+r[(m>3)+1][n].*c[n,m,p,k]+oam[k,(m>3)+1]*age[n]+odm[k,1,(m>3)+1]*dist[n,m,p]+odm[k,2,(m>3)+1]*abs(dist[n,m,p])+omm[k,(m>3)+1]*(P[m]+p-10)+mmm[k]*(m-2)+oqm[k,(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3)),{trns}(0.1*(sy[k,(m>3)+1]{po})));
                    lps[2]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n],{trns}(s2[k,(m>3)+1])); 
                 }}
                target+= log_sum_exp(lps)+log1m_inv_logit(tmpnan);
                }}
            else target+=log1m_inv_logit(tmp)+log_inv_logit(tmpnan);
        }}}}}}}}}}generated quantities {{
        real lpsTemp[2];
        real tmpTemp;
        real onTargetGen[N,M,9];
        for (m in 1:M){{
        for (n in 1:N){{
        if (mask[n,m]==1){{
        for (p in 1:(P[m+1]-P[m])){{
            tmpTemp=t[m][n]+omt*(p-10+P[m])+qmt[(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3));
            if ( (! is_nan(y[n,m,p,1])) ){{ 
                lpsTemp[1]=log_inv_logit(tmpTemp); 
                lpsTemp[2]=log1m_inv_logit(tmpTemp);
                for (k in 1:2){{
                    lpsTemp[1]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n]+r[(m>3)+1][n].*c[n,m,p,k],{trns}(0.1*(sy[k,(m>3)+1]{po})));
                    lpsTemp[2]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n],{trns}(s2[k,(m>3)+1])); 
                 }}
                onTargetGen[n,m,p]=exp(lpsTemp[1]-log_sum_exp(lpsTemp));
                }}
            else onTargetGen[n,m,p]=0; 
        }}}}}}}}}}'''        
    pn=['','+nas*age'][predictors]  #TODO remove age preds
    po=['','+oas*age[n]+oms*(P[m]+p-10)+mms*(m-2)'][predictors]# 
    modeldist=mdata+f'''
        real nomdist [M];
    }}parameters{{
        vector[N] o;
        real<lower=-10,upper=10> sy;
        real<lower=-10,upper=10> sm;
        real<lower=-100,upper=100> mm;
        real<lower=-100,upper=100> mmm;
        real<lower=-100,upper=100> qmm;
        real<lower=-100,upper=100> omm;
        real<lower=-100,upper=100> emm;
        real nam;'''+['','''
        real nas;
        real mms;
        real oms;
        real oas;'''][predictors]+f'''
    }} model {{
        o~normal(mm+nam*age,{trns}(sm{pn}));
        for (m in 1:M){{
        for (n in 1:N){{
        if (mask[n,m]==1){{
        for (p in 1:(P[m+1]-P[m])){{
            if ( (! is_nan(dist[n,m,p]))){{ 
                dist[n,m,p]~normal(o[n]+nomdist[m]+emm*(m>3)+omm*(P[m]+p-10)+mmm*(m-2)+qmm*(to_int(p==2) + to_int(p==3)),{trns}(sy{po}));
        }}}}}}}}}}}}'''  


         
    #with open(DPATH+f'smHADER{transform}.pkl', 'wb') as f: pickle.dump(sm, f)
    #with open(DPATH+f'smHADER{transform}.pkl', 'rb') as f: sm=pickle.load(f)
    #with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    ds=np.load(DPATH+f'ds{fn}.npy')
    ds=np.hstack((ds[0,:],ds[1,:]))
    if addSMI==2: ds=ds[:,3:,:,:,:]
    else:ds=ds[:,:(addSMI+1)*3,:,:,:]
    sel=~np.all(np.isnan(ds[:,:,eye,:,0]),axis=(1,2))
    #sel[50:]=False
    age=np.load(DPATH+'age.npy')[sel]
    age= age/30-7
    dist=(ds[sel,:,eye,:,6]-55)/10#-57.5
    y=ds[sel,:,eye,:,:2]
    c=ds[sel,:,eye,:,2:4]
    mask=np.int32(~np.all(np.isnan(y[:,:,:,0]),axis=2))
    print(y.shape)
    dat={'y':y,'N':y.shape[0],'c':c,'age':age,'dist':dist,'mask':mask,'M':y.shape[1],'P':[0,9,14,19,28,33,38],'E':y.shape[1]//3}#'J':(~np.isnan(y[:,:3,:,:])).sum()//2
    if False:
        print(modeldist)
        dat['nomdist']=(np.array([55,45,65,55,45,65])-55)/10
        sm = stan.build(program_code=modeldist,data=dat,random_seed=SEED) 
        fit = sm.sample(num_chains=6,num_thin=10,num_samples=5000,num_warmup=5000)
        saveStanFit(fit,dat,DPATH+f'sd2L06{fn}{transform}{eye}{predictors}{addSMI}',model=modeldist)
        stop
    on=np.ones((2,dat['E']))
    on1=np.ones(2)
    print(model)
    if transform==2: 
        tmp={'mas':on*0,'oas':on*0,'nas':on*0,'mms':on1*0,'oms':on*0,
            'ods':np.ones((2,2,dat['E']))*0,'sm':on*0,'so':on*0,'sy':on*0}
    else:
        tmp={'mas':on*.01,'oas':on*.01,'nas':on*.01,'mms':on1*.01,'oms':on*.01,
            'ods':np.ones((2,2,dat['E']))*.01,'sm':on*.5,'so':on*.5,'sy':on*.5}
    tmp['mt']=0;tmp['t']=np.zeros((y.shape[0],y.shape[1])).T
    sm = stan.build(program_code=model,data=dat,random_seed=SEED) 
    fit = sm.sample(num_chains=6,num_thin=int(max(ssmm*1,1)),
        num_samples=int(ssmm*500),
        num_warmup=int(ssmm*500)
        init=[tmp]*6)
    saveStanFit(fit,dat,DPATH+f'sm2L10{fn}{transform}{eye}{predictors}{addSMI}',model=model)
    
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
    if w is None: 
        with open(f'data/smHADER{trns}2{prediction}dva0.wfit','rb') as f: w=pickle.load(f)
    #assert(np.max(w['rhat'][0,:-1])<1.1)# assume convergence when R^hat<1.1
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
    
 


def generateFakeDataHT(age,Ms,P,Cs=None,Es=None,Nd=None,pnan=None,pOT=None,
    N=-1,K=1000,fn='',savefit=False):
    ''' age - in months
        Ms - array with calibration order (starting from 0) for each calibration
        P - array with 2-element list for each calibration, first element gives the 
            location-order of the first calibration location, second element gives the 
            location-order of the last calibration location
        Es - eyetracker id for each calibration
        Nd - nominal distance in cm for each calibration
        Xqmt - 2d ndarray with element for each cal loc and calibration, 
            equals 1 if ET malfunction is simulated
        N - number of infants, if -1 takes nr. of infants from the empirical investigation
    '''
    if Es is None: Es=np.ones(len(np.unique(Ms)),dtype=np.int32)
    if Nd is None: Nd=55*np.ones(len(np.unique(Ms)))
    if Cs is None: return #TODO
    if pOT is None: pOT=-np.ones(len(Ms))
    if pnan is None: pnan=-np.ones(len(Ms))
    mdat='''
        '''
    model='''data {
        int N,M,E,L; array[L] int P;array[L] int Ms;array[M] int MMs;array[M] int Es;
        array[L] int<lower=1,upper=9> Cs;array[L] real pnan;array[L] real pOnTarget;
        array[9,2] real c;array[N] real age;array[M] real nomdist;
        array[2,E] real sy;array[2,E] real sm;array[2,E] real mm;array[E] real sr;
        array[E] real mr;array[2,E] real nas;array[2,2,E] real ods;array[2] real mms;
        array[2,E] real oms;array[2,E] real oas; array[2,E] real s2;
        real mt,st,nat,mmt,omt;array[E] real qmt;
        real mmD,mmmD,qmmD,ommD,emmD,syD,smD,namD,nasD,mmsD,omsD,oasD;
    } transformed data{
        array[L] int Xqmt;
        for (l in 1:L) Xqmt[l]=(nomdist[Ms[l]]==-1)*((Cs[l]==2) + (Cs[l]==3));
    } generated quantities{
        array[N,L] int onTarget;array[N,L,2] real y;
        array[N,L] real<lower=-2.5,upper=4> dist;//hard limits at 30 and 95 cm
        array[2,E] vector[N] o;array[N] real oD;
        array[M] vector[N] t;array[E] vector[N] r;vector[2] temp;
        int nodat;
        for (n in 1:N){
            oD[n]=normal_rng(mmD+namD*age[n],exp(smD+nasD*age[n]));
        for (e in 1:E){
            r[e,n]=normal_rng(mr[e],sr[e]);
        for (k in 1:2) o[k,e,n]=normal_rng(mm[k,e],(exp(0.1*(sm[k,e]+nas[k,e]*age[n]))));}}
        
        
        
        for (n in 1:N){
        for (m in 1:M){
            t[m][n]=normal_rng(mt+mmt*(MMs[m]-2)+nat*age[n],st);}
        for (l in 1:L){
            nodat=bernoulli_rng(pnan[l]);
            if (nodat==1){
                dist[n,l]=4;//0.0/0;
                onTarget[n,l]=0;
                y[n,l,1]=0.0/0;
                y[n,l,2]=0.0/0;
            }else{
            dist[n,l]=normal_rng(oD[n]+nomdist[Ms[l]]+emmD*Es[Ms[l]]+ommD*(P[l]-9)+mmmD*(Ms[l]-2)+qmmD*(to_int(Cs[l]==2) + to_int(Cs[l]==3)),exp(syD+oasD*age[n]+omsD*(P[l]-9)+mmsD*(Ms[l]-2)));
            temp[2]=1-pnan[l];
            if (pOnTarget[l]==-1)   temp[1]=inv_logit(t[Ms[l]][n]+omt*(P[l]-9)+qmt[Es[Ms[l]]]*Xqmt[l]);
            else temp[1]=temp[2]*pOnTarget[l];
            if (temp[1]>=temp[2]) onTarget[n,l]=1;
            else onTarget[n,l]=bernoulli_rng(temp[1]/temp[2]);

            for (k in 1:2){
            if (onTarget[n,l]==1) y[n,l,k]=normal_rng(o[k,Es[Ms[l]],n]+r[Es[Ms[l]],n]*c[Cs[l],k],exp(0.1*(sy[k,Es[Ms[l]]]+oas[k,Es[Ms[l]]]*age[n]+ods[k,1,Es[Ms[l]]]*dist[n,l]+ods[k,2,Es[Ms[l]]]*abs(dist[n,l])+oms[k,Es[Ms[l]]]*(P[l]-9)+mms[k]*(Ms[l]+2))));
            else y[n,l,k]=normal_rng(o[k,Es[Ms[l]],n],exp(s2[k,Es[Ms[l]]]));
            }}}}}'''     
    dat={'M':len(np.unique(Ms)),'MMs':np.unique(Ms)+1,'Ms':np.array(Ms)+1,'L':len(P),
        'P':P,'E':2,'nomdist':(np.array(Nd)-55)/10,'Es':Es,'Cs':Cs}
    w=loadStanFit(f'data/sd2L06ThaccInfdva02211',excludeChains=[])
    for k in w.keys():
        if not k[-1]=='+' and not k in ('o'):
            #print(k,w[k].shape)
            dat[k+'D']=np.squeeze(np.median(w[k],0))
            assert(np.all(~np.isnan(dat[k+'D'])))
    w=loadStanFit(f'data/sm2L09ThaccInfdva02211',excludeChains=[])
    if N==-1:dat['N']=w['N+'];
    else:dat['N']=N
    dat['yold']=w['y+'];dat['distold']=w['dist+']
    dat['age']=(age-7)*np.ones(dat['N'])
    for k in w.keys():
        if not k[-1]=='+' and not k.endswith('Gen') and not k.endswith('Temp') and not k in ('o','t','r'):
            #print(k,w[k].shape) 
            dat[k]=np.squeeze(np.median(w[k],0))
            assert(np.all(~np.isnan(dat[k])))
    
    #dat['c']=np.nanmedian(w['c+'][:,0,:5,:],0)#todo 
    dat['c']=chunits(CTRUE,dva=0) 
    dat['pnan']=pnan;dat['pOnTarget']=pOT
    for l in range(len(Ms)):
        if dat['pnan'][l]==-1:dat['pnan'][l]=np.exp(dat['lpnan'])
    sm = stan.build(program_code=model,data=dat,random_seed=SEED) 
    fit = sm.fixed_param(num_chains=5,num_samples=K//5)
    if savefit: saveStanFit(fit,dat,'data/gen'+fn,model=model)
    else: 
        np.save('data/geny'+fn,fit['y'])
        np.save('data/geno'+fn,fit['onTarget'])
        np.save('data/genc'+fn,dat['c'])

def fakedataCalibrationWorker(L,e,nCP=100,selPred=10):
    import itertools
    y=np.rollaxis(np.load(f'data/genyLfkPref{L}{e}.npy'),-1,0)
    o=np.rollaxis(np.load(f'data/genoLfkPref{L}{e}.npy'),-1,0)
    c=np.load(f'data/c.npy')
    c=np.vstack([c,c])
    als=['avg','local','valid']
    ks=[3,-1]#[3,-1,0]
    I=np.reshape(np.arange((len(als))*3*(len(ks)+2)),[len(als),3,len(ks)+2])
    gzh=np.zeros((y.shape[0],y.shape[1],I.size,5,2))*np.nan
    mask=np.zeros(list(y.shape[:3])+[I.size],dtype=np.bool_)
    
    for i in range(y.shape[0]):
        todo=[]
        for n in range(y.shape[1]):
            #rpl=y[i,n,L+11:2*L+11,:,np.newaxis]
            rpl=np.rollaxis(np.reshape(np.copy(y[i,n,np.newaxis,L+11:3*L+11,:]),(2,-1,2)),0,3)

            for a,r,k in itertools.product(range(len(als)),range(I.shape[1]),range(len(ks))):
                res=selCPalgo(y[i,n,:L,:],c[:L,:],threshold=[3.5,3.5,np.inf,1.75][a],minNrCL=ks[k],#[4,1.5,np.inf,1.75]
                    repeat=[None,'block','trial'][r],algo=als[a],
                    replacement=[np.zeros((0,0,0)),rpl][int(r>0)])#1.75 
                if k==0: mask[i,n,:L,I[a,r,0]]=~np.isnan(res[2][:,0])
                gzh[i,n,I[a,r,k],:,:]=cmPredict(y[i,n,L:L+5,:],res[0])
            if False:            
                #NP+U
                todo.append(n) 
                if not np.isnan(resNP[0][0]): 
                    for do in todo:
                        gzh[i,do,2,:,:]=cmPredict(y[i,do,L:L+5,:],resNP[0])
                    todo=[]
                #NP1p
                resNP1p=selCPalgoTrialwise(y[i,n,:L,:],c[:L,:],MINVALIDCL=2, THACC=3.1,returnPars=True,onepar=True)
                gzh[i,n,3,:,:]=cmPredict(y[i,n,L:L+5,:],resNP1p[0])
    for i in range(y.shape[0]):
        #CP 
        for a,r in itertools.product(range(len(als)),range(I.shape[1])):
            aa=np.vstack([y[i-1,:nCP,:L,0][mask[i-1,:nCP,:L,I[a,r,0]]],y[i-1,:nCP,:L,1][mask[i-1,:nCP,:L,I[a,r,0]]]]).T
            ctrue=np.vstack([np.repeat(c[np.newaxis,:L,0],nCP,axis=0)[mask[i-1,:nCP,:L,I[a,r,0]]],
                            np.repeat(c[np.newaxis,:L,1],nCP,axis=0)[mask[i-1,:nCP,:L,I[a,r,0]]]]).T
            resCP=getCM(aa,ctrue)
            for n in range(y.shape[1]): 
                gzh[i,n,I[a,r,-1],:,:]=cmPredict(y[i,n,L+6:L+11,:],resCP[0])
                #NP+CP        
                tmp=cmPredict(y[i,n,L:L+5,:],resCP[0])
                for m in range(L,L+5):     
                    if np.isnan(gzh[i,n,I[a,r,0],m-L,0]): gzh[i,n,I[a,r,-2],m-L,:]=tmp[m-L,:] 
                    else:gzh[i,n,I[a,r,-2],m-L,:]=gzh[i,n,I[a,r,0],m-L,:]
                    for k in range(I.shape[2]):
                        if np.linalg.norm(gzh[i,n,I[a,r,k],m-L,:])>selPred: gzh[i,n,I[a,r,k],m-L,:]=np.nan     
    return gzh

def fakedataCalibration(e=1,ncpu=4,fn=''):
    from multiprocessing import Pool
    pool=Pool(ncpu)
    res=[]
    Ls=[3,5,9]
    for h in range(len(Ls)):
        temp=pool.apply_async(fakedataCalibrationWorker,[Ls[h],e])
        res.append(temp)
    pool.close()  
    from time import time,sleep
    tot=len(Ls)
    t0=time();prevdone=-1
    while True:
        done=0
        for r in res:
            done+=int(r.ready())
        sleep(1)
        if done==tot: break
    gz=[]
    for n in range(tot):
        gz.append(res[n].get())
    es=['tob','smi'][e-1] 
    np.save(f'data/gz{fn}{es}',gz)

def fakedataTTestPowerWorker(fn,ess,alpha,beta):
    def evalPower(dat):
        p=np.zeros(dat.shape[0])*np.nan
        for i in range(dat.shape[0]):
            sel=~np.isnan(dat[i,:])
            if sel.sum()>2:
                p[i]=ttest_1samp(dat[i,sel]+es/2,0,alternative='greater')[1]
                if np.isnan(p[i]):
                    print(dat[i,sel]);stop
        return (p<alpha).mean()>beta
    from scipy.stats import ttest_1samp
    gz=np.load(fn)
    n=-np.ones((ess.size,gz.shape[3],gz.shape[0],gz.shape[4],gz.shape[5]),dtype=np.int32)
    for ax in range(n.shape[4]):
        for m in range(n.shape[3]):
            for h in range(n.shape[2]):
                for k in range(n.shape[1]):
                    for esi,es in enumerate(ess):
                        un=200;ln=5
                        if not evalPower(gz[h,:,:un,k,m,ax]): n[esi,k,h,m,ax]=201
                        if evalPower(gz[h,:,:ln,k,m,ax]): n[esi,k,h,m,ax]=0
                        while n[esi,k,h,m,ax]==-1:
                            cn=(un+ln)//2
                            if evalPower(gz[h,:,:cn,k,m,ax]):un=cn
                            else:ln=cn
                            if un == ln+1: n[esi,k,h,m,ax]=un
    return n
def fakedataTTestPower(suf,ncpu=4,alpha=0.05,beta=.8):
    if suf=='NMG': ess=np.linspace(.1,2.1,21)
    else: ess=np.linspace(.5,1.5,21)
    from multiprocessing import Pool
    pool=Pool(ncpu)
    res=[]
    for e in range(2):
        es=['tob','smi'][e] 
        for a in range(3):
            temp=pool.apply_async(fakedataTTestPowerWorker,[f'data/gz{[4,7,10][a]}m{suf}{es}.npy',ess,alpha,beta])
            res.append(temp)
    pool.close()  
    from time import time,sleep
    tot=len(res)
    t0=time();prevdone=-1
    while True:
        done=0
        for r in res:
            done+=int(r.ready())
        sleep(1)
        if done==tot: break
    gz=np.load(f'data/gz7m{suf}tob.npy')
    n=-np.ones((ess.size,gz.shape[3],gz.shape[0],gz.shape[4],3,gz.shape[5],2),dtype=np.int32)
    k=0
    for e in range(2):
        for a in range(3):
            n[:,:,:,:,a,:,e]=res[k].get()
            k+=1
    np.save(f'data/n{suf}',n)

if __name__=='__main__':
    
    tp='MGsp10'#'NMG'
    for e in [2,1]:
        for a in [4,7,10]:
            for b in [3,5,9]:
                generateFakeDataHT(age=a,Ms=b*[0]+[1,2,3,4,5,6]+[0,1,2,3,4]+b*[1]+b*[2],
                    P=list(range(b+6))+list(range(5))+b*[b]+b*[b+1], 
                    Cs=(np.mod(np.arange(b),9)+1).tolist()+11*[1]+2*(np.mod(np.arange(b),9)+1).tolist(),
                    N=200,fn=f'LfkPref{b}{e}', pOT=[-1]*b+11*[[-1,1][tp=='NMG']]+[-1]*b*2,Es=7*[e],K=10)#,Nd=6*[45])   
            fakedataCalibration(e,fn=f'{a}m{tp}')   
    fakedataTTestPower(tp);stop
    generateFakeDataHT(age=7,Ms=9*[0]+5*[1]+5*[2]+9*[3]+5*[4]+5*[5],P=np.arange(38), Cs=np.array(list(range(9))+2*list(range(5))+list(range(9))+2*list(range(5)))+1, Es=[1,1,1,2,2,2], Nd=[55,45,65,55,45,65],N=-1,fn='empExp',savefit=True);stop
    

    # loading and preprocessing
    fns=checkFiles()             
    D=loadCalibrationData(fns)
    with open(DPATH+'D.out','wb') as f: pickle.dump(D,f)
    with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    dataPreprocessing(D,f'dsFixTh1_0dva2',thacc=1,dva=2);
    #dataPreprocessing(D,f'dsThaccInf');
    computeVarAll('ThaccInfdva0',transform=2,predictors=1,quick=False,addSMI=1);
    # replicate experiment
    Xqmt=np.zeros((6,9));Xqmt[1,[1,2]]=1;Xqmt[4,[1,2]]=1
    generateFakeDataHT(age=7,Ms=range(6),P=[[0,9],[9,14],[14,19],[19,28],[28,33],[33,38]],
        Es=[1,1,1,2,2,2],Nd=[55,45,65,55,45,65],Xqmt=Xqmt,N=-1,fn='empExp')

