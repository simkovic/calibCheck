import os,pickle,stan
import numpy as np
import pylab as plt
from matusplotlib import saveStanFit,loadStanFit,ndarray2latextable
from scipy.stats import scoreatpercentile as sap
MD=70 #monitor distance used to compute deg visual angle in the output files
#position of calibration points
CTRUE=np.array([[0,0],[-11,11],[11,11],[11,-11],[-11,-11],[11,0],[-11,0],[0,11],[0,-11]]) # true calibration locations in degrees
#CTRUE=CTRUEDEG/180*np.pi*MD # true calibartion locations in cm
SEED=9 # the seed of random number generator was fixed to make the analyses replicable
TC,TS,F,LX,LY,RX,RY,BX,BY,LD=range(10); RD=12;BD=15
DPI=300 #figure resolution
DPATH='data'+os.path.sep  
DPATHX='/run/media/matus/backup/research/work/K018/code/dataL/'
FIGPATH='../publication/fig/'
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
        if np.ndim(calib[i])<2: stop
    f.close() 
    return calib
    
def getCM(c,ctrue,normEnt=0,onepar=False):
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
            - residual at each calib. location
    '''
    coef=[np.nan,np.nan,np.nan,np.nan]
    resid=np.nan
    #assert(c.shape[0]==9)
    assert(c.shape[1]==2)
    assert(np.all(np.isnan(c[:,0])==np.isnan(c[:,1])))
    sel=~np.isnan(c[:,0])
    if sel.sum()<[3,2][int(onepar)]: return coef,resid,c,c[:,0]*np.nan
    temp=np.zeros(sel.sum())
    for k in range(2):
        if onepar: x=np.ones((sel.sum(),1))
        else:x=np.column_stack([np.ones(sel.sum()),c[sel,k]])
        res =list(np.linalg.lstsq(x,ctrue[sel,k],rcond=None))
        if len(res[1])==0: return [np.nan,np.nan,np.nan,np.nan],np.nan,c,c[:,0]*np.nan
        if onepar: coef[k*2:(k+1)*2]=np.array([res[0][0],0])
        else: coef[k*2:(k+1)*2]=res[0]
        temp+=np.square(ctrue[sel,k]-x.dot(res[0]))#res[1][0]
    #resid=np.max(np.sqrt(temp))
    w=np.power(temp,normEnt)
    w=w/w.sum()
    residVec=np.zeros(c.shape[0])*np.nan
    residVec[~np.isnan(c[:,0])]=np.sqrt(temp) 
    resid=np.sqrt((w*temp).sum())#*(sel.sum()**normEnt)/(sel.sum()**normEnt+(~sel).sum()**normEnt)*2
    assert(np.any(np.isnan(resid))==np.any(np.isnan(coef))) 
    assert(np.all(np.isnan(coef))==np.any(np.isnan(coef)))
    return coef,resid,c,residVec
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
    res=[None,np.inf,c,0]
    for k in range(-1,replacement.shape[2]):
        if k>=0:
            sel=np.isnan(res[2][:,0])
            res[2][sel,:]=replacement[sel,:,k]
        if algo=='local' or algo=='valid': 
            res=list(getCM(np.copy(res[2]),np.copy(ctrue),normEnt=normEnt, onepar=onepar))
            if algo=='local':res[2][res[3]>threshold,:]=np.nan   
        elif algo=='iter':
            res=selCPiter(np.copy(res[2]),np.copy(ctrue),minNrCL=minNrCL,normEnt=normEnt,onepar=onepar,th=threshold)
        elif algo=='avg':
            res=list(getCM(np.copy(res[2]),np.copy(ctrue),normEnt=normEnt, onepar=onepar))
            if repeat in ('block',None) and (res[1]>threshold):res[2][:,:]=np.nan
            elif repeat=='trial' and (res[1]>threshold): res[2][np.argmax(res[3]),:]=np.nan
        #if not enoughCLs(res) and repeat in ('block',None): res[2][:,:]=np.nan
        k+=1
        #if enoughCLs(res) and (np.all(res[1]<threshold) or algo=='valid'): break
        if enoughCLs(res): break
        elif repeat in ('block',None): 
            res[2][:,:]=np.nan 
            if repeat is None: break
    if res[0] is None or not enoughCLs(res): res=[np.nan*np.ones(4),np.nan,np.zeros((c.shape[0],2))*np.nan,k]
    res[3]=k
    return res
    #if not returnPars: return np.int32(~np.isnan(res[2][:,0]))
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

   
def dataPreprocessing(D,fn,thacc=0.5,thvel=10,minDur=0.06,
    dva=0,verbose=False,ncpu=8):  
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

###########################################################
    

def fitMDGP(fn,addSMI=False,transform=0,predictors=0,ssmm=1,eye=2):
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
    lbs=['0','-10'][int(transform==2)]    
    po=['','+oas[k,(m>3)+1]*age[n]+ods[k,1,(m>3)+1]*dist[n,m,p]+ods[k,2,(m>3)+1]*abs(dist[n,m,p])+oms[k,(m>3)+1]*(P[m]+p-10)+mms[k]*(m-2)+oqs[k,(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3))'][predictors]# 
    s2min=[5,1.6][int(transform==2)]
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
        //real<lower=-10,upper=10> oqm[2,E];
        '''+['','''
        real<lower=-10,upper=10> nas[2,E];
        real<lower=-10,upper=10> ods[2,2,E];
        real<lower=-10,upper=10> mms[2];
        real<lower=-10,upper=10> oms[2,E];
        real<lower=-10,upper=10> oqs[2,E];
        real<lower=-10,upper=10> oas[2,E];'''][predictors]+f'''
        real<lower={s2min}> s2[2,E];
        vector<lower=-100,upper=100>[N] t;
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
        for (e in 1:E){{ 
        for (k in 1:2){{
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
                    lps[1]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n]+r[k,(m>3)+1][n].*c[n,m,p,k]+odm[k,1,(m>3)+1]*dist[n,m,p]+odm[k,2,(m>3)+1]*abs(dist[n,m,p])+omm[k,(m>3)+1]*(P[m]+p-10)+mmm[k]*(m-2),{trns}(0.1*(sy[k,(m>3)+1]{po})));
                    lps[2]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n]+odm[k,1,(m>3)+1]*dist[n,m,p]+odm[k,2,(m>3)+1]*abs(dist[n,m,p])+omm[k,(m>3)+1]*(P[m]+p-10)+mmm[k]*(m-2),{trns}(s2[k,(m>3)+1])); 
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
            tmpTemp=t[n]+mmt*(m-2)+omt*(p-10+P[m])+qmt[(m>3)+1]*((m==2)+(m==5))*((p==2)+(p==3));
            if ( (! is_nan(y[n,m,p,1])) ){{ 
                lpsTemp[1]=log_inv_logit(tmpTemp); 
                lpsTemp[2]=log1m_inv_logit(tmpTemp);
                for (k in 1:2){{
                    lpsTemp[1]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n]+r[k,(m>3)+1][n].*c[n,m,p,k]+odm[k,1,(m>3)+1]*dist[n,m,p]+odm[k,2,(m>3)+1]*abs(dist[n,m,p])+omm[k,(m>3)+1]*(P[m]+p-10)+mmm[k]*(m-2),{trns}(0.1*(sy[k,(m>3)+1]{po})));
                    lpsTemp[2]+=normal_lpdf(y[n,m,p,k]|o[k,(m>3)+1,n]+odm[k,1,(m>3)+1]*dist[n,m,p]+odm[k,2,(m>3)+1]*abs(dist[n,m,p])+omm[k,(m>3)+1]*(P[m]+p-10)+mmm[k]*(m-2),{trns}(s2[k,(m>3)+1])); 
                 }}
                onTargetGen[n,m,p]=exp(lpsTemp[1]-log_sum_exp(lpsTemp));
                }}
            else onTargetGen[n,m,p]=0; 
        }}}}}}}}}}'''  
                
             
    pn=['','+nas*age'][predictors]  
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

    print(modeldist)
    dat['nomdist']=(np.array([55,45,65,55,45,65])-55)/10
    sm = stan.build(program_code=modeldist,data=dat,random_seed=SEED) 
    fit = sm.sample(num_chains=6,num_thin=10,num_samples=5000,num_warmup=5000)
    saveStanFit(fit,dat,DPATH+f'sd2L07{fn}{transform}{eye}{predictors}{addSMI}',model=modeldist)
    on=np.ones((2,dat['E']))
    on1=np.ones(2)
    print(model)
    if transform==2: 
        tmp={'mas':on*0,'oas':on*0,'nas':on*0,'mms':on1*0,'oms':on*0,
            'ods':np.ones((2,2,dat['E']))*0,'sm':on*0,'so':on*0,'sy':on*0}
    else:
        tmp={'mas':on*.01,'oas':on*.01,'nas':on*.01,'mms':on1*.01,'oms':on*.01,
            'ods':np.ones((2,2,dat['E']))*.01,'sm':on*.5,'so':on*.5,'sy':on*.5}
    tmp['mt']=0;tmp['t']=np.zeros(y.shape[0])  
         
    if transform==2: 
        tmp={'mas':on*0,'oas':on*0,'nas':on*0,'mms':on1*0,'oms':on*0,
            'ods':np.ones((2,2,dat['E']))*0,'sm':on*0,'so':on*0,'sy':on*0}
    else:
        tmp={'mas':on*.01,'oas':on*.01,'nas':on*.01,'mms':on1*.01,'oms':on*.01,
            'ods':np.ones((2,2,dat['E']))*.01,'sm':on*.5,'so':on*.5,'sy':on*.5}
    tmp['mt']=0;tmp['t']=np.zeros(y.shape[0]).T
    sm = stan.build(program_code=model,data=dat,random_seed=SEED) 
    fit = sm.sample(num_chains=7,num_thin=int(max(ssmm*1,1)),
        num_samples=int(ssmm*500),num_warmup=int(ssmm*500),init=[tmp]*7)
    saveStanFit(fit,dat,DPATH+f'sm2L13{fn}{transform}{eye}{predictors}{addSMI}',model=model)

def generateFakeDataHT(age,Ms,P,Cs=None,Es=None,Nd=None,
    N=-1,K=1000,fn='',savefit=False,numChains=1):
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
    mdat='''
        '''
    model='''data {
        int N,M,E,L; array[L] int P;array[L] int Ms;array[M] int MMs;array[M] int Es;
        //array[L] int<lower=1,upper=9> Cs;
        array[N,L,2,2] real c;array[N] real age;array[M] real nomdist;
        array[2,E] real sy;array[2,E] real sm;array[2,E] real mm;array[2,E] real sr;
        array[2,E] real mr;array[2,E] real nas;array[2,2,E] real ods;array[2] real mms;array[2,2,E] real odm;array[2,E] real omm;array[2] real mmm;
        array[2,E] real oms;array[2,E] real oas;array[2,E] real oqs; array[2,E] real s2;
        real mt,st,nat,mmt,omt;array[E] real qmt;array[E] real onan;array[E] real qnan;
        real mmD,mmmD,qmmD,ommD,emmD,syD,smD,namD,nasD,mmsD,omsD,oasD,anan,pnan,mnan;
        array[2,E] real nam;
    } transformed data{
        array[N,L] int Xqmt;
        for (n in 1:N){
        for (l in 1:L) Xqmt[n,l]=(nomdist[Ms[l]]==-1)*(c[n,l,1,2]>10);}
    } generated quantities{
        array[N,L] int onTarget;array[N,L,2] real y;
        array[N,L] real dist;//hard limits at 30 and 95 cm <lower=-2.5,upper=4>
        array[N,L,2,2] real cr;
        array[2,E] vector[N] o;array[N] real oD;
        vector[N] t;array[2,E] vector[N] r;vector[2] temp;
        array[N] real agevar;
        int nodat;array[L] real lpnodat;
        for (n in 1:N){
            if (age[n]==-7) agevar[n]=uniform_rng(-3,3); 
            else agevar[n]=age[n];
            oD[n]=normal_rng(mmD+namD*agevar[n],exp(smD+nasD*agevar[n]));
        for (e in 1:E){
            
        for (k in 1:2){ 
            r[k,e,n]=normal_rng(mr[k,e],sr[k,e]);
            o[k,e,n]=normal_rng(mm[k,e]+nam[k,e]*agevar[n],(exp(0.1*(sm[k,e]+nas[k,e]*agevar[n]))));}}}

        for (n in 1:N){
            t[n]=normal_rng(mt+nat*agevar[n],st);
        for (l in 1:L){
            lpnodat[l]=onan[Es[Ms[l]]]+pnan*(P[l]-9)+qnan[Es[Ms[l]]]*Xqmt[n,l]+mnan*(Ms[l]-2)+anan*agevar[n];
            nodat=bernoulli_logit_rng(lpnodat[l]);
            if (nodat==1){
                dist[n,l]=4;//0.0/0;
                onTarget[n,l]=0;
                y[n,l,1]=0.0/0;
                y[n,l,2]=0.0/0;
            }else{
            dist[n,l]=normal_rng(oD[n]+nomdist[Ms[l]]+emmD*(Es[Ms[l]]-1)+ommD*(P[l]-9)+mmmD*(Ms[l]-2)+qmmD*Xqmt[n,l],exp(syD+oasD*agevar[n]+omsD*(P[l]-9)+mmsD*(Ms[l]-2)));
            temp[2]=1-inv_logit(lpnodat[l]);
            temp[1]=inv_logit(t[n]+mmt*(Ms[l]-2) +omt*(P[l]-9)+qmt[Es[Ms[l]]]*Xqmt[n,l]);
            if (temp[1]>=temp[2]) onTarget[n,l]=1;
            else onTarget[n,l]=bernoulli_rng(temp[1]/temp[2]);

            for (k in 1:2){
            cr[n,l,1,k]=atan2(c[n,l,1,k],dist[n,l]*10+55)/pi()*180;//translate to deg with dist in cm
            cr[n,l,2,k]=atan2(c[n,l,2,k],dist[n,l]*10+55)/pi()*180;//distractor
            if (onTarget[n,l]==1) y[n,l,k]=normal_rng(o[k,Es[Ms[l]],n]+r[k,Es[Ms[l]],n]*cr[n,l,1,k]+odm[k,1,Es[Ms[l]]]*dist[n,l]+odm[k,2,Es[Ms[l]]]*abs(dist[n,l])+omm[k,Es[Ms[l]]]*(P[l]-9)+mmm[k]*(Ms[l]-2),exp(0.1*(sy[k,Es[Ms[l]]]+oas[k,Es[Ms[l]]]*agevar[n]+ods[k,1,Es[Ms[l]]]*dist[n,l]+ods[k,2,Es[Ms[l]]]*abs(dist[n,l])+oms[k,Es[Ms[l]]]*(P[l]-9)+mms[k]*(Ms[l]-2)+oqs[k,Es[Ms[l]]]*Xqmt[n,l])));
            else y[n,l,k]=normal_rng(o[k,Es[Ms[l]],n]+odm[k,1,Es[Ms[l]]]*dist[n,l]+odm[k,2,Es[Ms[l]]]*abs(dist[n,l])+omm[k,Es[Ms[l]]]*(P[l]-9)+mmm[k]*(Ms[l]-2),exp(s2[k,Es[Ms[l]]]));
            }}}}}'''   
    dat={'M':len(np.unique(Ms)),'MMs':np.unique(Ms)+1,'Ms':np.array(Ms)+1,'L':len(P),
        'P':P,'E':2,'nomdist':(np.array(Nd)-55)/10,'Es':Es}
    w=loadStanFit(f'data/sd2L07FixTh1_0dva22211',excludeChains=[])
    for k in w.keys():
        if not k[-1]=='+' and not k in ('o'):
            #print(k,w[k].shape)
            dat[k+'D']=np.squeeze(np.median(w[k],0))
            assert(np.all(~np.isnan(dat[k+'D'])))
    w=loadStanFit(f'data/sm2L13FixTh1_0dva22211',excludeChains=[3],computeRhat=False)
    if N==-1:dat['N']=w['N+'];
    else:dat['N']=N
    dat['yold']=w['y+'];dat['distold']=w['dist+']
    dat['age']=(age-7)*np.ones(dat['N'])
    for k in w.keys():
        if not k[-1]=='+' and not k.endswith('Gen') and not k.endswith('Temp') and not k in ('o','t','r'):
            #print(k,w[k].shape) 
            dat[k]=np.squeeze(np.median(w[k],0))
            assert(np.all(~np.isnan(dat[k])))
    dat['c']=[]
    for n in range(dat['N']):
        dat['c'].append([])
        for i in range(len(Cs)):
            if type(Cs[i]) is int:
                dat['c'][-1].append([list(chunits(CTRUE[Cs[i]-1,:],dva=0)),[0,0]])
            else: 
                if np.mod(n,2)==0: dat['c'][-1].append([np.array(Cs[i]),-np.array(Cs[i])])
                else: dat['c'][-1].append([-np.array(Cs[i]),np.array(Cs[i])])
    dat['c']=np.array(dat['c'])
    assert(np.all(dat['c'][:,0,1,:]==0))
    #for l in range(len(Ms)):if dat['pnan'][l]==-1:dat['pnan'][l]=np.exp(dat['lpnan'])
    sm = stan.build(program_code=model,data=dat,random_seed=SEED) 
    fit = sm.fixed_param(num_chains=numChains,num_samples=K)
    assert(not np.all(np.isnan(fit['y'][:,:,:,0])))
    if savefit:
        w=saveStanFit(fit,dat,DPATHX+'gen'+fn,model=model,computeRhat=False)
    else: 
        np.save(DPATHX+'gcR'+fn,fit['y'])
        #np.save(DPATHX+'dist'+fn,fit['dist'])
        np.save(DPATHX+'cl'+fn,fit['cr'])
    del sm, fit
def fakedataCalibrationWorker(L,fn,nCP=50,ect=10):
    import itertools
    y=np.rollaxis(np.load(DPATHX+f'gcR{L}{fn}.npy'),-1,0)
    #o=np.rollaxis(np.load(f'data/genoLfkPref{L}{fn}.npy'),-1,0)
    c=np.rollaxis(np.load(DPATHX+f'cl{L}{fn}.npy'),-1,0)
    #c=np.vstack([c,c])
    als=['avg','local','valid']
    ks=[3,0]#[3,-1,0]
    I=np.reshape(np.arange((len(als))*3*(len(ks)+2)),[len(als),3,len(ks)+2])
    gzh=np.zeros((y.shape[0],y.shape[1],I.size,15,2))*np.nan
    mask=np.zeros(list(y.shape[:2])+[L,I.size],dtype=np.bool_)
    
    for i in range(y.shape[0]):
        todo=[]
        for n in range(y.shape[1]):
            #rpl=y[i,n,L+11:2*L+11,:,np.newaxis]
            rpl=np.rollaxis(np.reshape(np.copy(y[i,n,np.newaxis,L:3*L,:]),(2,-1,2)),0,3)

            for a,r,k in itertools.product(range(len(als)),range(I.shape[1]),range(len(ks))):
                res=selCPalgo(y[i,n,:L,:],c[i,n,:L,0,:],threshold= [4,1.5,np.inf][a],minNrCL=ks[k],#[3.5,3.5,np.inf,1.75]
                    repeat=[None,'block','trial'][r],algo=als[a],
                    replacement=[np.zeros((0,0,0)),rpl][int(r>0)])#1.75 
                if k==0: mask[i,n,:,I[a,r,0]]=~np.isnan(res[2][:,0])
                if not int(res[3]) in (0,1,2): print(res[3])
                assert(int(res[3]) in (0,1,2))
                gzh[i,n,I[a,r,k],:5,:]=cmPredict(y[i,n,(3*L+res[3]):(3*L+res[3]+5),:],res[0])
                gzh[i,n,I[a,r,k],5:,:]=cmPredict(y[i,n,(3*L+7+10*res[3]):(3*L+7+10*(res[3]+1)),:],res[0])
    for i in range(y.shape[0]):
        #CP 
        for a,r in itertools.product(range(len(als)),range(I.shape[1])):
            aa=np.vstack([y[i-1,:nCP,:L,0][mask[i-1,:nCP,:,I[a,r,0]]],y[i-1,:nCP,:L,1][mask[i-1,:nCP,:,I[a,r,0]]]]).T
            ctrue=np.vstack([c[i-1,:nCP,:L,0,0][mask[i-1,:nCP,:,I[a,r,0]]],
                            c[i-1,:nCP,:L,0,1][mask[i-1,:nCP,:,I[a,r,0]]]]).T
            resCP=getCM(aa,ctrue)
            for n in range(y.shape[1]): 
                gzh[i,n,I[a,r,-1],:,:]=cmPredict(y[i,n,(3*L+37):,:],resCP[0])
                #NP+CP        
                #tmp=cmPredict(y[i,n,3*L+2:(3*L+17),:],resCP[0])#TODO add res[3]
                for m in range(gzh.shape[3]):     
                #if np.isnan(gzh[i,n,I[a,r,0],m,0]): gzh[i,n,I[a,r,-2],m,:]=tmp[m,:] 
                #else:gzh[i,n,I[a,r,-2],m,:]=gzh[i,n,I[a,r,0],m,:]
                    for k in range(I.shape[2]):
                        if np.linalg.norm(gzh[i,n,I[a,r,k],m,:]-c[i,n,(3*L+37)+m,0,:])>ect and np.linalg.norm(gzh[i,n,I[a,r,k],m,:]-c[i,n,(3*L+37)+m,1,:])>ect:
                            gzh[i,n,I[a,r,k],m,:]=np.nan    
    np.save(DPATHX+f'gcC{L}{fn}',gzh) 
    return

def fakedataCalibration(fn,b,S,ncpu=4,suf=''):
    from multiprocessing import Pool
    pool=Pool(ncpu)
    res=[]
    for i in range(13,S):
        temp=pool.apply_async(fakedataCalibrationWorker,[b,f'{fn}{i:02d}'])
        res.append(temp)
    pool.close()  
    from time import time,sleep
    tot=len(res)
    t0=time();prevdone=-1
    while True:
        done=0
        for r in res:
            done+=int(r.ready())
            if done>prevdone:
                prevdone=done
                print(done,' out of ',tot)
        sleep(1)
        if done==tot: break
    
    gz=[]
    for i in range(S):
        gz.append(res[i].get())
    return
    #gz=np.reshape(gz,[3,51])
    np.save(DPATHX+f'gcC{b}{fn}{suf}',gz)

def fakedataTTestPowerWorker(fn,ax,alpha,beta,test):
    def evalPowerTtest(indat):
        dr=np.ones((1,indat.shape[1]))
        dr[0,1::2]=-1
        dat=dr*np.copy(indat)# remove side balancing
        p=np.zeros(dat.shape[0])*np.nan
        for i in range(dat.shape[0]):
            sel=~np.isnan(dat[i,:])
            if sel.sum()>2:
                p[i]=ttest_1samp(dat[i,sel],0,alternative='greater')[1]
                if np.isnan(p[i]):
                    print(dat[i,sel]);stop
        res=(p<alpha).mean()>(1-beta)
        #del dr,p
        return res
    def evalPowerTOST(dat):
        p=np.zeros((dat.shape[0],2))*np.nan
        for i in range(dat.shape[0]):
            sel=~np.isnan(dat[i,:])
            if sel.sum()>2:
                p[i,0]=ttest_1samp(dat[i,sel],es/2,alternative='less')[1]
                p[i,1]=ttest_1samp(dat[i,sel],-es/2,alternative='greater')[1]
                if np.isnan(p[i,0]) or np.isnan(p[i,1]):
                    print(dat[i,sel]);stop
        return np.logical_and(p[:,0]<alpha,p[:,1]<alpha).mean()>(1-beta)
        
    from scipy.stats import ttest_1samp
    if test=='ttest':evalPower=evalPowerTtest
    elif test=='TOST':evalPower=evalPowerTOST
    #gz=np.load(DPATHX+fn+'.npy')#'data/gz{a}{e}{ax}{suf}'
    n=-np.ones((25,36,6),dtype=np.int32)
    print(fn) 
    for esi in range(n.shape[0]):
        gz=np.load(DPATHX+fn+f'{esi:02d}.npy')
        for k in range(n.shape[1]):
            for m in range(n.shape[2]):
                if m==n.shape[2]-1:g=np.nanmean(gz[:,:,k,m:,ax],axis=2)
                else: g=np.copy(gz[:,:,k,m,ax])
                un=200;ln=5
                if np.all(np.isnan(g[:,:ln])):
                    n[esi,k,m]=-9
                    continue
                elif np.all(np.isnan(g[:,:un])):
                    n[esi,k,m]=-8
                    continue
                if not evalPower(g[:,:un]): n[esi,k,m]=201
                if evalPower(g[:,:ln]): n[esi,k,m]=0
                while n[esi,k,m]==-1:
                    cn=(un+ln)//2
                    if evalPower(g[:,:cn]):un=cn
                    else:ln=cn
                    if un == ln+1: n[esi,k,m]=un
    return n
def fakedataTTestPower(ax,fn,ncpu=4,alpha=0.05,beta=.2,test='ttest',ecthreshold=10):
    from multiprocessing import Pool
    pool=Pool(ncpu)
    res=[]
    bs=[3,5,9]
    aas=[0,4,7,10]
    for b in bs:
        for a in aas:
            temp=pool.apply_async(fakedataTTestPowerWorker,[f'gcC{b}p{a}m{fn}',ax,alpha,beta,test])
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
    n=[]
    k=0
    for b in range(len(bs)):
        n.append([])
        for a in range(len(aas)):
            n[-1].append(res[k].get())
            k+=1
    np.save(DPATH+f'ssnfp{fn}{test}',n)
    
def label2index(lbl):
    if lbl[-3:]=='CP9':lbl=lbl[:-1]
    r='nbt'.index(lbl[0])
    a='ALV'.index(lbl[1])
    k=['3','K','CP+NP','CP'].index(lbl[2:])
    I=np.reshape(np.arange(4*3*4),[4,3,4])
    return I[a,r,k]
def label2ls(lbl): 
    if lbl[-2:]=='CP':return 'ok-'
    #if lbl[-3:]=='CP9':return 'ok-'
    r='nbt'.index(lbl[0])
    a='ALV'.index(lbl[1])
    g='K3'.index(lbl[2])
    return ['^-','x--','v:'][r]+'rgbmyc'[a+g*3]

def fakedataEracc(dim,e,lbls=None,):
    def computeExr(dat):
        if dat.ndim==4: return np.median(np.all(np.isnan(dat[:,:,:,0]),axis=2).mean(1))
        else:return np.median(np.isnan(dat[:,:,0]).mean(1))
    def computeAcc(dat,avg=False):
        if dat.ndim==4: return np.median(np.nanmean(np.linalg.norm(dat,axis=3),axis=(1,2)))
        else: return np.median(np.nanmean(np.linalg.norm(dat,axis=2),1))
        
    if lbls is None:     
        lbls=['nLK','bLK','tLK','nAK','bAK','tAK','nVK','bVK','tVK','nL3','bL3',
            'tL3','nA3','bA3','tA3','nV3','bV3','tV3','nLCP','nACP','nVCP']
    res=np.zeros((25,3,4,3,len(lbls),2))*np.nan
    suf=['Tob','Smi'][e]+['Hor','Ver'][dim]
    for a in range(res.shape[2]):
        for h in range(res.shape[3]): 
            fn=f'{[3,5,9][h]}p{[0,4,7,10][a]}m'+suf
            for esi in range(res.shape[0]):
                d=[[],[],[],[]]
                gz=np.load(DPATHX+f'gcC{fn}{esi:02d}.npy')
                c=np.rollaxis(np.load(DPATHX+f'cl{fn}{esi:02d}.npy'),-1,0)
                for lbli,lbl in enumerate(lbls):
                    for b in range(2):
                        compute=[computeExr,computeAcc][b]
                        for k in range(2):  res[esi,k,a,h,lbli,b]=compute(gz[:,:,label2index(lbl),k,:]-c[:,:,(3*[3,5,9][h]+37)+k,0,:])
                        res[esi,2,a,h,lbli,b]=compute(gz[:,:,label2index(lbl),5:15,:]-c[:,:,(3*[3,5,9][h]+37)+5:,0,:]) 
            del gz
    np.save(DPATH+'eracc'+suf,res)  
    np.savetxt(DPATH+'eraccLbls',lbls,fmt='%s')  
    
####################################################################

def plotPowerSel(lblNP,lblCP,e):    
    gridspec = dict(wspace=0.05, width_ratios=[1, 1,0.09,1,1,0.09,1,1],hspace=.04)
    fig, axs = plt.subplots(figsize=(16,12),dpi=DPI,nrows=4, ncols=8, gridspec_kw=gridspec)
    ess=np.load(f'data/ess.npy')

    for dim in range(2):
        suf=['Tob','Smi'][e]+['Hor','Ver'][dim]
        n=np.load(DPATH+f'ssnfp{suf}ttest.npy')
        for a in range(n.shape[1]):
            axs[a][2].set_visible(False)
            axs[a][5].set_visible(False)
            for p in range(3):
                ax=axs[[3,0,1,2][a]][3*p+dim]
                for h in range(n.shape[0]):
                    d=np.float32(n[h,a,:,label2index(lblNP),[0,1,5][p]])
                    ax.plot(ess,d,'k',alpha=[.2,.5,1][h])
                    d=np.float32(n[h,a,:,label2index(lblCP[:-1]),[0,1,5][p]])
                    if '359'[h]==lblCP[-1]:
                        ax.plot(ess,d,'r--',alpha=1)
                
                if p==0 and dim==0:
                    if a==0:ax.set_ylabel(f'mixed age')
                    else: ax.set_ylabel(f'{[4,7,10][a-1]}-month-old')
                elif dim==1: ax.set_yticklabels([])
        
                if p==1 and dim==0 and a==3:ax.legend(['NP3','NP5',lblCP[-3:],'NP9'])
                ax.grid()
                if p<2:ax.set_xlim([0.25,1.5]);
                else: ax.set_xlim([0.1,0.7]);
                #plt.xlim([ess[0],ess[-1]])
                ax.set_ylim([0,200])
                h=['B1','B2','AVG'][p];tmp=['horizont','vertic'][dim]
                if a==1: ax.set_title(['Tobii','SMI'][e]+f', {tmp}al,'+' $H_\\mathrm{'+h+'}$')
                if a!=0: ax.set_xticklabels([])
    plt.savefig(FIGPATH+f'powerSel'+['TOB','SMI'][e]+'.png',dpi=DPI, bbox_inches='tight')
    
def plotPowerAll(lblCP,e):    
    gridspec = dict(wspace=0.05, width_ratios=[1, 1,1,0.09,1,1,1],hspace=.04)
    ess=np.load(f'data/ess.npy')    
    for p in range(3):
        for pref in 'tbna':
            fig, axs = plt.subplots(figsize=(16,12),dpi=DPI,nrows=4, ncols=7, gridspec_kw=gridspec)
            for dim in range(2):
                suf=['Tob','Smi'][e]+['Hor','Ver'][dim]
                n=np.load(DPATH+f'ssnfp{suf}ttest.npy')
                for a in range(n.shape[1]):
                    axs[a][3].set_visible(False)
                    
                    for h in range(n.shape[0]):
                        ax=axs[[3,0,1,2][a]][4*dim+h]
                        if pref!='a':
                            lbls=['VK','AK','LK','V3','A3','L3']
                            lbls=list(map(lambda x: pref+x,lbls))
                        else:lbls=['tL3','bA3','nL3',lblCP]
                        for lbl in lbls:
                            d=np.float32(n[h,a,:,label2index(lbl),[0,1,5][p]])
                            ax.plot(ess,d,label2ls(lbl))
                
                        if h==0 and dim==0:
                            if a==0:ax.set_ylabel(f'mixed age')
                            else: ax.set_ylabel(f'{[4,7,10][a-1]}-month-old')
                        elif dim==1: ax.set_yticklabels([])
            
                        if h==0 and dim==0 and a==1:ax.legend(lbls)
                        ax.grid()
                        ax.set_xlim([0,1.5]);
                        #ax.set_xlim([0.1,0.7]);
                        #plt.xlim([ess[0],ess[-1]])
                        ax.set_ylim([0,200])
                        tmp=['horizont','vertic'][dim]
                        if a==1: ax.set_title(['3','5','9'][h]+f' cal. loc., {tmp}al')
                        if a!=0: ax.set_xticklabels([])
            plt.savefig(FIGPATH+f'e'+['TOB','SMI'][e]+f'h{p+1}{pref}.png',dpi=DPI, bbox_inches='tight')
  
def plotEracc(dim):
    gridspec = dict(wspace=0.05, width_ratios=[1, 1,1,0.11,1,1,1],hspace=.04)
    fig, axs = plt.subplots(figsize=(12,9),dpi=DPI,nrows=4, ncols=7, gridspec_kw=gridspec)
    esi=10
    lbls=np.loadtxt(DPATH+'eraccLbls',dtype=np.str_).tolist()#,fmt='%s')
    resTob=np.load(DPATH+'eraccTob'+['Hor','Ver'][dim]+'.npy')
    resSmi=np.load(DPATH+'eraccSmi'+['Hor','Ver'][dim]+'.npy')
    for b in range(2):
        for a in range(resTob.shape[2]):
            axs[a][3].set_visible(False)
            for h in range(resTob.shape[3]):
                #plt.subplot(3,6,a*6+h+1+b*3)
                ax=axs[[3,0,1,2][a]][h+b*4]
                for k in range(resTob.shape[1]):
                    ax.plot(resTob[esi,k,a,h,:-2,b],np.arange(len(lbls)-2),'x-'+'rgb'[k])
                    if k==0: ax.plot(resSmi[esi,k,a,h,:-2,b],np.arange(len(lbls)-2),'x-'+'k'[k])
                if a==1: ax.set_title(['Exclusion Rate','Accuracy'][b]+f'\n{[3,5,9][h]} Calibration Locations')
                if h==0: 
                    if b==0:
                        if a>0: ax.set_ylabel(f'{[0,4,7,10][a]} months')
                        elif a==0: ax.set_ylabel(f'mixed age')
                    ax.set_yticklabels(lbls[:-3]+['CP']);
                else:ax.set_yticklabels([]);
                ax.set_yticks(range(len(lbls[:-2])));
                if h+b*4==6:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_ticks_position('both')
                    ax.set_yticklabels(lbls[:-3]+['CP']);
                ax.grid(True)
                if b:
                    ax.set_xticks(np.linspace(.5,3.5,7));
                    ax.set_xlim([1.2,3.2])
                else: ax.set_xlim([-.1,1.1])
                if not a==0: ax.set_xticklabels([])
                if a==1 and h==2 and b==0: ax.legend(['Tobii,  $H_\\mathrm{B1}$','SMI, $H_\\mathrm{B1}$','Tobii, $H_\\mathrm{B2}$','Tobii, $H_\\mathrm{AVG}$'],loc='lower left',fontsize=7)
                if a==3 and h==1:
                    print(resTob[esi,2,a,h,lbls.index('bAK'),b]) 
                    print(resSmi[esi,2,a,h,lbls.index('bAK'),b]) 
    plt.savefig(FIGPATH+f'eracc'+['Hor','Ver'][dim]+'.png',dpi=DPI, bbox_inches='tight')


if __name__=='__main__':
    sampleDescr(2);stop
    # loading and preprocessing
    fns=checkFiles()             
    D=loadCalibrationData(fns)
    with open(DPATH+'D.out','wb') as f: pickle.dump(D,f)
    with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    dataPreprocessing(D,f'dsFixTh1_0dva2',thacc=1,dva=2);
    #fit model of data-generating process
    fitMDGP('ThaccInfdva0',transform=2,predictors=1,ssmm=2,addSMI=1);
    # generate fake data and compute power and accuracy
    ttn=10
    ess=np.linspace(0,1.5,26)[1:]
    np.save(DPATH+'ess',ess)
    for e in [0,1]:#two eyetrackers
        for ax in [0,1]: # horizontal and vertical axis
            fna=['Tob','Smi'][e]+['Hor','Ver'][ax]
            for b in [3,5,9]:# nr of locations
                for a in [0,4,7,10]:# age, 0 = mixed age
                    for i in range(ess.size): 
                        coord=[0,0]
                        coord[ax]=ess[i]  
                        generateFakeDataHT(Ms=b*[0]+b*[1]+b*[2]+ [1,2,3,4,5,6,7]+ttn*[1]+ttn*[2]+ttn*[3]+[0,1,2,3,4]+ttn*[0],P=list(range(b))+b*[b]+b*[b+1]+list(range(b,b+7))+list(range(b,b+10))+list(range(b+1,b+11))+list(range(b+2,b+12))+list(range(5))+list(range(10)), Cs=3*(np.arange(b)+1).tolist()+(7+30+5+10)*[coord],N=200,fn=f'{b}p{a}m{fna}{i:02d}',Es=8*[e+1],age=a,K=1000)
                    fakedataCalibration(f'p{a}m{fna}',b,ess.size,ncpu=4)
                    
            fakedataTTestPower(ax,fna,ncpu=3)#,test='TOST');stop
            fakedataEracc(ax,e)
    # generate figures
    for i in range(2):
        plotPowerSel('nL3','nLCP5',i)
        plotEracc(i)
        
    # supplemental material
    figurePreproc()#figure S1
    sampleDescr(2)
    # section 3 and figure S2
    #replicate experiment with infants
    Xqmt=np.zeros((6,9));Xqmt[1,[1,2]]=1;Xqmt[4,[1,2]]=1
    generateFakeDataHT(age=7,Ms=range(6),P=[[0,9],[9,14],[14,19],[19,28],[28,33],[33,38]],Es=[1,1,1,2,2,2],Nd=[55,45,65,55,45,65],Xqmt=Xqmt,N=-1,fn='empExp')
    for i in range(2): plotPowerAll('nLCP',i) #section 4
