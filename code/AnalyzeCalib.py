import os,pystan,pickle
import numpy as np
import pylab as plt
from matusplotlib import ndarray2latextable,figure,subplot
from scipy.stats import scoreatpercentile as sap
MD=70 #monitor distance used to compute deg visual angle in the output files
#position of calibration points
CTRUE=np.array([[0,0],[-11,11],[11,11],[11,-11],[-11,-11],[11,0],[-11,0],[0,11],[0,-11]]) # true calibration locations in degrees
#CTRUE=CTRUEDEG/180*np.pi*MD # true calibartion locations in cm
SEED=5 # the seed of random number generator was fixed to make the analyses replicable
TC,TS,F,LX,LY,RX,RY,BX,BY,LD=range(10); RD=12;BD=15

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
    import os
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
    if not ofn==None:plt.savefig(ofn+'.png')
       
def checkFiles(plot=False):
    '''checks the congruency between meta-data and log files'''
    opath=os.getcwd()[:-4]+'output'+os.path.sep+'good'+os.path.sep
    vpinfo=np.int32(np.loadtxt(opath[:-5]+'vpinfo.res',delimiter=','))

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
    opath=os.getcwd()[:-4]+'output'+os.path.sep+'good'+os.path.sep
    vpinfo=np.int32(np.loadtxt(opath[:-5]+'vpinfo.res',delimiter=','))
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
    
def extractFixations(inG,eye,thvel=10,hz=60,minDur=0.3,dva=0): 
    def _chunits(inn,dva,i):
        assert(i==0 or i==1)
        cm=inn/180*np.pi*MD
        if dva==0: x=cm #deg vis angle at 180/pi cm distance
        elif dva==1: x=np.arctan(cm/57.5)/np.pi*180
        elif dva==2: 
            x=np.arctan((cm-np.nanmean(G[~sel,LD+i+3*eye],0))/
                np.nanmean(G[~sel,LD+2+3*eye],0))/np.pi*180    
        elif dva==3: 
            x=np.arctan((cm-G[~sel,LD+i+3*eye])/
                G[~sel,LD+2+3*eye])/np.pi*180
        elif dva==4: #deg vis angle at 180/pi cm distance
            x=57.5*(cm-G[~sel,LD+i%2+3*eye])/G[~sel,LD+2+3*eye] 
        elif dva==5 or dva==6 or dva==7:
            x=np.arctan(cm/[57.5,47.5,67.5][dva-5])/np.pi*180
        return x
    
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
        x=_chunits(G[~sel,ii],dva,i)
        C[~sel,i]=_chunits(C[~sel,i],dva,i)
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
            #TODO check whether level of nonzero() output has more than one elements
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
            1- visual angle constant head pos; 
            2 - visual angle 
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
    I=np.load(DPATH+fn)
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
    print(res.shape)
    top=[]
    for p in range(2):
        for e in ['L','R','B']:
            if p==1 and e=='L': top.extend(['$m$','ET','$d$'])    
            for i in range(3): top.append(e+str(i+1))
    res= np.array([top]+list(res),dtype=object)
    print(res.shape)
    ndarray2latextable(res,decim=0,hline=[0,3,6,9,12,15],
        nl=0,vline=[2,5,8,9,10,11,14,17]) 
def figureSample(fn,dev=0):
    I=np.load(DPATH+fn+'.npy')
    
    CLRS=['0.3','0.5','0.7']#['k','gray','w']
    xtcs=[]
    figure(size=3,aspect=0.6,dpi=400)
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
            ax.set_xticklabels(['L','R','B']*3)
            
    #plt.show()
    plt.savefig('../publication/figs/%s%d.png'%(fn,dev),bbox_inches='tight')       
                

 
def computePA(suf,docompile=True,short=False,dev=None,m=None):
    ''' computes accuracy estimates '''
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
    if dev is None: devs=range(2)
    else: devs=[dev]
    if m is None: ms=range(2)
    else: ms=[m]
    for dev in devs:
        for eye in range(3):
            for m in ms:
                y=ds[dev,:,m+1,eye,:5,:2]
                c=ds[dev,:,m+1,eye,:5,2:4]
                assert(np.all(np.isnan(y[:,:,0]).sum(1)<6))
                sel=~(np.isnan(y[:,:,0]).sum(1)>2)
                age=np.load(DPATH+'age.npy')[sel]
                age=(age-210)/120
                dist=ds[dev,:,m+1,eye,:5,6]
                dist=(dist[sel,:]-57.5)/10
                #c=ds[dev,:,m+1,eye,:5,2:4]
                dat={'y':y[sel,:,:],'N':sel.sum(),'c':c[sel,:],'age':age,'dist':dist}
                if short: doi=[1,5,9,13]
                else: doi=range(16) 
                for i in doi:
                    print(dev,eye,m,i)
                    fit = sms[i].sampling(data=dat,iter=6000,
                        chains=6,thin=10,warmup=3000,n_jobs=6,seed=SEED)
                    with open(DPATH+f'd{dev}e{eye}m{m}i{i}{suf}.stanfit','wb') as f: 
                        pickle.dump(fit,f,protocol=-1)               
def tablePA(fn,m=1,dev=0,novelLocations=False,dva=0,units='deg',
    plot=0,pref='a',legend=False):
    ''' prints code for latex table to console
        table shows accuracy estimates
        dva: 0 - euclidean metric
            1- angular metric 
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
                with open(DPATH+f'sm{i}.pkl','rb') as f: sm=pickle.load(f)
                try:
                    with open(DPATH+f'd{dev}e{eye}m{m}i{i}{fn}dva{dva}.stanfit','rb') as f: fit=pickle.load(f)
                except: continue
                #print(fit);bla
                smr=fit.summary()
                sr=smr['summary'][:-1,-1]
                inds=(sr>1.1).nonzero()[0]
                if len(inds)>0:
                    print(dev,eye,top[2+i],smr['summary_rownames'][inds],sr[inds])
                    if eye==0: lchat=np.nan
                    elif eye==1: rchat=np.nan
                    res[1+eye,2+i]='-';continue
                else: print(dev,eye,i,top[2+i],'CONVERGED') 
                #print(f'd{dev}e{eye}m{m}i{i}',len(inds))
                w=fit.extract()
                o=np.mean(w['o'],axis=0)
                
                if o.ndim==1: o=o[AX,:]
                slope= o[:,AX,2:] 
                if slope.ndim==2: slope=slope[:,:,AX]
                #sel=~np.all(np.isnan(ds[dev,:,m+1,eye,:5,0]),axis=1)
                sel=~(np.isnan(ds[dev,:,m+1,eye,:5,0]).sum(1)>2)
                age=np.load(DPATH+'age.npy')[sel]
                age=(age-210)/120
                y=ds[dev,sel,0,eye,:,:2]
                ctrue=ds[dev,sel,0,eye,:,2:4]
                dist=ds[dev,:,0,eye,:,6]
                dist=(dist[sel,:]-57.5)/10
                chat= o[:,AX,:2]+slope*y
                if 'ady' in w.keys(): 
                    ady=np.mean(w['ady'],0)
                    chat+= (ady[AX,AX,:]* dist[:,:,AX])
                if eye<2: 
                    lrchat[eye,sel,:,:2]=chat
                    lrchat[eye,sel,:,2:4]=ctrue
                    lrchat[eye,sel,:,4:]=ds[dev,sel,0,eye,:,4:]
            elif eye==3: 
                chat=np.nanmean(lrchat[:,:,:,:2],axis=0)
                sel=~np.all(np.isnan(chat[:,:,0]),axis=1)
                chat=chat[sel,:,:]
                ctrue=np.nanmean(lrchat[:,sel,:,2:4],axis=0)
            if np.all(np.isnan(chat)):
                res[1+eye,2+i]='-';continue
            if eye<3: dsd=ds[dev,sel,0,eye,:,4:7]
            elif eye==3: dsd=np.nanmean(lrchat[:,sel,:,4:],axis=0)
            if dva==0 or dva==4: 
                tempcm=np.sqrt(np.sum(np.square(ctrue-chat),axis=2))
                if units=='cm': temp=tempcm
                elif units=='deg':
                    ctruecm=ctrue+dsd[:,:,:2];chatcm=chat+dsd[:,:,:2];
                    dctrue2=np.sum(np.square(ctruecm),axis=2)+np.square(dsd[:,:,2])
                    dchat2=np.sum(np.square(chatcm),axis=2)+np.square(dsd[:,:,2])
                    temp=np.arccos((dctrue2+dchat2-np.square(tempcm))/2/
                        np.sqrt(dctrue2*dchat2))/np.pi*180
            #elif dva==1: 
            #    chatcm=np.tan(chat/180*np.pi)*57.5
            #    ctruecm=chatcm=np.tan(ctrue/180*np.pi)*57.5
            #    tempcm=np.sqrt(np.sum(np.square(ctruecm-chatcm),axis=2))
            #    temp=np.arctan(tempcm/57.5)/np.pi*180
            elif dva==1 or dva==2 or dva==3 or dva==5:
                if dva==1:ddd=np.array([0,0,57.5])[AX,AX,AX,:] 
                elif dva==2:ddd=np.nanmean(dsd,1)[:,AX,AX,:]
                elif dva==3:ddd=dsd[:,:,AX,:]
                elif dva==5: ddd=np.array([0,0,[57.5,47.5,67.5][m+1]])[AX,AX,AX,:] 
                
                chatcm=np.tan(chat/180*np.pi)*ddd[:,:,:,2]+ddd[:,:,0,:2]
                ctruecm=np.tan(ctrue/180*np.pi)*ddd[:,:,:,2]+ddd[:,:,0,:2]
                tempcm2=np.sum(np.square(ctruecm-chatcm),axis=2)
                dctrue2=np.sum(np.square(ctruecm),axis=2)+np.square(ddd[:,:,0,2])
                dchat2=np.sum(np.square(chatcm),axis=2)+np.square(ddd[:,:,0,2])
                temp=np.arccos((dctrue2+dchat2-tempcm2)/2/np.sqrt(dctrue2*dchat2))
                if units=='deg': temp=temp/np.pi*180
                elif units=='cm': temp=np.sqrt(tempcm2)
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
        print(top[2+i],'acc right eye - acc left eye: m= %.3f, 95p CI (%.3f,%.3f), r=%.3f'%(mm,mm-1.96*se,mm+1.96*se, np.corrcoef(lracc[1,~sel],lracc[0,~sel])[0,1]))      
    sel=resout==''
    resout[sel]=np.nan
    if plot>0:
        suf=['','N'][int(novelLocations)]
        ffn=pref+['Tob','Smi'][dev]+['55','45','65'][m+1]+f'Dva{dva}{units}{suf}'
        figureAccuracy(resout,ffn,short=plot==2,legend=legend,dev=dev)
    else: 
        ndarray2latextable(res.T,decim=0,hline=[1,5,9,13],nl=1); 
        return resout


def figureAccuracy(res,fn,short=False,legend=False,dev=0):
    ''' plots accuracy estimates'''
    plt.close('all')
    if short: res=res[:,[0,1,3,7,11,15],:]
    clrs=['g','r','y','b']
    if short: figure(size=2,aspect=0.8,dpi=400)
    else: figure(size=4,aspect=0.7,dpi=400)
    #else: plt.figure(figsize=(12,4),dpi=400)
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
        if short:
            plt.plot(xs+ofs-0.05,res[col+k*4,2:,5],mfc=clrs[col-1],
                mec=clrs[col-1],marker='_',lw=0) 
            mm=se=res[col+k*4,2:,5];se=res[col+k*4,2:,6]
            plt.plot(x-0.05,[mm-1.96*se,mm+1.96*se],color=clrs[col-1],alpha=0.7)     
        handles.append(out[0])
        plt.plot()
    ax=plt.gca()
    ax.set_xticks(xs+0.1);
    mps=['complete pooling','no pooling','hierarchical', 'hier. with predictors']
    if not short: 
        ax.set_xticklabels(['1s','2s','1sc','2sc']*4)
        plt.xlabel(([7,11][int(short)]*'   ').join(mps))
    else: ax.set_xticklabels(mps);
    plt.yticks(np.arange(0,5,0.5))
    plt.ylim([0,5])
    plt.grid(True,axis='y')
    #if not k: plt.xlabel('LC model')
    plt.ylabel(['Tobii X3 120','SMI Redn'][dev]+'\nAccuracy in degrees')
    if legend:plt.legend(handles[::-1],lbls[::-1],loc=1)
    #plt.title(['Tobii X3 120','SMI Redn'][k])
    plt.savefig('../publication/figs/%s.png'%fn,bbox_inches='tight') 
    plt.close('all')   
                 
def computeVar(fn,includePredictors=True,dev=None,dva=0,doCompile=True):
    ''' computes variance estimates with the three-level models'''
    pred=int(includePredictors)
    model='''
    data {
        int<lower=0> N; //nr subjects
        vector[2] y[N,3,9];
        vector[2] c[N,3,9];
        real age[N];
        real dist[N,3,9];
    }parameters{
        vector<lower=-100,upper=100>[3] o[N,3];
        vector<lower=0,upper=100>[2] sy;
        corr_matrix[2] ry;
        vector<lower=0,upper=100>[3] so;
        corr_matrix[3] ro;
        vector<lower=-100,upper=100>[3] mo[N];
        vector<lower=0,upper=100>[3] sm;
        corr_matrix[3] rm;
        vector<lower=-100,upper=100>[3] mm;'''+['','''
        vector<lower=-100,upper=100>[3] nam;
        vector<lower=-100,upper=100>[2] odm;
        vector<lower=0,upper=10>[2] nas;
        vector<lower=0,upper=10>[2] ods;
        vector<lower=0,upper=10>[2] mms;
    }transformed parameters{
        vector[3] tnas; vector[3] tmms;
        tnas[1]=nas[1];tnas[2]=nas[2];tnas[3]=0;
        tmms[1]=mms[1];tmms[2]=mms[2];tmms[3]=0;'''][pred]+'''
    } model {
        real cumo[3];
        cumo[1]=0;cumo[2]=9;cumo[3]=14;
        sy~cauchy(0,20);
        so~cauchy(0,20);
        sm~cauchy(0,20);
        for (n in 1:N){
            mo[n]~multi_normal(mm'''+['','+nam*age[n]'][pred]+',quad_form_diag(rm,sm'+ \
            ['','+tnas*age[n]'][pred]+'''));
        for (m in 1:3){
            o[n,m]~multi_normal(mo[n],quad_form_diag(ro,so'''+['','+tmms*(m-2)'][pred]+'''));
        for (p in 1:9){
            if (! is_nan(y[n,m,p][1]))
                c[n,m,p]~multi_normal(segment(o[n,m],1,2)+o[n,m][3]*y[n,m,p]'''+ \
                ['','+odm*dist[n,m,p]'][pred]+',quad_form_diag(ry,sy'+ \
                ['','+ods*dist[n,m,p]'][pred]+'));}}}}'
    if doCompile:
        sm = pystan.StanModel(model_code=model)                
        with open(DPATH+f'smHADR{pred}.pkl', 'wb') as f: pickle.dump(sm, f)
    with open(DPATH+f'smHADR{pred}.pkl', 'rb') as f: sm=pickle.load(f)
    with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    ds=np.load(DPATH+f'ds{fn}dva{dva}.npy')
    if dev is None: devs=range(2)
    else: devs=[dev]
    for dev in devs:
        for eye in range(3):
            sel=~np.all(np.all(np.isnan(ds[dev,:,:,eye,:,0]),axis=2),axis=1)
            #sel[30:]=False
            age=np.load(DPATH+'age.npy')[sel]
            age= 1-age/360
            dist=ds[dev,sel,:,eye,:,6]/100
            y=ds[dev,sel,:,eye,:,:2]
            c=ds[dev,sel,:,eye,:,2:4]

            dat={'y':y,'N':y.shape[0],'c':c,'age':age,'dist':dist}
            fit = sm.sampling(data=dat,iter=6000,chains=6,thin=10,warmup=3000,n_jobs=6,seed=SEED)
            #print(fit);bla
            with open(DPATH+f'smHADR{pred}{dev}{eye}dva{dva}.stanfit','wb') as f: 
                pickle.dump(fit,f,protocol=-1)

def tableVar(fn,correlation=False,dev=0,plot=True): 
    ''' prints code for latex table to console
        table includes variance estimates '''
     
    left=[]
    from scipy.stats import scoreatpercentile as sap
    v=int(correlation)
    #for e in range(2):#device
    for i in range(3): # eye
        for l in range(3): # level
            if l==0: ii=['L','R','B'][i]
            else: ii=''
            #if l==0: ee=['Tobii','Smi'][e]
            #else: ee=''
            left.append([ii,['location','session','participant'][l]])
    
    lleft=list(np.array(left).T)
    top=['eye','level']
    for d in range(3): # dimension
        temp1=['\\sigma','\\rho'][v]
        temp2=[['x','y','s'],['{xy}','{xs}','{ys}']][v][d]
        top.append(f'${temp1}_{temp2}$')
        lleft.append(['']*len(lleft[0]))
    res=np.array([top]+list(np.array(lleft).T),dtype='U256')
    with open(DPATH+f'sm{fn}.pkl', 'rb') as f: sm=pickle.load(f)
    ax=np.newaxis 
    e=dev#for e in range(2):# device
    resout=np.zeros((res.shape[0]-1,res.shape[1]-2,5))*np.nan
    qntls=[50,2.5,97.5,25,75]
    totvar=[[],[],[]]
    for i in range(3): # eye
        try: 
            with open(DPATH+f'sm{fn}{e}{i}dva0.stanfit','rb') as f: fit=pickle.load(f)
        except: continue
        smr=fit.summary()
        sr=smr['summary'][:-1,-1]
        inds=(sr>1.1).nonzero()[0]
        if len(inds)>0: 
            print(e,i,smr['summary_rownames'][inds], sr[inds])
        else: print(e,i,'CONVERGED') 
        w=fit.extract()
        ods=0;tnas=0
        for l in range(3): # level
            if v==0 and 'ods' in w.keys(): prd=[w['ods']*55/100,0,w['tnas']*(1-30*7/360)][l]
            else: prd=0
            temp=w[['s','r'][v]+['y','o','m'][l]]+prd
            for d in range(3): # dimension
                if l==0 and d==2 or l==0 and d==1 and v==1:continue
                if v==0: tmp=np.square(temp[:,d])
                else:tmp=temp[:,[0,0,1][d],[1,2,2][d]]
                #if v==0 and d==2: tmp*=10# or v==1 and d>0:
                res[1+l+3*i,2+d]='\\textbf{%.2f} (%.2f,%.2f)'%(np.median(tmp),sap(tmp,2.5),sap(tmp,97.5))
                resout[l+3*i,d,:]=list(map(lambda x: sap(tmp,x),qntls))
        if v: continue
        totvar[i]=(np.square(w['sy']).sum(1)+np.square(w['so']).sum(1))
        perc= np.square(w['so']).sum(1)/totvar[i]
        print('eye %d perc. between-session var / tot var = %.3f, (%.3f,%.3f)'%(i,sap(perc,50),sap(perc,2.5),sap(perc,97.5)))
        print(f'tot accc=%.3f, (%.3f,%.3f)'%(sap(np.sqrt(totvar[i]),50),sap(np.sqrt(totvar[i]),2.5),sap(np.sqrt(totvar[i]),97.5)) )
    accc=np.sqrt(totvar[0]/4+totvar[1]/4)
    print(f'tot accc binoc=%.3f, (%.3f,%.3f)'%(sap(accc,50),sap(accc,2.5),sap(accc,97.5)) )
    ndarray2latextable(res,decim=0,hline=[0,3,6,9],nl=3);
    print('')  
    ffn='v'+['Hier','Pred'][int(fn[-1])]+['Tob','Smi'][dev]
    np.save('test',resout)
    figureVar(resout,ffn)
def figureVar(dat,fnout):
    plt.close('all') 
    clrs=['g','r','y']
    handles=[]
    print(fnout)
    figure(size=2,aspect=0.6,dpi=400)
    for a in range(2):
        for e in range(3):
            xs=4*np.arange(3)+e+a*12#-0.25
            #(col-1.5)/1.5*0.2
            x=np.array([xs,xs])
            out=plt.plot(x,dat[e*3:(e+1)*3,a,1:3].T,color=clrs[e])
            #plt.plot(x,dat[e*3:(e+1)*3,a,3:5].T,color=clrs[e],lw=3,solid_capstyle='round')
            plt.plot(xs,dat[e*3:(e+1)*3,a,0],mfc=clrs[e],       
                mec=clrs[e],ms=8,marker='_',mew=2,lw=0) 
            handles.append(out)
    plt.grid(True,axis='y')
    plt.plot([11,11],[0,10],'k',lw=0.5)
    #if not k: plt.xlabel('LC model')
    plt.ylabel('Variance in $\mathrm{cm}^2$')
    plt.ylim([0,5])
    ax=plt.gca()
    ax.set_xticks([1,5,9,13,17,21])
    ax.set_xticklabels(['L','S','P','L','S','P'])
    #plt.legend([handles[0][0],handles[1][0],handles[-1][0]],['left eye','right eye','binocular'],loc=0)
    plt.title('horizontal axis'+30*' '+'vertical axis')
    plt.savefig('../publication/figs/%s.png'%fnout,bbox_inches='tight')  
    

                                         
def tableSlope(fn,dev=0):
    ''' prints code for latex table to console
        table includes estimates of regression coefficients'''  
    left=[]
    #for e in range(2):#device
    for i in range(3): # eye
            ii=['L','R','B'][i]
            #ee=['Tobii','Smi'][e]
            left.append([ii])

    left=list(np.array(left).T)
    top=['eye']
    for i in ['\\mu','\\gamma_A','\\delta_A','\\gamma_D','\\delta_D','\\delta_S']:
        for j in ['x','y','s']:
            if not (len(top)>7 and j=='s'): top.append('$'+i+'_'+j+'$')
    
    for i in range(len(top)-1):
        left.append(['']*len(left[0]))
    #print(len(left),top)
    res=np.array([top]+list(np.array(left).T),dtype='U256').T
    vrs=['mm','nam','nas','odm','ods','mms']
    with open(DPATH+f'sm{fn}.pkl', 'rb') as f: sm=pickle.load(f)
    #for e in range(2):#device
    e=dev
    for i in range(3): # eye
        try:
            with open(DPATH+f'sm{fn}{e}{i}dva0.stanfit','rb') as f: fit=pickle.load(f)
        except: continue
        smr=fit.summary()
        sr=smr['summary'][:-1,-1]
        inds=(sr>1.1).nonzero()[0]
        if len(inds)>0: 
            print(e,i,smr['summary_rownames'][inds], sr[inds])
        else: print(e,i,'CONVERGED')
        w=fit.extract()
        #print(fit)
        g=1
        for v in range(len(vrs)):
            if vrs[v]=='mm': 
                odm=np.concatenate([w['odm'],np.zeros((w['odm'].shape[0],1))],axis=1)
                temp=w['mm']+w['nam']*(1-30*7/360)+odm*55/100
            elif vrs[v][:2]=='na': temp=-6*30*w[vrs[v]]/360
            elif vrs[v][:2]=='od':temp= 10*w[vrs[v]]/100
            else: temp=w[vrs[v]]
            for k in range(temp.shape[1]):
                #print(res.shape,g,1+i+2*e,temp.shape)
                res[g,1+i]='\\%.2f (%.2f,%.2f)'%(np.median(temp[:,k]),
                    sap(temp[:,k],2.5),sap(temp[:,k],97.5))
                g+=1
    ndarray2latextable(res,decim=0,hline=[0,3,6,8,10,12],nl=1);
    print('')  
    
def figureOverview():
    figure(size=3,aspect=0.4,dpi=400)
    pms=[[0,1,0,0,0],[0,1,0,1,0],[0,0,0,0,0],[0,-1,0,0,1],[1,1,0,0,0],[0,1,1,0,0],[0,1,3,0,0]]
    res=[]
    for pm in pms:
        res.append(tablePA(['FixTh1_0','FixTh2Vel20minDur0_1'][pm[0]],m=pm[1],
            dva=pm[2],dev=pm[3], novelLocations=bool(pm[4]),plot=0)[-1,7,:])
    #np.save('res',res)
    #dat=np.load('res.npy',allow_pickle=True)
    ordr=[4,2,3,0,5,6,1]
    dat=np.array(res)[ordr,:]
    lbls=['Tobii 65 cm (Fig. 1)','SMI 65 cm (Fig. 1)','Tobii 45 cm (Fig. S5)','Tobii 55cm (Fig. S6)','inclusive sample (Fig. S8)','deg. units A (Fig. S9)','deg. units B (Fig. S10)']
    xs=np.arange(dat.shape[0])
    x=np.array([xs,xs])
    clr='b';ms=6
    plt.plot(dat[:,1:3].T,x,color=clr,alpha=0.5)
    plt.plot(dat[:,3:5].T,x,color=clr,lw=3,solid_capstyle='round')
    plt.plot(dat[:,0],xs,mfc=clr,mec=clr,marker='d',lw=0) 
    plt.plot(dat[:,5],xs+0.25,mfc=clr,mec=clr,marker='|',lw=0) 
    mm=dat[:,5];se=dat[:,6]
    plt.plot([mm-1.96*se,mm+1.96*se],x+0.25,color=clr,alpha=0.7)
    #plt.xlim([0,dat.shape[0]])
    plt.xlim([0,5])
    plt.grid(False,axis='y')
    plt.grid(True,axis='x')
    #if not k: plt.xlabel('LC model')
    plt.xlabel('Accuracy in degrees')
    ax=plt.gca()
    ax.set_yticks(xs)
    ax.set_yticklabels(np.array(lbls)[ordr])
    #plt.show()
    plt.savefig('../publication/figs/aSupp.png',bbox_inches='tight') 
def figurePreproc():
    figure(size=3,aspect=0.4,dpi=400)
    d=np.load('d.npy',allow_pickle=True)
    raw=np.concatenate(d,0)
    subplot(1,2,1)
    plt.plot((raw[:,1]-raw[0,1])/1000000,raw[:,3])
    plt.plot((raw[:,1]-raw[0,1])/1000000,raw[:,4])
    plt.ylim([-30,50])
    e=np.load('e.npy',allow_pickle=True)
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
    f=np.load('f.npy')
    k=0;h=0
    val=[1,5,7,14,16,19,21]
    for ff in f:
        if e[ff[1],0]-e[ff[0],0]>0.1:
            plt.plot([e[ff[0],0],e[ff[1],0]],[50-1*k,50-1*k],['r','g'][int(h in set(val))]);
            k+=1
        h+=1
    plt.savefig('../publication/figs/preproc.png',bbox_inches='tight')
     
if __name__=='__main__':
    import pickle
    # loading and preprocessing
    fns=checkFiles()             
    D=loadCalibrationData(fns)
    with open(DPATH+'D.out','wb') as f: pickle.dump(D,f)

    with open(DPATH+'D.out','rb') as f: D=pickle.load(f)
    dataPreprocessing(D,'dsFixTh2Vel20minDur0_1dva0',thacc=2,
        thvel=20,dva=0,minDur=0.1)
    # estimate LC parameters (took appr. a week on i7 haswell CPU)
    computePA('FixTh1_0dva0',docompile=False)
    computePA('FixTh1_0dva0',docompile=False,short=True,dev=1)
    computePA('FixTh1_0dva0',docompile=False,short=True,m=-1,dev=0)
    computePA('FixTh2Vel20minDur0_1dva0',docompile=False,short=True,dev=0)
    computePA('FixTh1_0dva2',docompile=False,short=True)
    computePA('FixTh1_0dva1',docompile=False,short=True)
    computePA('FixTh1_0dva4',docompile=False,short=True)
    computePA('FixTh1_0dva3',docompile=False,short=True)
    # estimate variance with three-level model
    computeVar('FixTh1_0',dva=0,dev=0,includePredictors=False,doCompile=False)
    computeVar('FixTh1_0',dva=4,dev=0,includePredictors=True,doCompile=False)
    computeVar('FixTh1_0',dva=0,dev=0,includePredictors=True,doCompile=False)
    computeVar('FixTh1_0',dva=0,dev=1,includePredictors=False,doCompile=False)

    # figures  
    figureSample(f'dsFixTh1_0dva0incl',dev=0)
    figureSample(f'dsFixTh2Vel20minDur0_1dva0incl')
    figureSample(f'dsFixTh1_0dva0incl',dev=1);

    figurePreproc()
    tablePA('FixTh1_0',m=1,dva=0,dev=0,plot=2,legend=True)
    tablePA('FixTh1_0',m=1,dva=0,dev=1,plot=2)
    tablePA('FixTh1_0',m=0,dva=0,dev=0,plot=2)
    tablePA('FixTh1_0',m=-1,dva=0,dev=0,plot=2,novelLocations=True)
    tablePA('FixTh1_0',m=1,dva=0,dev=0,plot=1,pref='aL')
    tablePA('FixTh2Vel20minDur0_1',m=1,dev=0,plot=2,pref='aP')
    tablePA('FixTh1_0',m=1,dva=3,dev=0,plot=2) 
    tablePA('FixTh1_0',m=1,dva=1,dev=0,plot=2) 
    tableVar('HADR0',correlation=False,dev=1)
    tableVar('HADR0',correlation=False,dev=0)
    figureOverview() 
    #tableSlope('HADR1')
    
    




