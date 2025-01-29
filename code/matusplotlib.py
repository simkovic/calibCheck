import numpy as np
import pylab as plt
import matplotlib as mpl
from scipy.stats import scoreatpercentile as sap
from scipy.special import erfinv
from scipy import stats
import pickle,os
from sys import stdout
from PIL import ImageFont, ImageDraw, Image


__all__ = ['getColors','errorbar','pystanErrorbar',
           'printCI','formatAxes','saveStanFit','loadStanFit',
           'figure','subplot','subplotAnnotate',
           'hist','histCI','plotCIttest1','plotCIttest2',
           'ndarray2latextable','ndarray2gif','plotGifGrid',
           'str2img','plothistCI','fit2dict']
CLR=(0.2, 0.5, 0.6)
# size of figure columns
FIGCOL=[3.27,4.86,6.83] # plosone
FIGCOL=[3.3,5, 7.1] # frontiers
FIGCOL=[3.3,5,7.1] # plosone new
FIGCOL=[2.87,4.3,5.74,9]# peerj


SMALL=6
MEDIUM=9
plt.rc('axes', titlesize=MEDIUM)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)    # fontsize of the tick labels

# TODO custom ppl style histogram
def getColors(N):
    ''' creates set of colors for plotting

        >>> len(getColors(5))
        5
        >>> getColors(5)[4]
        (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
        >>> N=14
        >>> c=getColors(N)
        >>> plt.scatter(range(N),range(N),color=c)
        >>> plt.show()
    '''
    clrs=[]
    cm = plt.get_cmap('Paired')
    for i in range(N+1):
        clrs.append(cm(1.*i/float(N))) 
        #plt.plot(i,i,'x',color=clrs[i])
    clrs.pop(-2)
    return clrs

def imshow(*args,**kwargs):
    plt.imshow(*args,**kwargs)

#plt.ion()
#imshow(np.array([[1,2,3],[2,1,2],[3,1,2]]))

def formatAxes(ax):
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(False,axis='x')
    ax.grid(True,axis='y')
    
def figure(*args,**kwargs):
    ''' wrapper around matplotlib.figure
        additionally supports following kwargs
        size - 1,2 or 3 respectively for small, medium, large width
        aspect - [0,inf] height to width ratio 
    '''
    if not 'figsize' in kwargs:
        if 'size' in kwargs: w= FIGCOL[kwargs.pop('size')-1]
        else: w=FIGCOL[0]
        if 'aspect' in kwargs: h=kwargs.pop('aspect')*w
        else: h=w
        kwargs['figsize']=(w,h)
    fig=plt.figure(*args,**kwargs)
    formatAxes(plt.gca())
    return fig

def hist(*args,**kwargs):
    '''
        >>> dat=np.random.randn(1000)*10+5
        >>> x=np.linspace(-30,30,60)
        >>> hist(dat,bins=x)
    '''
    if not 'facecolor' in kwargs: kwargs['facecolor']=CLR
    if not 'edgecolor' in kwargs: kwargs['edgecolor']='w'
    plt.hist(*args,**kwargs)

def histCI(*args,**kwargs):
    '''
        >>> x=np.random.randn(1000)
        >>> bn=np.linspace(-3,3,41)
        >>> histCI(x,bins=bn)
    '''
    if 'plot' in kwargs: plot=kwargs.pop('plot')
    else: plot=True
    if 'alpha' in kwargs: alpha=kwargs.pop('alpha')
    else: alpha=0.05
    a,b=np.histogram(*args,**kwargs)
    m=b.size
    n=args[0].shape[0]
    c=stats.norm.ppf(alpha/(2.*m))/2.*(m/float(n))**0.5
    l=np.square(np.maximum(np.sqrt(a)-c,0))
    u=np.square(np.sqrt(a)+c)
    if plot: plothistCI(a,b,l,u)
    return a,b,l,u
def symhist(x1,x2,bins):
    ''' symmetric histogram of two data sets
        >>> symhist(np.random.randn(100),np.random.randn(100)+1,np.linspace(-3,4,15))
        >>> plt.show()
    '''
    bw=bins[1]-bins[0]
    a1=np.histogram(x1,bins=bins,normed=True)
    plt.barh(bins[:-1],-a1[0],ec='w',fc='y',height=bw,lw=0.1)
    a2=np.histogram(x2,bins=bins,normed=1)
    plt.barh(bins[:-1],a2[0],ec='w',fc='y',height=bw,lw=0.1)
    xmax=max(plt.xlim())
    plt.xlim([-xmax,xmax])
    plt.ylim([bins[0],bins[-1]])
    ax=plt.gca()
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])


def plothistCI(a,b,l,u):
    '''
        >>> x=np.random.randn(1000)
        >>> bn=np.linspace(-3,3,41)
        >>> a,b,l,u=histCI(x,bins=bn)
        >>> plothistCI(a,b,l,u)
    '''
    b=b[:-1]+np.diff(b)/2.
    plt.plot(b,a)
    x=np.concatenate([b,b[::-1]])
    ci=np.concatenate([u,l[::-1]])
    plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                alpha=0.2,fill=True,fc='red',ec='red'))

def kernelreg(x,y,xnew,ciwidth=0.5,Kwidth=1):
    '''
        Nadaray-Watson kernel estimator 
        from Wasserman(2004) chap 20.4
        x - predictor NxK1x...xKd ndarray
        y - outcome, N-length ndarray
        xnew - predictor for prediction MxK1x...xKd ndarray
        returns list of
            estimated outcome for xnew, M-length ndarray
            lower confidence interval
            upper confidence interval
            leave-one-out crossvalidation score
        example:
        >>> x=np.linspace(-10,10,200)
        >>> x+=np.random.randn(x.size)*0.1
        >>> x=np.sort(x)
        >>> y=np.sin(x)+np.random.randn(x.size)
        >>> xnew=np.linspace(-10,10,1000)
        >>> for k in [0.05,0.5,1,2,5,10]:
        >>>     plt.figure()
        >>>     plt.plot(x,y,'.')
        >>>     ynew,lcf,ucf,J=kernelreg(x,y,xnew,Kwidth=k)
        >>>     plt.plot(xnew,ynew)
        >>>     plt.plot(xnew,lcf,'r')
        >>>     plt.plot(xnew,ucf,'r')
        >>>     plt.title('h=%.01f, J=%.01f'%(k,J))
    '''
    K=lambda x: (2*np.pi)**-0.5*np.exp(-np.square(x)/2.)
    xlist=x.tolist();N=len(xlist)
    xnlist=xnew.tolist();M=len(xnlist)
    ynew=np.zeros(M)*np.nan
    w=np.zeros((M,N))
    for i1 in range(M):
        for i2 in range(N):
            d=np.linalg.norm(xnlist[i1]-xlist[i2])
            w[i1,i2]=K((d)/float(Kwidth))
    for i1 in range(M):
        ynew[i1]=w[i1,:].dot(y)/w[i1,:].sum()
    sigma=np.square(np.diff(y))
    sigma= (sigma.sum()/float(2*sigma.size))**0.5
    se=sigma*np.sqrt(np.square(w).sum(1))
    rng=0
    w=np.zeros((N,N))
    r=np.zeros(N)*np.nan
    for i1 in range(N):
        for i2 in range(N):
            d=np.linalg.norm(xnlist[i1]-xlist[i2])
            if d>rng: rng=d
            w[i1,i2]=K((d)/float(Kwidth))
    for i1 in range(N):
        r[i1]=w[i1,:].dot(y)/w[i1,:].sum()
    J=np.square((y-r)/(1-K(0)/w.sum(axis=1))).sum()
    q=erfinv(0.5*(1+ciwidth**(3*Kwidth/rng)))
    return ynew, ynew-q*se, ynew+q*se,J

def subplot(*args):
    ax=plt.subplot(*args)
    formatAxes(ax)
    return ax

def subplotAnnotate(loc='nw',nr=None,clr='k',fs=12,ax=None):
    if type(loc) is list and len(loc)==2: ofs=loc
    elif loc=='nw': ofs=[0.1,0.9]
    elif loc=='sw': ofs=[0.1,0.1]
    elif loc=='se': ofs=[0.9,0.1]
    elif loc=='ne': ofs=[0.9,0.9]
    else: raise ValueError('loc only supports values nw, sw, se and ne')
    if ax is None:ax=plt.gca()
    ax.numRows = ax.get_subplotspec().get_geometry()[0]
    ax.numCols = ax.get_subplotspec().get_geometry()[1] 
    if nr is None:
        nr=ax.get_subplotspec().colspan.start*ax.numRows +ax.get_subplotspec().rowspan.start
    elif np.isnan(nr):nr=ax.get_subplotspec().rowspan.start*ax.numCols+ax.get_subplotspec().colspan.start
    x=ax.get_xlim();y=ax.get_ylim()
    ax.text(x[0]+ofs[0]*(x[1]-x[0]),y[0]+ofs[1]*(y[1]-y[0]), 
            str(chr(65+nr)),horizontalalignment='center',verticalalignment='center',
            fontdict={'weight':'bold'},fontsize=fs,color=clr)

def _errorbar(out,x,clr='k',vertical=True):
    if vertical:
        plt.plot([x,x],out[1:3],color=clr)
        if len(out)>3:plt.plot([x,x],out[3:5],color=clr,lw=3,solid_capstyle='round')
        plt.plot([x],[out[0]],mfc=clr,mec=clr,ms=8,marker='_',mew=2)
    else:
        plt.plot(out[1:3],[x,x],color=clr)
        if len(out)>3:plt.plot(out[3:5],[x,x],color=clr,lw=3,solid_capstyle='round')
        plt.plot([out[0]],[x],mfc=clr,mec=clr,ms=8,marker='|',mew=2)


def _horebar(d,xs,clr):
    ''' code snippet for horizontal errorbar'''
    for i in range(d.shape[1]):
        x=xs[i]
        plt.plot([sap(d[:,i],2.5),sap(d[:,i],97.5)],[x,x],color=clr)
        plt.plot([sap(d[:,i],25),sap(d[:,i],75)],[x,x],
            color=clr,lw=3,solid_capstyle='round')
        plt.plot([np.median(d[:,i])],[x],mfc=clr,mec=clr,ms=8,marker='|',mew=2)
    plt.gca().set_yticks(xs)
    
def plotCIwald(y,x=0,df=None,alpha=0.05,alpha2=0.25,clr=CLR,verticalErrorbar=True):
    if df is None: m=np.nanmean(y,0);df=np.sum(~np.isnan(y),0)
    else:  m=y
    se=np.sqrt(m*(1-m)/df)
    cil=stats.norm.ppf(alpha/2.)*se
    out=[m,m-cil,m+cil]
    if not alpha2 is None:
        cii=stats.norm.ppf(alpha2/2.)*se
        out.extend([m-cii,m+cii])
    out=np.array(out).T
    if out.ndim==1: out=np.array([out])
    for k in range(out.shape[0]):
        _errorbar(out[k,:],x=x+k,clr=clr,vertical=verticalErrorbar)  
    return out

def plotCIttest1(y,x=0,alpha=0.05,clr=CLR,verticalErorbar=True):
    ''' single group t-test'''
    m=y.mean();df=y.size-1
    se=y.std()/y.size**0.5
    cil=stats.t.ppf(alpha/2.,df)*se
    cii=stats.t.ppf(0.25,df)*se
    out=[m,m-cil,m+cil,m-cii,m+cii]
    _errorbar(out,x=x,clr=clr,vertical=verticalErrorbar)
    return out
    
def plotCIttest2(y1,y2,x=0,alpha=0.05,clr='k',verticalErrorbar=True):
    n1=float(y1.size);n2=float(y2.size);
    v1=y1.var();v2=y2.var()
    m=y2.mean()-y1.mean()
    s12=(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))**0.5
    se=s12*(1/n1+1/n2)**0.5
    df= (v1/n1+v2/n2)**2 / ( (v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1)) 
    cil=stats.t.ppf(alpha/2.,df)*se
    cii=stats.t.ppf(0.25,df)*se
    out=[m,m-cil,m+cil,m-cii,m+cii]
    _errorbar(out,x=x,clr=clr,vertical=verticalErrorbar)
    return out
    
def errorbar(y,clr='k',x=None,pi=95,pi2=50,
    labels=None,plot=True,verticalErrorbar=True,linkf=None):
    ''' customized error bars
        y - NxM ndarray containing results of
            N simulations of M random variables
        x - array with M elements, position of the bars on x axis 
        clr - bar color
        labels - array with xtickslabels
        linkf - function that is used to transform parameters

        >>> errorbar(np.random.randn(1000,10)+1.96)    
    '''
    out=[]
    if linkf is None: linkf=lambda x: x
    d=np.array(y);
    if d.ndim<2: d=np.array(y,ndmin=2).T
    d=linkf(d)
    if not x is None: x=np.array(x)
    if x is None or x.size==0: x=np.arange(d.shape[1])
    elif x.size==1: x=np.ones(d.shape[1])*x[0]
    elif x.ndim!=1 or x.shape[0]!=d.shape[1]:
        print('errorbar:dimension mismatch')
        x=np.arange(0,d.shape[1])
    ax=plt.gca()
    aa=(100-pi)/2;
    if not pi2 is None:bb=(100-pi2)/2
    for i in range(d.shape[1]):
        out.append([np.median(d[:,i]),sap(d[:,i],aa),sap(d[:,i],aa+pi)])
        if not pi2 is None:out[-1].extend([sap(d[:,i],bb),sap(d[:,i],bb+pi2)])
        if len(clr)==d.shape[1]:c=clr[i]
        else: c=clr
        if plot: _errorbar(out[-1],x=x[i],clr=c,vertical=verticalErrorbar)
    if plot and verticalErrorbar:
        ax.set_xticks(x)
        if not labels is None: ax.set_xticklabels(labels)
        plt.xlim([np.floor(np.min(x)-1),np.ceil(np.max(x)+1)])
    elif plot and not verticalErrorbar:
        ax.set_yticks(x)
        if not labels is None: ax.set_yticklabels(labels)
        plt.ylim([np.floor(np.min(x)-1),np.ceil(np.max(x)+1)])
    return np.array(out)

def pystanErrorbar(w,keys=None,linkf=None):
    """ plots errorbars for variables in fit
        fit - dictionary with data extracted from Pystan.StanFit instance
        linkf - function that is used to transform parameters
    """
    if linkf is None: linkf=lambda x: x
    kk=0
    ss=[];sls=[]
    if keys is None: 
        keys=list(w.keys())
        keys.remove('lp__')
    for k in keys:
        d= w[k]
        if d.ndim==1:
            ss.append(d);sls.append(k)
            continue
            #d=np.array(d,ndmin=2).T
        d=np.atleast_3d(d)
        for h in range(d.shape[2]):
            kk+=1; figure(num=kk)
            #ppl.boxplot(plt.gca(),d[:,:,h],sym='')
            errorbar(d[:,:,h],linkf=linkf)
            plt.title(k)
    #ss=np.array(ss)
    for i in range(len(ss)):
        print(sls[i], ss[i].mean(), 'CI [%.3f,%.3f]'%(sap(ss[i],2.5),sap(ss[i],97.5)))
def printCI(w,var=None,decimals=3):
    sfmt=' {:.{:d}f} [{:.{:d}f},{:.{:d}f}]'
    def _print(b):
        d=np.round([np.median(b), sap(b,2.5),sap(b,97.5)],decimals).tolist()
        print(sfmt.format(d[0],decimals,d[1],decimals,d[2],decimals))
        #print var+' %.3f, CI %.3f, %.3f'%tuple(d) 
    if var is None: d=w;var='var'
    else: d=w[var]
    if d.ndim==2:
        for i in range(d.shape[1]):
            _print(d[:,i])
    elif d.ndim==1: _print(d)

def pearsonrCI(x,y,alpha=0.05):
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r,lo, hi,p
    
def printRhat(w):
    from arviz import summary
    print('checking convergence')
    azsm=summary(w)
    nms=azsm.axes[0].to_numpy()
    rhat = azsm.to_numpy()[:,-1]
    srt=np.argsort(rhat)
    nms=nms[srt]
    ess=np.zeros((srt.size,2)) 
    ess[:,0]=azsm.to_numpy()[srt,-3]
    ess[:,1]=azsm.to_numpy()[srt,-2]
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
    return i.size>0,nms,rhat,ess

def saveStanFit(fit,dat,fname,model=None,computeRhat=True): 
    if computeRhat:
        converged,nms,rhat,ess=printRhat(fit)
        w={'nms++':nms,'rhat++':rhat,'ess++':ess}
    else: w={}
    for k in fit.keys():w[k]=np.rollaxis(fit[k],-1,0)
    #assert(not np.all(np.isnan(w['y'][0,:,:,:])))
    for k in dat.keys():w[k+'+']=dat[k]
    #w['model']=fit.get_stancode()
    if not model==None:w['model_code++']=model
    w['num_chains++']=fit.num_chains
    with open(fname+'.wfit','wb') as f: pickle.dump(w,f,protocol=-1)
    return w 
def loadStanFit(fname,excludeChains=[],computeRhat=True):
    with open(fname+'.wfit','rb') as f: w=pickle.load(f)
    if len(excludeChains)==0: return w  
    wnew={}
    chinds=list(range(w['num_chains++']))
    for ch in excludeChains: chinds.remove(ch)
    #print(w.keys())
    for k in w.keys():
        if k[-1]=='+': continue 
        wnew[k]=np.reshape(w[k],[w['num_chains++'],w[k].shape[0]//w['num_chains++']]+ list(w[k].shape[1:]),order='F')[chinds,]
    if computeRhat: converged,nms,rhat,ess=printRhat(wnew)
    for k in wnew.keys():
        wnew[k]=np.reshape(wnew[k],[wnew[k].shape[0]*wnew[k].shape[1]]+ list(wnew[k].shape[2:]),order='F')
    for k in w.keys():
        if k[-1]=='+':wnew[k]=w[k]  
    if computeRhat: wnew['nms++']=nms;wnew['rhat++']=rhat;wnew['ess++']=ess
    #with open(fname+'.wfit','wb') as f: pickle.dump(w,f,protocol=-1)
    return wnew
    
    

def ndarray2latextable(array,decim=2,hline=[0],vline=None,nl=1):
    ''' array - 2D numpy.ndarray with shape (rows,columns)
                or 3D numpy.array with shape (rows,columns,STAN samples)
        decim - decimal precision of float, use 0 for ints
            should be scalar or a list with len(decim)=nr cols
    '''
    ecol=' \\\\\n';shp=array.shape
    out='\\begin{table}\n\\centering\n\\begin{tabular}{|'
    if vline is None: vline=range(shp[1])
    for i in range(shp[1]):
        out+=['c','l'][int(i<nl)]+['','|'][int(i in set(vline))]
    out=out+'|}\n\\hline\n'
    for i in range(shp[0]):
        for j in range(shp[1]):
            if type(decim) is list: dc=decim[j]
            else: dc=decim
            if type(array[i,j])==np.str_ or type(array[i,j])==str: out+='%s'%array[i,j]
            #elif dc==0: out+='%d'%int(array[i,j])
            else:
                if array.ndim==3:
                    flt='{: .%df} [{: .%df}, {: .%df}]'%(dc,dc,dc)
                    out+=flt.format(np.median(array[i,j,:]),
                        stats.scoreatpercentile(array[i,j,:],2.5),
                        stats.scoreatpercentile(array[i,j,:],97.5))
                else:
                    flt='{: .%df}'%dc
                    out+=flt.format(np.round(array[i,j],dc))
            if j<shp[1]-1: out+=' & '
        out+=ecol
        if hline.count(i)==1: out+='\\hline\n'
    out+='\\hline\n\\end{tabular}\n\\end{table}'
    print(out)


def ndarray2gif(path,array,duration=0.1,addblank=False,
                plottime=False,snapshot=1,UC=255,LC=20):
    '''
    path - file path, including filename and suffix,
        supports .gif, .avi (lossless h264 codec)
    array - 3d numpy array, zeroth dim is the time axis,
            dtype uint8 or float in [0,1]
    duration - frame duration
    snapshot - 0=first frame, 1=midframe, 2=last frame

    Example:
    
    >>> im = np.zeros((200,200), dtype=np.uint8)
    >>> im[10:30,:] = 100
    >>> im[:,80:120] = 255
    >>> im[-50:-40,:] = 50
    >>> images = [im*1.0, im*0.8, im*0.6, im*0.4, im*0]
    >>> images = [im, np.rot90(im,1), np.rot90(im,2), np.rot90(im,3), im*0]
    >>> ndarray2gif('test',np.array(images), duration=0.5) 
    '''
    path,suf=path.rsplit('.')
    if array.dtype.type is np.float64 or array.dtype.type is np.float32:
        array=np.uint8(255*array)
    if plottime:
        shp=list(array.shape); shp[1]+=50;
        T=np.ones(shp,dtype=np.uint8)*UC;T[:,:-50,:]=array
        for f in [-1,0,1]:
            T[:,-40:-10,shp[2]/10+f]=LC
            T[:,-25+f,shp[2]/10:9*shp[2]/10]=LC
            T[:,-40:-10,9*shp[2]/10+f]=LC
            for ff in [3,5,7]: T[:,-35:-15,ff*shp[2]/10+f]=LC
            for t in range(T.shape[0]):
                T[t,-40:-10,int((1+float(8*t)/T.shape[0])*shp[2]/10.)+f]=LC
        array=T
    if addblank:
        temp=np.zeros((array.shape[0]+1,array.shape[1],array.shape[2]),dtype=np.uint8)
        temp[1:,:,:]=array
        array=tem
    if array.shape[1]%2==1: array=array[:,:-1,:]
    if array.shape[2]%2==1: array=array[:,:,:-1]
    for k in range(array.shape[0]):
        I=Image.fromarray(array[k,:,:])
        I.save('temp%04d.png'%k)   
    if suf=='gif': os.system('convert -delay %f temp*.png %s.gif'%(duration,path))
    elif suf=='avi':
        cmd='avconv -r '+str(int(1/duration))+ ' -i temp%04d.png -c:v h264 '+path+'.avi'
        print(cmd)
        os.system(cmd)
    else: raise ValueError
    shp=array.shape[0]
    os.system('cp temp%04d.png %s.png'%([0,shp/2,shp-1][snapshot],path))
    for k in range(array.shape[0]):
        os.system('rm temp%04d.png'%k)

def str2img(inp,size,fontpath="/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"):
    image= Image.fromarray(np.zeros((size,1+int(np.round(size*2/3.*len(inp))))))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontpath,size)
    draw.text((0,0), inp, font=font)
    return np.asarray(image)

def plotGifGrid(dat,fn='test',bcgclr=0,forclr=None,text=[],duration=0.1,
                plottime=False,snapshot=False,F=68,P=64,tpG=False,tpL=False):
    '''
        text = [[text1,textsize1,posx1,posy1],[text2, ...
    '''
    if forclr==None: forclr=1-bcgclr
    if forclr==bcgclr: forclr=1
    offset=8 # nr pixels for border padding
    if tpG: cols=len(dat); rows=len(dat[0])
    else: rows=len(dat); cols=len(dat[0])
    h=P;w=P;t=F
    R=bcgclr*np.ones((t+1,(h+offset)*rows,(w+offset)*cols),
                     dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            i=((offset+h)*row+offset/2,(offset+w)*col+offset/2)
            if tpG: temp=dat[col][row]
            else: temp=dat[row][col]
            #temp=np.rollaxis(dat[row][col].T,2)
            for tt in range(F):
                if temp.shape[0]==F: tempp=temp[tt,:,:]
                elif temp.shape[1]==F: tempp=temp[:,tt,:]
                elif temp.shape[2]==F: tempp=temp[:,:,tt]
                else: raise ValueError
                if tpL: tempp=tempp.T
                R[1+tt,i[0]:i[0]+h,i[1]:i[1]+w]=tempp
                
    mnxy=[0,0]
    for t in text:
        t.append(str2img(t[0],t[1]))
        shp=t[-1].shape
        if mnxy[0]>t[2]-shp[0]/2-offset:mnxy[0]=t[2]-shp[0]/2-offset
        if mnxy[1]>t[3]-shp[1]/2-offset:mnxy[1]=t[3]-shp[1]/2-offset
    if np.any(np.array(mnxy)<0):
        shp=list(R.shape); shp[1]-=mnxy[0];shp[2]-=mnxy[1]
        T=np.ones(shp,dtype=np.float32)*bcgclr
        T[:,-mnxy[0]:,-mnxy[1]:]=np.copy(R)
        R=T
    for t in text:
        px,py=t[-1].nonzero()
        R[:,px+t[2]+t[-1].shape[0]/2+offset,
          py+t[3]+t[-1].shape[1]/2+offset]=forclr
    ndarray2gif(fn,np.uint8(R*255),duration=duration,
                plottime=plottime,snapshot=snapshot,
                UC=np.uint8(bcgclr*255),LC=np.uint8(255-bcgclr*255))
def printProgress(iteration, total, time,prefix='', decimals=1, bar_length=30):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = 'O' * filled_length + '-' * (bar_length - filled_length)
    if iteration>0:
        suffix=" ETA "+str_format.format(time*(total/iteration-1)/60.)+ 'mins'
    else: suffix=''
    stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))
    if iteration == total:
        stdout.write('\n')
    stdout.flush()  


def plotMarkovChain(S,R,th=.2):
    # S - list of states, each state is a tuple 
    #    with x,y pos, state name and (optional) orientation 
    #    of the self-loop in degrees of angle, 0 deg = point to right
    # R - matrix with transition probabilities 
    # th - each edge weight greater than th will be drawn
    # example: 
    #>>> plotMarkovChain([[0,0,'A',180],[0,2,'B']],[[1,0],[0.75,0.25]])
    R=np.array(R)
    assert(len(S)==R.shape[0])
    ax=plt.gca()
    r=0.5
    g=.2
    slw=60 #self-loop start point to end point angle
    for i in range(len(S)):
        ax.add_patch(plt.Circle(S[i][:2],radius=r,fill=False,lw=2.5))
        plt.text(S[i][0],S[i][1],S[i][2],ha='center',va='center',size=20)
        for j in range(len(S)):
            if R[i,j]<th:continue
            if i!=j:
                dx=S[j][0]-S[i][0];dy=S[j][1]-S[i][1]
                ty=np.sin(np.arctan2(dy,dx))*r
                tx=np.cos(np.arctan2(dy,dx))*r
                s=(S[i][0]+tx,S[i][1]+ty);e=(S[i][0]+dx-tx,S[i][1]+dy-ty)
                tx=(3*s[0]+2*e[0])/5;ty=(3*s[1]+2*e[1])/5
                tphi=np.rad2deg(np.arctan2(dy,dx))
                if R[i,j]>th and R[j,i]>th: 
                    m=.5
                    
                    tx+=np.cos(np.arctan2(dy,dx)-np.pi/2)*.2
                    ty+=np.sin(np.arctan2(dy,dx)-np.pi/2)*.2
                else: m=0

            else:
                if len(S[i])>3: tht=S[i][3]
                else: tht=0
                s=(S[i][0]+np.cos(np.radians(tht-slw/2))*r,
                    S[i][1]+np.sin(np.radians(tht-slw/2))*r)
                e=(S[i][0]+np.cos(np.radians(tht+slw/2))*r,
                    S[i][1]+np.sin(np.radians(tht+slw/2))*r)
                tx=(s[0]+e[0])/2;ty=(s[1]+e[1])/2
                tx+=np.cos(np.deg2rad(tht))*.5
                ty+=np.sin(np.deg2rad(tht))*.5
                tphi=180
                m=2
            ax.add_patch(mpl.patches.FancyArrowPatch(s,e,color='k',
                connectionstyle='arc3,rad='+str(m),lw=1,
                arrowstyle='simple,head_width=15,head_length=15'))
            plt.text(tx,ty,'%.2f'%np.round(R[i,j],2),
                bbox={'alpha':.8,'fc':'w','lw':0},
                backgroundcolor='w',ha='center',va='center',size=8,
                rotation=(tphi+90)%180-90)
    xd=np.float32(list(map(lambda x: x[0],S)))
    plt.xlim([np.min(xd)-1,1+np.max(xd)])
    yd=np.float32(list(map(lambda x: x[1],S)))
    plt.ylim([np.min(yd)-1,1+np.max(yd)])
    ax.set_aspect(1)
    plt.axis(False);
    
    
def list2d2latextable(lst,decim=2,header=None,colheader=None):
    ''' array - 2D numpy.ndarray with shape (rows,columns)
        decim - decimal precision of float, use 0 for ints
            should be int scalar or a list of list,len(decim)=nr cols
    '''
    assert(np.all(np.array(list(map(len,lst)))==len(lst[0])))
    ecol=' \\\\\n';shp=(len(lst),len(lst[0]))
    out='\\begin{table}\n\\centering\n\\begin{tabular}{|'+ \
        ['l|',''][int(colheader is None)]+shp[1]*'r|'+'}\n\\hline\n'
    if not header is None:
        s=len(header)*'{} & '
        out+=s.format(*header)[:-2]+ecol+'\\hline\n'   
    for i in range(shp[0]):
        if not colheader is None:
            out+='%s & '%colheader[i]
        for j in range(shp[1]):
            if type(decim) is list: dc=decim[j]
            else: dc=decim
            if type(lst[i][j])==str:
                out+='%s'%lst[i][j]
            elif dc==0: out+='%d'%int(lst[i][j])
            elif np.isnan(lst[i][j]): out+=' '
            else:
                flt='{: .%df}'%dc
                out+=flt.format(np.round(lst[i][j],dc))
            if j<shp[1]-1: out+=' & '
        out+=ecol
    out+='\\hline\n\\end{tabular}\n\\end{table}'
    print(out)
 
def mscorrcoef(x,y,N=0):
    sel=np.logical_and(~np.isnan(x),~np.isnan(y))
    r=np.corrcoef(x[sel],y[sel])[0,1]
    z=np.log((1+r)/(1-r))/2
    l=z-(1.96/(sel.sum()-3)**0.5)
    u=z+(1.96/(sel.sum()-3)**0.5)
    l=(np.exp(2*l)-1)/(np.exp(2*l)+1)
    u=(np.exp(2*u)-1)/(np.exp(2*u)+1)
    if N==0: return r,l,u,sel.sum()
    res=np.zeros(N)
    for i in range(N):
        res[i]= np.corrcoef(np.random.permutation(x[sel]),y[sel])[0,1] 
    return r,l,u,sel.sum(),np.median(res)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
