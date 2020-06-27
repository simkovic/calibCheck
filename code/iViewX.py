from ctypes import *
from sys import exit
from PIL import Image
import numpy as np
iViewXAPI = windll.LoadLibrary("iViewXAPI64.dll")
__all__=['connect','disconnect','getSample','getCurrentTimestamp',
    'getTrackingStatus','getTrackingMonitor','getEyeImage','getGeometry']
#structs

class CSystem(Structure):
	_fields_ = [("samplerate", c_int),
	("iV_MajorVersion", c_int),
	("iV_MinorVersion", c_int),
	("iV_Buildnumber", c_int),
	("API_MajorVersion", c_int),
	("API_MinorVersion", c_int),
	("API_Buildnumber", c_int),
	("iV_ETDevice", c_int)]

class CCalibration(Structure):
	_fields_ = [("method", c_int),
	("visualization", c_int),
	("displayDevice", c_int),
	("speed", c_int),
	("autoAccept", c_int),
	("foregroundBrightness", c_int),
	("backgroundBrightness", c_int),
	("targetShape", c_int),
	("targetSize", c_int),
	("targetFilename", c_char * 256)]

class CEye(Structure):
	_fields_ = [("gazeX", c_double),
	("gazeY", c_double),
	("diam", c_double),
	("eyePositionX", c_double),
	("eyePositionY", c_double),
	("eyePositionZ", c_double)]

class CSample(Structure):
	_fields_ = [("timestamp", c_longlong),
	("leftEye", CEye),
	("rightEye", CEye),
	("planeNumber", c_int)]

class CEvent(Structure):
	_fields_ = [("eventType", c_char),
	("eye", c_char),
	("startTime", c_longlong),
	("endTime", c_longlong),
	("duration", c_longlong),
	("positionX", c_double),
	("positionY", c_double)]

class CAccuracy(Structure):
	_fields_ = [("deviationLX",c_double),
				("deviationLY",c_double),				
				("deviationRX",c_double),
				("deviationRY",c_double)]

class CGazeChannelQuality(Structure):
	_fields_ = [("gazeChannelQualityBinocular", c_double),
	("gazeChannelQualityLeft", c_double),
	("gazeChannelQualityRight", c_double)]	
	
class CEyePosition(Structure):
    _fields_=[("validity",c_int),
    ("relativePositionX",c_double),
    ("relativePositionY",c_double),
    ("relativePositionZ",c_double),
    ("positionRatingX",c_double),
    ("positionRatingY",c_double),
    ("positionRatingZ",c_double)]
class CTrackingStatus(Structure):
    _fields_=[("timestamp",c_longlong),
        ("leftEye",CEyePosition),
        ("rightEye",CEyePosition),
        ("total",CEyePosition)]
        
class CImage(Structure):
    _fields_ = [("imageHeight", c_int),
    ("imageWidth", c_int),
    ("imageSize", c_int),
    ("imageBuffer", c_void_p)]
    
class CREDGeometry(Structure):
    _fields_=[("redGeometry",c_int),
    ("monitorSize",c_int),
    ("setupName",c_char * 256),
    ("stimX",c_int),
    ("stimY",c_int),
    ("stimHeightOverFloor",c_int),
    ("redHeightOverFloor",c_int),
    ("redStimDist",c_int),
    ("redInclAngle",c_int),
    ("redStimDistHeight",c_int),
    ("redStimDistDepth",c_int)]

# variables
ett=c_longlong(0)
sd=CSample(0,CEye(0,0,0),CEye(0,0,0),0)
ts=CTrackingStatus(0,CEyePosition(0,0,0,0,0,0,0), 
        CEyePosition(0,0,0,0,0,0,0),CEyePosition(0,0,0,0,0,0,0))
buf=cast(create_string_buffer(300*240*24),c_void_p)
imgTM=CImage(0,0,0,buf)
buf2=cast(create_string_buffer(700*240*8),c_void_p)
imgEI=CImage(0,0,0,buf2)
gcq=CGazeChannelQuality(0,0,0)
systemData = CSystem(0, 0, 0, 0, 0, 0, 0, 0)
redGeometry=CREDGeometry(0,0,b'',0,0,0,0,0,0,0,0)

# function wrappers

def connect(ipsend='127.0.0.1',portsend=4444,iprec='127.0.0.1',portrec=5555,
    refreshRate=60,trackingMode=2):
    '''ipsend,iprec - string, defines IP adress
        portsend,portrec - integert, port id
    '''
    res=iViewXAPI.iV_Start(2)# start ivng server
    res=iViewXAPI.iV_Connect(c_char_p(ipsend.encode()),c_int(portsend), 
            c_char_p(iprec.encode()), c_int(portrec))
    iViewXAPI.iV_SetTrackingMode(trackingMode)
    iViewXAPI.iV_SetSpeedMode(refreshRate)
    if res == 104:print("Eye Tracker not available")
    elif res == 105:print("Port not available")
    elif res == 123:print("Port blocked by another process")
    elif res == 201:print("Iview server not available")
    elif res==112: print ("Invalid parameter")
    elif res==100: print("Could not connect")
    elif res==1: pass
    else: print("IviewX.connect: Unknown return code %d"%res)
    if res!=1: exit()

def getSystemInfo():
    '''returns tuple with
            sampling rate as integer in Hz
            viewx version as string
            api version as string
       returns None if no data available
    '''
    devs=['None','RED','REDm','HiSpeed','MRI','HED','','Custom','REDn']
    res=iViewXAPI.iV_GetSystemInfo(byref(systemData))
    if res==1:
        sampleRate=np.int32(systemData.samplerate)
        iviewxv=(str(systemData.iV_MajorVersion) + "." + 
            str(systemData.iV_MinorVersion) + "." + 
            str(systemData.iV_Buildnumber))
        apiv=(str(systemData.API_MajorVersion) + "." + 
            str(systemData.API_MinorVersion) + "." + 
            str(systemData.API_Buildnumber))
        temp=c_longlong(0);res=iViewXAPI.iV_GetTrackingMode(byref(temp))
        if res==1: 
            tm=['Smart binocular','Monocular left','Monocular right',
                'Binocular','Smart monocular'][temp.value]
        else: tm='Unknown'
        return sampleRate,iviewxv,apiv, 'SMI '+devs[systemData.iV_ETDevice],tm
    elif res==2: return None
    else: print("IviewX.getSystemInfo: Unknown return code %d"%res)
    
       
def getGeometry():
    res=iViewXAPI.iV_GetCurrentREDGeometry(byref(redGeometry))
    if res==1: 
        out=dict( (field,getattr(redGeometry,field)) for field, _ in redGeometry._fields_)
        out['setupName']=out['setupName'].decode()
        if out['redGeometry']==1: out.pop('monitorSize')
        return out
    elif res==101: print("IviewX.getGeometry: Eyetracker not connected")
    else: print("IviewX.getGeometry: Unknown return code %d"%res)
    

def getCurrentTimestamp():
    '''returns current time from the eye tracker as numpy.int64 '''
    res=iViewXAPI.iV_GetCurrentTimestamp(byref(ett))
    if res==1: return np.int64(ett)
    elif res==2: return None
    elif res==101: print("IviewX.getCurrentTimestamp: Eyetracker not connected")
    else: print("IviewX.getCurrentTimestamp: Unknown return code %d"%res)


def getSample():
    '''return current data sample from the eyetracker as a list with
       0: timestamp, 1-2: left eye gaze x,y 3-4: right eye gaze x,y 
       5-6: right and left eye pupil diameter
       7-10: left eye position x,y,z 11-14 right eye position x,y,z
       returns None if no data available
    '''
    res=iViewXAPI.iV_GetSample(byref(sd))
    if res==1: 
        out=[np.int64(sd.timestamp),sd.leftEye.gazeX,sd.leftEye.gazeY,
        sd.rightEye.gazeX,sd.rightEye.gazeY,
        sd.leftEye.diam,sd.rightEye.diam]
        for e in [sd.leftEye,sd.rightEye]:
            out.extend([e.eyePositionX,e.eyePositionY,e.eyePositionZ])
        return out
    elif res==2: return None
    elif res==101: print("IviewX.getSample: Eyetracker not connected")
    else: print("IviewX.getSample: Unknown return code %d"%res)
def getTrackingStatus():
    '''return current tracking status from the eyetracker as a list with
       0: timestamp, 1-3: left eye relative position x,y,z 
       5-7: right eye relative position x,y,z
       4,8: validity code for left eye and right eye resp.
       returns None if no data available
    '''
    res=iViewXAPI.iV_GetTrackingStatus(byref(ts))
    if res==1: 
        out=[np.int64(ts.timestamp)]
        for e in [ts.leftEye,ts.rightEye]:
            out.extend([e.relativePositionX,e.relativePositionY,
                e.relativePositionZ,e.validity])
        return out
    elif res==2: return None
    elif res==101: print("IviewX.getSample: Eyetracker not connected")
    else: print("IviewX.getSample: Unknown return code %d"%res)
    
    
def getTrackingMonitor():
    '''return current tracking monitor image from the eyetracker 
        as a numpy.ndarray with dimensions (height,width,3)
       returns None if no data available
    '''
    res=iViewXAPI.iV_GetTrackingMonitor(byref(imgTM))
    if res==1: 
        h=imgTM.imageHeight;w=imgTM.imageWidth
        if h==0 or w==0: return None
        bf= (c_char*(h*w*24)).from_address(imgTM.imageBuffer)
        out=Image.frombuffer('RGB',(w,h),bf,'raw','RGB',0,1)
        return np.fliplr(np.asarray(out))
    elif res==2: return None
    elif res==101: print("IviewX.getTrackingMonitor: Eyetracker not connected")
    else: print("IviewX.getTrackingMonitor: Unknown return code %d"%res)
    
def getEyeImage():
    '''return current eye image from the eyetracker 
        as a numpy.ndarray with dimensions (height,width)
       returns None if no data available
    '''
    res=iViewXAPI.iV_GetEyeImage(byref(imgEI))
    if res==1: 
        h=imgEI.imageHeight;w=imgEI.imageWidth
        if h==0 or w==0: return None
        bf= (c_char*(h*w*8)).from_address(imgEI.imageBuffer)
        out=Image.frombuffer('L',(w,h),bf,'raw','L',0,1)
        return np.fliplr(np.asarray(out))
    elif res==2: return None
    elif res==101: print("IviewX.getEyeImage: Eyetracker not connected")
    else: print("IviewX.getEyeImage: Unknown return code %d"%res)

def disconnect():
    res=iViewXAPI.iV_Disconnect()
    if res==124: print("IviewX.disconnect: failed to delete socket")
    iViewXAPI.iV_Quit()