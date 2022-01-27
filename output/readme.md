# Data files
## vpinfo.res 
csv file that contains metainformation about participants, columns are: 
* 1 participant id
* 2 cohort (4, 7 or 10 month-old)
* 5 age in days
* 6 id of the person in charge of experiment (look up the author names and the names in the acknowledgments section to see who messed up)
* the remaining columns are irrelevant for this study
## log files in anonPublish
each csv file contains eye-tracking data. Filename consists of participant id, cohort and tobii or smi eye-tracking device indicator. The columns are: 
* 1 pc time (in seconds)
* 2 eye tracker time in microsecs
* 4 left eye horizontal gaze coordinate in degrees of visual angle assuming constant screen-to-eye distance of 70cm, origin at screen center,  positive values go in top-right direction from participants POV, asusMG279 in ExperimentManager.py includes details about the monitor geometry
* 5 left eye vertical
* 6 right eye horizontal
* 7 right eye vertical
* 8-9 left and right eye pupil diameter, units as defined by the SMI API
* 10-12 left eye position in mm; horizontal, vertical and depth respectively origin at eye tracker center; etxyz2roomxyz in SMIthread.py provides details about the computation
* 13-16 right eye position in mm 
