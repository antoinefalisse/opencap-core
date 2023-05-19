import sys
sys.path.append("./..")
import scipy
import numpy as np
import os


from videoIDs import getData
from utils import downloadVideosFromServer, getDataDirectory
from main import main

from syncMarkerDataGetMPJEs_f import main_sync
from gatherMarkerDataPerSubject_f import main_gather
from runOsimPipeline_f import runOpenSimPipeline

from scipy.spatial.transform import Rotation as R

# %% Messy helper function
def get_subject(subjects_to_process):
    sessionDetails = {        
        'subject2': {
            'Session20210813_0001': {},
            'Session20210813_0002': {}},        
        'subject3': {
            'Session20210816_0001': {},
            'Session20210816_0002': {}},        
        'subject4': {
            'Session20210819_0001': {},
            'Session20210819_0002': {}},        
        'subject5': {
            'Session20210820_0001': {},
            'Session20210820_0002': {}},        
        'subject6': {
            'Session20210823_0001': {},
            'Session20210823_0002': {}},        
        'subject7': {
            'Session20210825_0001': {},
            'Session20210825_0002': {}},        
        'subject8': {
            'Session20210827_0001': {},
            'Session20210827_0002': {}},        
        'subject9': {
            'Session20210830_0001': {},
            'Session20210830_0002': {}},        
        'subject10': {
            'Session20210903_0001': {},
            'Session20210903_0002': {}},        
        'subject11': {
            'Session20210910_0001': {},
            'Session20210910_0002': {}}}
    sessionDetails_out = {}
    sessionNames_out = []
    for subject in subjects_to_process:
        sessionDetails_out[subject] = {}
        for session in sessionDetails[subject]:
            sessionDetails_out[subject][session] = {}   
            sessionNames_out.append(session)
    
    return sessionNames_out, sessionDetails_out    

# %% Validation: please keep here, hack to get all trials at once.
subjects_to_process = ['subject' + str(i) for i in range(2,12)]
sessionNames, sessionDetails = get_subject(subjects_to_process)

videoToMarkers = False
syncMocapVideo = True
gatherData = False
runOpenSim = False

# sessionNames = ['Session20210813_0001', 'Session20210813_0002']
# sessionDetails = {    
#     'subject2': {
#         'Session20210813_0001': {},
#         'Session20210813_0002': {}}}

# cameraSetups = ['5-cameras']
cameraSetups = ['3-cameras', '5-cameras']
# augmenter_models = ['v0.1','v0.56']

poseDetectors = ['mmpose']
augmenter_models = ['v0.1','v0.2']

# poseDetectors = ['mmpose']
# augmenter_models = ['v0.1','v0.2','v0.45','v0.54','v0.55','v0.56']

dataDir = getDataDirectory()

# %% Functions for processing the data.
def process_trial(trial_id, trial_name=None, session_name='', isDocker=False,
                  session_id=None, cam2Use=['all'],
                  intrinsicsFinalFolder='Deployed', extrinsicsTrial=False,
                  alternateExtrinsics=None, calibrationOptions=None,
                  markerDataFolderNameSuffix=None,
                  imageUpsampleFactor=4, poseDetector='OpenPose',
                  resolutionPoseDetection='default', scaleModel=False, bbox_thr=0.8, 
                  augmenter_model='v0.7', genericFolderNames=False, offset=False,
                  benchmark=True, dataDir=None):
    
    # Download videos
    trial_name = downloadVideosFromServer(session_id, trial_id, isDocker=True, 
                                          trial_name=trial_name,
                                          session_name=session_name,
                                          benchmark=augmenter_model)

    # Run main.
    main(session_name, trial_name, trial_name, camerasToUse=cam2Use, intrinsicsFinalFolder=intrinsicsFinalFolder, 
          isDocker=isDocker, extrinsicsTrial=extrinsicsTrial, alternateExtrinsics=alternateExtrinsics, 
          calibrationOptions=calibrationOptions, markerDataFolderNameSuffix=markerDataFolderNameSuffix,
          imageUpsampleFactor=imageUpsampleFactor, poseDetector=poseDetector,
          resolutionPoseDetection=resolutionPoseDetection, scaleModel=scaleModel, 
          bbox_thr=bbox_thr, augmenter_model=augmenter_model, genericFolderNames=genericFolderNames,
          offset=offset, benchmark=benchmark,dataDir=dataDir)
    
    return

# %% Process trials.
if not 'cameraSetups' in locals():
    cameraSetups = ['all-cameras']
for count, sessionName in enumerate(sessionNames):
    print('Processing session: ' + sessionName)
    for poseDetector in poseDetectors:
        for cameraSetup in cameraSetups:
            for augmenter_model in augmenter_models:

                if augmenter_model == 'v0.12' or augmenter_model == 'v0.13' or augmenter_model == 'v0.14': # TODO keep adding to this list
                    withTrackingMarkers = False
                else:
                    withTrackingMarkers = True

                # subjectName = subjectNames[count]
                data = getData(sessionName)
                if 'camera_setup' in data:
                    cam2Use = data['camera_setup'][cameraSetup]
                else:
                    cam2Use = ['all']                
                for trial in data['trials']:
                    
                    name = None # default
                    if "name" in data['trials'][trial]:
                        name = data['trials'][trial]["name"]
                        
                    intrinsicsFinalFolder = 'Deployed' # default
                    if "intrinsicsFinalFolder" in data['trials'][trial]:
                        intrinsicsFinalFolder = data['trials'][trial]["intrinsicsFinalFolder"]
                        
                    extrinsicsTrial = False # default
                    if "extrinsicsTrial" in data['trials'][trial]:
                        extrinsicsTrial = data['trials'][trial]["extrinsicsTrial"]
                    
                    alternateExtrinsics = None # default, otherwise list of camera names
                    if "alternateExtrinsics" in data['trials'][trial]:
                        alternateExtrinsics = data['trials'][trial]['alternateExtrinsics']        
                        
                    imageUpsampleFactor = 4 # default
                    if "imageUpsampleFactor" in data['trials'][trial]:
                        imageUpsampleFactor = data['trials'][trial]['imageUpsampleFactor']
                        
                    # resolutionPoseDetection = '1x736' # default_OpenCap
                    resolutionPoseDetection = 'default' # default
                    # resolutionPoseDetection = '1x1008_4scales'
                    if "resolutionPoseDetection" in data['trials'][trial]:
                        resolutionPoseDetection = data['trials'][trial]['resolutionPoseDetection']
                        
                    scaleModel = False # default
                    if "scaleModel" in data['trials'][trial]:
                        scaleModel = data['trials'][trial]['scaleModel']
                        
                    bbox_thr = 0.8 # default
                    if "bbox_thr" in data['trials'][trial]:
                        bbox_thr = data['trials'][trial]['bbox_thr']
                        
                    if videoToMarkers:
                        try:
                            process_trial(data['trials'][trial]["id"], name, session_name=sessionName,
                                        session_id=data['session_id'],
                                        isDocker=False, cam2Use=cam2Use, 
                                        intrinsicsFinalFolder=intrinsicsFinalFolder,
                                        extrinsicsTrial=extrinsicsTrial,
                                        alternateExtrinsics=alternateExtrinsics,
                                        markerDataFolderNameSuffix=cameraSetup,
                                        imageUpsampleFactor=imageUpsampleFactor,
                                        poseDetector=poseDetector,
                                        resolutionPoseDetection=resolutionPoseDetection,
                                        scaleModel=scaleModel, bbox_thr=bbox_thr, 
                                        augmenter_model=augmenter_model)
                        except:
                            # Append name to a file
                            with open('failedTrials.txt', 'a') as f:
                                f.write('Failed trial: {} {} {} {} {} {} \n'.format(sessionName, data['trials'][trial]["id"], poseDetector, resolutionPoseDetection, cameraSetup, augmenter_model))
                            # print('Failed trial: ' + sessionName + ' ' + data['trials'][trial]["id"] + ' ' + poseDetector + ' ' + resolutionPoseDetection + ' ' + cameraSetup + ' ' + augmenter_model)
                            print('Failed trial: {} {} {} {} {} {}'.format(sessionName, data['trials'][trial]["id"], poseDetector, resolutionPoseDetection, cameraSetup, augmenter_model))


                            continue    


    print("DONE: video to markers")

if poseDetector == 'mmpose':
    suff_pd = '_' + str(bbox_thr)
elif poseDetector == 'OpenPose':
    suff_pd = '_' + resolutionPoseDetection        
poseDetector_name = poseDetector + suff_pd

# %% Syncing with mocap
if syncMocapVideo:
    videoParameters = {}
    videoParameters['0001'] = {}
    videoParameters['0001']['originName'] = 'extrinsicLargeOvergroundOrigin1.trc'
    videoParameters['0001']['r_fromMarker_toVideoOrigin_inLab'] = np.array([0 , 7 ,0])  #mm in lab frame # this one is for large backwall 0001
    videoParameters['0001']['R_video_opensim'] = R.from_euler('y',-90,degrees=True)
    videoParameters['0001']['R_opensim_xForward'] = R.from_euler('y',90,degrees=True)
    videoParameters['0001']['mocapFiltFreq'] = 30

    videoParameters['0002'] = {}
    videoParameters['0002']['originName'] = 'extrinsicLargeWalkingOrigin1.trc'
    videoParameters['0002']['r_fromMarker_toVideoOrigin_inLab'] = np.array([-7, 0, 0]) # mm in lab frame # this one is for large backwall 0001
    videoParameters['0002']['R_video_opensim'] = R.from_euler('y',0,degrees=True)
    videoParameters['0002']['R_opensim_xForward'] = R.from_euler('y',0,degrees=True)
    videoParameters['0002']['mocapFiltFreq'] = 12

    MPJEs = {}
    for subjectName in sessionDetails:
        c_sessions = sessionDetails[subjectName]
        MPJEs[subjectName] = main_sync(dataDir, subjectName, c_sessions, [poseDetector_name], cameraSetups, augmenter_models, videoParameters, saveProcessedMocapData=False,
        overwriteMarkerDataProcessed=False, overwriteForceDataProcessed=False,
        overwritevideoAndMocap=False, writeMPJE_condition=True, writeMPJE_session=True,
        csv_name='MPJE_fullSession_do_not_trust')

    print("DONE: syncing to mocap")


# %% Gather data from different sessions
if gatherData:
    for subjectName in sessionDetails:
        c_sessions = sessionDetails[subjectName]
        main_gather(dataDir, subjectName, c_sessions, [poseDetector_name], cameraSetups, augmenter_models)

    print("DONE: gathering data")

# OpenSim pipeline
if runOpenSim:
    scriptDir = os.getcwd()
    repoDir = os.path.dirname(scriptDir)
    opensimPipelineDir = os.path.join(repoDir, 'opensimPipeline')

    for subjectName in sessionDetails:
        runOpenSimPipeline(dataDir, opensimPipelineDir, subjectName, [poseDetector_name], cameraSetups, augmenter_models, runMocap=False, runVideoAugmenter=True, runVideoPose=False, withTrackingMarkers=withTrackingMarkers)

    print("DONE: OpenSim pipeline")
        
test=1


