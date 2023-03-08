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

# %% Validation: please keep here, hack to get all trials at once.
# sessionNames = [
#                 'Session20210813_0001', 'Session20210813_0002', 
#                 'Session20210816_0001', 'Session20210816_0002',
#                 'Session20210819_0001', 'Session20210819_0002',
#                 'Session20210820_0001', 'Session20210820_0002', 
#                 'Session20210823_0001', 'Session20210823_0002',
#                 'Session20210825_0001', 'Session20210825_0002', 
#                 'Session20210827_0001', 'Session20210827_0002', 
#                 'Session20210830_0001', 'Session20210830_0002', 
#                 'Session20210903_0001', 'Session20210903_0002', 
#                 'Session20210910_0001', 'Session20210910_0002']

# sessionDetails = {
    
#     'subject2': {
#         'Session20210813_0001': {},
#         'Session20210813_0002': {}},
    
#     'subject3': {
#         'Session20210816_0001': {},
#         'Session20210816_0002': {}},
    
#     'subject4': {
#         'Session20210819_0001': {},
#         'Session20210819_0002': {}},
    
#     'subject5': {
#         'Session20210820_0001': {},
#         'Session20210820_0002': {}},
    
#     'subject6': {
#         'Session20210823_0001': {},
#         'Session20210823_0002': {}},
    
#     'subject7': {
#         'Session20210825_0001': {},
#         'Session20210825_0002': {}},
    
#     'subject8': {
#         'Session20210827_0001': {},
#         'Session20210827_0002': {}},
    
#     'subject9': {
#         'Session20210830_0001': {},
#         'Session20210830_0002': {}},
    
#     'subject10': {
#         'Session20210903_0001': {},
#         'Session20210903_0002': {}},
    
#     'subject11': {
#         'Session20210910_0001': {},
#         'Session20210910_0002': {}},

#     }

videoToMarkers = False
syncMocapVideo = False
gatherData = False
runOpenSim = True

sessionNames = ['Session20210813_0002']
sessionDetails = {    
    'subject2': {
        'Session20210813_0001': {},
        'Session20210813_0002': {}}}
poseDetectors = ['OpenPose']
# cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
cameraSetups = ['2-cameras']
augmenter_model = 'v0.8'

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
    for poseDetector in poseDetectors:
        for cameraSetup in cameraSetups:
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
                    
                # resolutionPoseDetection = 'default' # default
                resolutionPoseDetection = '1x1008_4scales'
                if "resolutionPoseDetection" in data['trials'][trial]:
                    resolutionPoseDetection = data['trials'][trial]['resolutionPoseDetection']
                    
                scaleModel = False # default
                if "scaleModel" in data['trials'][trial]:
                    scaleModel = data['trials'][trial]['scaleModel']
                    
                bbox_thr = 0.8 # default
                if "bbox_thr" in data['trials'][trial]:
                    bbox_thr = data['trials'][trial]['bbox_thr']
                    
                if videoToMarkers:
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
        MPJEs[subjectName] = main_sync(dataDir, subjectName, c_sessions, [poseDetector_name], cameraSetups, [augmenter_model], videoParameters, saveProcessedMocapData=False,
        overwriteMarkerDataProcessed=False, overwriteForceDataProcessed=False,
        overwritevideoAndMocap=True, writeMPJE_condition=True, writeMPJE_session=True,
        csv_name='MPJE_fullSession_v0.8')


# %% Gather data from different sessions
if gatherData:
    for subjectName in sessionDetails:
        c_sessions = sessionDetails[subjectName]
        main_gather(dataDir, subjectName, c_sessions, [poseDetector_name], cameraSetups, [augmenter_model])

# OpenSim pipeline
if runOpenSim:
    scriptDir = os.getcwd()
    repoDir = os.path.dirname(scriptDir)
    opensimPipelineDir = os.path.join(repoDir, 'opensimPipeline')

    for subjectName in sessionDetails:
        runOpenSimPipeline(dataDir, opensimPipelineDir, subjectName, [poseDetector_name], cameraSetups, [augmenter_model], runMocap=False, runVideoAugmenter=True, runVideoPose=False)
        
test=1


