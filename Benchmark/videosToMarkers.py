import sys
sys.path.append("./DataProcessing")
from videosID import getData
from utils import downloadVideosFromServer
from main import main

# %% Validation: please keep here, hack to get all trials at once.
sessionNames = [
                'Session20210813_0001', 'Session20210813_0002', 
                'Session20210816_0001', 'Session20210816_0002',
                'Session20210819_0001', 'Session20210819_0002',
                'Session20210820_0001', 'Session20210820_0002', 
                'Session20210823_0001', 'Session20210823_0002',
                'Session20210825_0001', 'Session20210825_0002', 
                'Session20210827_0001', 'Session20210827_0002', 
                'Session20210830_0001', 'Session20210830_0002', 
                'Session20210903_0001', 'Session20210903_0002', 
                'Session20210910_0001', 'Session20210910_0002']
poseDetectors = ['OpenPose']
cameraSetups = ['5-cameras']

# %% Functions for processing the data.
def process_trial(trial_id, trial_name=None, session_name='', isDocker=False,
                  session_id=None, cam2Use=['all'],
                  intrinsicsFinalFolder='Deployed', extrinsicsTrial=False,
                  alternateExtrinsics=None, markerDataFolderNameSuffix=None,
                  imageUpsampleFactor=4, poseDetector='OpenPose',
                  resolutionPoseDetection='default', scaleModel=False, bbox_thr=0.8, 
                  augmenter_type='separateLowerUpperBody',
                  augmenter_markers='OpenPose'):
    
    # Download videos
    trial_name = downloadVideosFromServer(session_id, trial_id, isDocker=True, 
                                          trial_name=trial_name,
                                          session_name=session_name)

    # Run main.
    main(session_name, trial_name, cam2Use, intrinsicsFinalFolder, isDocker,
          extrinsicsTrial, alternateExtrinsics, markerDataFolderNameSuffix,
          imageUpsampleFactor, poseDetector,
          resolutionPoseDetection=resolutionPoseDetection, scaleModel=scaleModel, 
          bbox_thr=bbox_thr, augmenter_type=augmenter_type, 
          augmenter_markers=augmenter_markers)

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
                    
                augmenter_type = 'separateLowerUpperBody' # default
                if "augmenter_type" in data['trials'][trial]:
                    augmenter_type = data['trials'][trial]['augmenter_type']
                    
                augmenter_markers = 'OpenPose' # default
                if "augmenter_markers" in data['trials'][trial]:
                    augmenter_markers = data['trials'][trial]['augmenter_markers']
                    
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
                              augmenter_type=augmenter_type, 
                              augmenter_markers=augmenter_markers)

    print("DONE")
