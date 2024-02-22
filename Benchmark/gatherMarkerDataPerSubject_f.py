"""
    This script:
        Gathers marker data of each subject in one folder, instead of in many
        session folders.        
"""

import os
import sys
sys.path.append("..") # utilities in child directory
import shutil

def main_gather(dataDir, subjectName, c_sessions, poseDetectors, cameraSetups, augmenterTypes, allVideoOnly=False):


    if allVideoOnly:
        subjectDir = os.path.join(dataDir, 'Data', subjectName, 'MarkerData', 'Video')
        for poseDetector in poseDetectors:
            poseDetectorDir = os.path.join(subjectDir, poseDetector)
            for cameraSetup in cameraSetups:
                cameraSetupDir = os.path.join(poseDetectorDir, cameraSetup)            
                for augmenterType in augmenterTypes:
                    augmenterTypeDir = os.path.join(cameraSetupDir, augmenterType + '_allVideoOnly')
                    os.makedirs(augmenterTypeDir, exist_ok=True)
                    for sessionName in c_sessions:
                        postDir = 'PostAugmentation_' + augmenterType
                        trcDir = os.path.join(dataDir, 'Data', sessionName, 'MarkerData', 
                                                poseDetector, cameraSetup, 
                                                postDir, 'videoAndMocap')
                        for file in os.listdir(trcDir):
                            if file.endswith('_video.trc'):
                                pathFile = os.path.join(trcDir, file)
                                pathFileEnd = os.path.join(augmenterTypeDir, file)
                                shutil.copy2(pathFile, pathFileEnd)
    else:
        subjectDir = os.path.join(dataDir, 'Data', subjectName, 'MarkerData', 'Video')
        for poseDetector in poseDetectors:
            poseDetectorDir = os.path.join(subjectDir, poseDetector)
            for cameraSetup in cameraSetups:
                cameraSetupDir = os.path.join(poseDetectorDir, cameraSetup)            
                for augmenterType in augmenterTypes:
                    augmenterTypeDir = os.path.join(cameraSetupDir, augmenterType)
                    os.makedirs(augmenterTypeDir, exist_ok=True)
                    for sessionName in c_sessions:
                        postDir = 'PostAugmentation_' + augmenterType
                        trcDir = os.path.join(dataDir, 'Data', sessionName, 'MarkerData', 
                                                poseDetector, cameraSetup, 
                                                postDir, 'videoAndMocap')
                        for file in os.listdir(trcDir):
                            if file.endswith('_videoAndMocap.trc'):
                                pathFile = os.path.join(trcDir, file)
                                pathFileEnd = os.path.join(augmenterTypeDir, file)
                                shutil.copy2(pathFile, pathFileEnd)
                                
                                # pathFileEndOld = os.path.join(cameraSetupDir, file)
                                # if os.path.exists(pathFileEndOld):
                                #     print('In 2')
                                    # os.remove(pathFileEndOld)

# # Mocap data
# for subjectName in sessionDetails:
#     subjectDir = os.path.join(dataDir, 'Data', subjectName, 'MarkerData', 'Mocap')
#     os.makedirs(subjectDir, exist_ok=True)
#     for sessionName in sessionDetails[subjectName]:
#         trcDir = os.path.join(dataDir, 'Data', sessionName, 'mocap', 
#                               'MarkerDataProcessed')
#         for file in os.listdir(trcDir):
#             if file[-3:] == 'trc':
#                 pathFile = os.path.join(trcDir, file)
#                 pathFileEnd = os.path.join(subjectDir, file)
#                 shutil.copy2(pathFile, pathFileEnd)
                
# # Force data
# for subjectName in sessionDetails:
#     subjectDir = os.path.join(dataDir, 'Data', subjectName, 'ForceData')
#     os.makedirs(subjectDir, exist_ok=True)
#     for sessionName in sessionDetails[subjectName]:
#         trcDir = os.path.join(dataDir, 'Data', sessionName, 'mocap', 
#                               'ForceDataProcessed')
#         for file in os.listdir(trcDir):
#             if file[-3:] == 'mot':
#                 pathFile = os.path.join(trcDir, file)
#                 pathFileEnd = os.path.join(subjectDir, file)
#                 shutil.copy2(pathFile, pathFileEnd)

# # Metadata
# for subjectName in sessionDetails:
#     for count, sessionName in enumerate(sessionDetails[subjectName]):
#         if count == 0:        
#             file = 'sessionMetadata.yaml'
#             pathFile = os.path.join(dataDir, 'Data', sessionName, file)
#             pathFileEnd = os.path.join(dataDir, 'Data', subjectName, file)
#             shutil.copy2(pathFile, pathFileEnd)
