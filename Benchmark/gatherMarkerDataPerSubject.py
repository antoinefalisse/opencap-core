"""
    This script:
        Gathers marker data of each subject in one folder, instead of in many
        session folders.        
"""

import os
import sys
sys.path.append("..") # utilities in child directory
import shutil

from utils import getDataDirectory

# %% Session details
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
        'Session20210910_0002': {}}

    }
poseDetectors = ['OpenPose_1x1008_4scales']
cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
augmenterTypes = ['v0.7']

# %% Process
mainDir = getDataDirectory(False)
dataDir = os.path.join(mainDir, 'Data')

# Video data
for subjectName in sessionDetails:
    subjectDir = os.path.join(dataDir, subjectName, 'MarkerData', 'Video')
    for poseDetector in poseDetectors:
        poseDetectorDir = os.path.join(subjectDir, poseDetector)
        for cameraSetup in cameraSetups:
            cameraSetupDir = os.path.join(poseDetectorDir, cameraSetup)            
            for augmenterType in augmenterTypes:
                augmenterTypeDir = os.path.join(cameraSetupDir, augmenterType)
                os.makedirs(augmenterTypeDir, exist_ok=True)
                for sessionName in sessionDetails[subjectName]:
                    postDir = 'PostAugmentation_' + augmenterType
                    trcDir = os.path.join(dataDir, sessionName, 'MarkerData', 
                                          poseDetector, cameraSetup, 
                                          postDir, 'videoAndMocap')
                    for file in os.listdir(trcDir):
                        if file[-3:] == 'trc':
                            pathFile = os.path.join(trcDir, file)
                            pathFileEnd = os.path.join(augmenterTypeDir, file)
                            shutil.copy2(pathFile, pathFileEnd)
                            
                            pathFileEndOld = os.path.join(cameraSetupDir, file)
                            if os.path.exists(pathFileEndOld):
                                print('In 2')
                                # os.remove(pathFileEndOld)

# # Mocap data
# for subjectName in sessionDetails:
#     subjectDir = os.path.join(dataDir, subjectName, 'MarkerData', 'Mocap')
#     os.makedirs(subjectDir, exist_ok=True)
#     for sessionName in sessionDetails[subjectName]:
#         trcDir = os.path.join(dataDir, sessionName, 'mocap', 
#                               'MarkerDataProcessed')
#         for file in os.listdir(trcDir):
#             if file[-3:] == 'trc':
#                 pathFile = os.path.join(trcDir, file)
#                 pathFileEnd = os.path.join(subjectDir, file)
#                 shutil.copy2(pathFile, pathFileEnd)
                
# # Force data
# for subjectName in sessionDetails:
#     subjectDir = os.path.join(dataDir, subjectName, 'ForceData')
#     os.makedirs(subjectDir, exist_ok=True)
#     for sessionName in sessionDetails[subjectName]:
#         trcDir = os.path.join(dataDir, sessionName, 'mocap', 
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
#             pathFile = os.path.join(dataDir, sessionName, file)
#             pathFileEnd = os.path.join(dataDir, subjectName, file)
#             shutil.copy2(pathFile, pathFileEnd)
