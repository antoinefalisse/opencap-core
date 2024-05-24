"""
    This script computes and analyses the RMSEs between mocap-based on video-
    based kinematics.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
sys.path.append("..") # utilities in child directory
from utils import getDataDirectory, storage2df
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import utilsDataman as dm
import seaborn as sns
import copy
import scipy.stats as stats

plots = False
saveAndOverwriteResults = True
overwriteResults = False

scriptDir = os.getcwd()
repoDir = os.path.dirname(scriptDir)
mainDir = getDataDirectory(False)
dataDir = os.path.join(mainDir)
outputDir = os.path.join(dataDir, 'Results-paper-augmenterV2-updated')
os.makedirs(outputDir, exist_ok=True)

# %% User inputs.
subjects = ['subject' + str(i) for i in range(2, 12)]

# poseDetectors = ['OpenPose_1x1008_4scales']
# poseDetectors = ['OpenPose_default']
# poseDetectors = ['OpenPose_1x736']
# poseDetectors = ['OpenPose_1x736', 'OpenPose_1x1008_4scales']

# poseDetectors = ['OpenPose_default', 'OpenPose_1x736', 'OpenPose_1x1008_4scales']
poseDetectors = ['mmpose_0.8']
cameraSetups = ['2-cameras']
augmenterTypes = {
    'pose_updated': {'run': False},
    'v0.1_updated': {'run': False},
    'v0.2_updated': {'run': False},
    # 'v0.63': {'run': False},
    # 'v0.45': {'run': False},
    # 'v0.54': {'run': False},
    # 'v0.57': {'run': True},
    # 'v0.58': {'run': True},
    
    # 'v0.55': {'run': False},
    'v0.63_updated': {'run': False},
    # 'v0.71': {'run': True},
    # 'v0.70': {'run': False},
    'v1.0_updated': {'run': False},
    'v2.0_updated': {'run': False},
    # 'v0.62': {'run': False},
    # 'v0.63': {'run': False},
    # 'v0.63': {'run': True},
}

# setups_t = list(augmenterTypes.keys())
setups_t = ['Video keypoints', 'Uhlrich et al. 2023', 'OpenCap 1.0', 'LSTM', 'Transformer', 'Linear regression']

# processingTypes = ['IK_IK', 'addB_addB', 'IK_addB', 'addB_IK']
processingTypes = ['IK_IK']

# Cases to exclude for paper
cases_to_exclude_paper = ['static', 'stsasym', 'stsfast', 'walkingti', 'walkingto']

# Old
# augmenterTypeOffset = 'v0.7'
# Cases to exclude to make sure we have the same number of trials per subject
# cases_to_exclude_trials = {'subject2': ['walkingTS3']}
# Cases to exclude because of failed syncing (mocap vs opencap)
# cases_to_exclude_syncing = {}
# cases_to_exclude_syncing = {
#     'subject3': {'OpenPose_1x1008_4scales': {'2-cameras': ['walking1', 'walkingTI1', 'walkingTI2', 'walkingTO1', 'walkingTO2', 'walkingTS3', 'walkingTS4']}}}
# # Cases to exclude because of failed algorithm (opencap)
# cases_to_exclude_algo = {
#     'subject2': {'OpenPose_default': {'3-cameras': ['walkingTS1', 'walkingTS2', 'walkingTS4', 'DJ1', 'DJ2', 'DJ3', 'DJAsym1', 'DJAsym4', 'DJAsym5'],
#                                       '5-cameras': ['walkingTS2']}},
#     'subject3': {'OpenPose_default': {'2-cameras': ['walkingTS2', 'walkingTS4']},
#                  'mmpose_0.8': {'2-cameras': ['STSweakLegs1']}}}

# New
# Cases to exclude to make sure we have the same number of trials per subject
cases_to_exclude_trials = {'subject2': ['walkingTS3']}
# Cases to exclude because of failing syncing / algo.
cases_to_exclude_syncing = {}
cases_to_exclude_algo = {}
# cases_to_exclude_algo = {

#     # 'subject8': {'mmpose_0.8': {'2-cameras': ['walkingTS2']}},
#     # 'subject3': {'OpenPose_default': {'2-cameras': ['walking1', 'walkingTS3']},
#     #              'OpenPose_1x736': {'2-cameras': ['walking1', 'walkingTS3', 'walkingTS4']},
#     #              'OpenPose_1x1008_4scales': {'2-cameras': ['walking1', 'walkingTS3', 'walkingTS4']},
#     #              'mmpose_0.8': {'2-cameras': ['STSweakLegs1']},

#     # Subject 2
#     'subject2': {'OpenPose_1x1008_4scales': {'5-cameras': {'v0.45': ['walkingTS4'],
#                                                            'v0.54': ['walkingTS4']}},   # MPJE
#                  'mmpose_0.8': {'3-cameras': {'v0.45': ['walking1', 'walkingTS2'],   # algo  
#                                                'v0.54': ['walking1', 'walkingTS2'],   # algo 
#                                                'v0.55': ['walking1', 'walkingTS2'],   # algo 
#                                                'v0.56': ['walking1', 'walkingTS2']},   # algo 
#                                  '5-cameras': {'v0.45': ['walking1', 'walkingTS2'],   # algo 
#                                                'v0.54': ['walking1', 'walkingTS2'],   # algo 
#                                                'v0.55': ['walking1', 'walkingTS2'],   # algo 
#                                                'v0.56': ['walking1', 'walkingTS2']}},  # algo 
#                  },
#     # Subject 3
#     'subject3': {'OpenPose_1x1008_4scales': {'2-cameras': {'v0.1': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.2': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.45': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.54': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.61': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.62': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.62': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.63': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.66': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.67': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.68': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.69': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                             'v0.70': ['walking1', 'walkingTS3', 'walkingTS4']}},# MPJE
#                   'OpenPose_1x736': {'2-cameras': {'v0.1': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                    'v0.2': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                    'v0.45': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                    'v0.54': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                    'v0.57': ['walking1', 'walkingTS3', 'walkingTS4'],# MPJE
#                                                    'v0.58': ['walking1', 'walkingTS3', 'walkingTS4']}},# MPJE
#                   'mmpose_0.8': {'2-cameras': {'v0.45': ['STSweakLegs1'],# algo 
#                                                'v0.54': ['STSweakLegs1'],# algo 
#                                                'v0.55': ['STSweakLegs1'],# algo 
#                                                'v0.56': ['STSweakLegs1'],# algo
#                                                'v0.57': ['STSweakLegs1'],# algo
#                                                'v0.58': ['STSweakLegs1'],# algo
#                                                'v0.59': ['STSweakLegs1'],# algo,
#                                                'v0.60': ['STSweakLegs1']},# algo},
#                                 '3-cameras': {'v0.45': ['STSweakLegs1'],# algo 
#                                                 'v0.54': ['STSweakLegs1'],# algo 
#                                                 'v0.55': ['STSweakLegs1'],# algo 
#                                                 'v0.56': ['STSweakLegs1']},# algo 
#                                 '5-cameras': {'v0.45': ['STSweakLegs1'],# algo 
#                                                 'v0.54': ['STSweakLegs1'],# algo 
#                                                 'v0.55': ['STSweakLegs1'],# algo 
#                                                 'v0.56': ['STSweakLegs1']}},# algo 
#                   },
#     # Subject 4
#     'subject4': {'OpenPose_1x736': {'3-cameras': {'v0.1': ['walkingTS2'],# MPJE
#                                                   'v0.2': ['walkingTS2'],# MPJE
#                                                   'v0.45': ['walkingTS2'],# MPJE
#                                                   'v0.54': ['walkingTS2']}},# MPJE 
#                  'mmpose_0.8': {'5-cameras': {'v0.45': ['squats1'],# algo 
#                                               'v0.54': ['squats1'],# algo 
#                                               'v0.55': ['squats1'],# algo 
#                                               'v0.56': ['squats1']}},# algo
#                  },
#     # Subject 5
#     'subject5': {'mmpose_0.8': {'3-cameras': {'v0.45': ['walking1', 'walking2', 'walkingTS3'],# algo 
#                                               'v0.54': ['walking1', 'walking2', 'walkingTS3'],# algo 
#                                               'v0.55': ['walking1', 'walking2', 'walkingTS3'],# algo 
#                                               'v0.56': ['walking1', 'walking2', 'walkingTS3']},
#                                 '5-cameras': {'v0.45': ['walking1', 'walking2', 'walkingTS3'],# algo 
#                                                 'v0.54': ['walking1', 'walking2', 'walkingTS3'],# algo 
#                                                 'v0.55': ['walking1', 'walking2', 'walkingTS3'],# algo 
#                                                 'v0.56': ['walking1', 'walking2', 'walkingTS3']}},# algo
#                  },
#     # Subject 7
#     'subject7': {'mmpose_0.8': {'5-cameras': {'v0.45': ['walking1'],# algo 
#                                               'v0.54': ['walking1'],# algo 
#                                               'v0.55': ['walking1'],# algo 
#                                               'v0.56': ['walking1']}},# algo
#                  },
#     # Subject 8
#     'subject8': {'mmpose_0.8': {'5-cameras': {'v0.45': ['DJAsym3'],# algo 
#                                               'v0.54': ['DJAsym3'],# algo 
#                                               'v0.55': ['DJAsym3'],# algo 
#                                               'v0.56': ['DJAsym3']},
#                                 '2-cameras': {'v0.45': ['walkingTS2'],# MPJE 
#                                                 'v0.54': ['walkingTS2'],# MPJE 
#                                                 'v0.55': ['walkingTS2'],# MPJE 
#                                                 'v0.56': ['walkingTS2'],# MPJE
#                                                 'v0.57': ['walkingTS2'],# MPJE
#                                                 'v0.58': ['walkingTS2'],# MPJE
#                                                 'v0.59': ['walkingTS2'],# MPJE
#                                                 'v0.60': ['walkingTS2'],# MPJE
#                                                 'v0.63': ['walkingTS2'],# MPJE
#                                                 'v0.64': ['walkingTS2'],# MPJE
#                                                 'v0.65': ['walkingTS2'],# MPJE
#                                                 'v0.66': ['walkingTS2'],# MPJE
#                                                 'v0.67': ['walkingTS2'],# MPJE
#                                                 'v0.68': ['walkingTS2'],# MPJE
#                                                 'v0.69': ['walkingTS2'],# MPJE
#                                                 'v0.70': ['walkingTS2'],# MPJE
#                                                 'v0.71': ['walkingTS2'],# MPJE
#                                                 'v1.0': ['walkingTS2']}},# MPJE
#                  },
#     # Subject 9
#     'subject9': {'mmpose_0.8': {'5-cameras': {'v0.45': ['STS1'],# algo 
#                                               'v0.54': ['STS1'],# algo 
#                                               'v0.55': ['STS1'],# algo 
#                                               'v0.56': ['STS1']}},# algo
#                  },
#     # Subject 10
#     'subject10': {'mmpose_0.8': {'5-cameras': {'v0.45': ['walkingTS1'],# MPJE 
#                                               'v0.54': ['walkingTS1'],# MPJE 
#                                               'v0.55': ['walkingTS1'],# MPJE 
#                                               'v0.56': ['walkingTS1']}},# MPJE
#                  },
#     # Subject 11
#     'subject11': {'mmpose_0.8': {'5-cameras': {'v0.45': ['walkingTS3'],# algo 
#                                               'v0.54': ['walkingTS3'],# algo 
#                                               'v0.55': ['walkingTS3'],# algo 
#                                               'v0.56': ['walkingTS3']},
#                                  '3-cameras': {'v0.45': ['walkingTS3'],# algo 
#                                                 'v0.54': ['walkingTS3'],# algo 
#                                                 'v0.55': ['walkingTS3'],# algo 
#                                                 'v0.56': ['walkingTS3']}},# algo
#                  },
#     }

# %%
genericModel4ScalingName = 'LaiArnoldModified2017_poly_withArms_weldHand.osim'
coordinates = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_l', 'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r',
    'subtalar_angle_l', 'subtalar_angle_r',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
nCoordinates = len(coordinates)
# Bilateral coordinates
coordinates_bil = [
    'hip_flexion', 'hip_adduction', 'hip_rotation',
    'knee_angle', 'ankle_angle', 'subtalar_angle']
# coordinates without the right side, such that bilateral coordinates are 
# combined later. 
coordinates_lr = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
coordinates_lr_tr = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
coordinates_lr_rot = coordinates_lr.copy()
# Translational coordinates.
coordinates_tr = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
for coordinate in coordinates_lr_tr:
    coordinates_lr_rot.remove(coordinate)
motions = ['walking', 'DJ', 'squats', 'STS']

addBiomechanicsMocapModel = 'LaiArnold2107_OpenCapMocap'
addBiomechanicsVideoModel = 'LaiArnold2107_OpenCapVideo'

fixed_markers = False # False should be default (better results)

# %%
# if addBiomechanicsMocap:
#     suffix_files = '_addB'
# else:
#     suffix_files = ''


if not os.path.exists(os.path.join(outputDir, 'RMSEs.npy')): 
    RMSEs = {}
else:  
    RMSEs = np.load(os.path.join(outputDir, 'RMSEs.npy'), allow_pickle=True).item()    
if not os.path.exists(os.path.join(outputDir, 'MAEs.npy')): 
    MAEs = {}
else:  
    MAEs = np.load(os.path.join(outputDir, 'MAEs.npy'), allow_pickle=True).item()
if not os.path.exists(os.path.join(outputDir, 'MEs.npy')): 
    MEs = {}
else:  
    MEs = np.load(os.path.join(outputDir, 'MEs.npy'), allow_pickle=True).item()

if not 'all' in RMSEs:
    RMSEs['all'] = {}
if not 'all' in MAEs:
    MAEs['all'] = {}
if not 'all' in MEs:
    MEs['all'] = {}
    
for motion in motions:
    if not motion in RMSEs:
        RMSEs[motion] = {}
    if not motion in MAEs:
        MAEs[motion] = {}
    if not motion in MEs:
        MEs[motion] = {}
    
for subjectName in subjects:
    if not subjectName in RMSEs:
        RMSEs[subjectName] = {}
    if not subjectName in MAEs:
        MAEs[subjectName] = {}
    if not subjectName in MEs:
        MEs[subjectName] = {}
    print('\nProcessing {}'.format(subjectName))
    if fixed_markers:
        osDir = os.path.join(dataDir, 'Data', subjectName, 'OpenSimData_fixed')
    else:
        osDir = os.path.join(dataDir, 'Data', subjectName, 'OpenSimData')
    markerDir = os.path.join(dataDir, 'Data', subjectName, 'MarkerData')
        
    # Hack
    mocapDirAll = os.path.join(osDir, 'Mocap_updated', 'IK', genericModel4ScalingName[:-5])    
    trials = []
    for trial in os.listdir(mocapDirAll):
        if not trial[-3:] == 'mot':
            continue
        trials.append(trial[:-4])
    count = 0
    for trial in os.listdir(mocapDirAll):
        if not trial[-3:] == 'mot':
            continue        
        for poseDetector in poseDetectors:
            if not poseDetector in RMSEs[subjectName]:
                RMSEs[subjectName][poseDetector] = {}
            if not poseDetector in RMSEs['all']:
                RMSEs['all'][poseDetector] = {}                
            for motion in motions:
                if not poseDetector in RMSEs[motion]:
                    RMSEs[motion][poseDetector] = {}
                    
            if not poseDetector in MAEs[subjectName]:
                MAEs[subjectName][poseDetector] = {}
            if not poseDetector in MAEs['all']:
                MAEs['all'][poseDetector] = {}                
            for motion in motions:
                if not poseDetector in MAEs[motion]:
                    MAEs[motion][poseDetector] = {} 

            if not poseDetector in MEs[subjectName]:
                MEs[subjectName][poseDetector] = {}
            if not poseDetector in MEs['all']:
                MEs['all'][poseDetector] = {}                
            for motion in motions:
                if not poseDetector in MEs[motion]:
                    MEs[motion][poseDetector] = {}
                
            poseDetectorDir = os.path.join(osDir, 'Video', poseDetector)
            poseDetectorMarkerDir = os.path.join(markerDir, 'Video', poseDetector)
            for cameraSetup in cameraSetups:
                if not cameraSetup in RMSEs[subjectName][poseDetector]:
                    RMSEs[subjectName][poseDetector][cameraSetup] = {}
                if not cameraSetup in RMSEs['all'][poseDetector]:
                    RMSEs['all'][poseDetector][cameraSetup] = {}
                for motion in motions:
                    if not cameraSetup in RMSEs[motion][poseDetector]:
                        RMSEs[motion][poseDetector][cameraSetup] = {}
                    
                if not cameraSetup in MAEs[subjectName][poseDetector]:
                    MAEs[subjectName][poseDetector][cameraSetup] = {}
                if not cameraSetup in MAEs['all'][poseDetector]:
                    MAEs['all'][poseDetector][cameraSetup] = {}
                for motion in motions:
                    if not cameraSetup in MAEs[motion][poseDetector]:
                        MAEs[motion][poseDetector][cameraSetup] = {}

                if not cameraSetup in MEs[subjectName][poseDetector]:
                    MEs[subjectName][poseDetector][cameraSetup] = {}
                if not cameraSetup in MEs['all'][poseDetector]:
                    MEs['all'][poseDetector][cameraSetup] = {}
                for motion in motions:
                    if not cameraSetup in MEs[motion][poseDetector]:
                        MEs[motion][poseDetector][cameraSetup] = {}
                    
                for augmenterType in augmenterTypes:
                    
                    if not augmenterType in RMSEs[subjectName][poseDetector][cameraSetup]:
                        RMSEs[subjectName][poseDetector][cameraSetup][augmenterType] = {}
                    if not augmenterType in RMSEs['all'][poseDetector][cameraSetup]:
                        RMSEs['all'][poseDetector][cameraSetup][augmenterType] = {}
                    for motion in motions:
                        if not augmenterType in RMSEs[motion][poseDetector][cameraSetup]:
                            RMSEs[motion][poseDetector][cameraSetup][augmenterType] = {}
                            
                    if not augmenterType in MAEs[subjectName][poseDetector][cameraSetup]:
                        MAEs[subjectName][poseDetector][cameraSetup][augmenterType] = {}
                    if not augmenterType in MAEs['all'][poseDetector][cameraSetup]:
                        MAEs['all'][poseDetector][cameraSetup][augmenterType] = {}
                    for motion in motions:
                        if not augmenterType in MAEs[motion][poseDetector][cameraSetup]:
                            MAEs[motion][poseDetector][cameraSetup][augmenterType] = {}

                    if not augmenterType in MEs[subjectName][poseDetector][cameraSetup]:
                        MEs[subjectName][poseDetector][cameraSetup][augmenterType] = {}
                    if not augmenterType in MEs['all'][poseDetector][cameraSetup]:
                        MEs['all'][poseDetector][cameraSetup][augmenterType] = {}
                    for motion in motions:
                        if not augmenterType in MEs[motion][poseDetector][cameraSetup]:
                            MEs[motion][poseDetector][cameraSetup][augmenterType] = {}
                            
                    for processingType in processingTypes:  
                        
                        if processingType == 'IK_IK' or processingType == 'IK_addB':
                            addBiomechanicsMocap = False
                        else:
                            addBiomechanicsMocap = True
                            
                            
                        if processingType == 'IK_IK' or processingType == 'addB_IK':
                            addBiomechanicsVideo = False
                        else:
                            addBiomechanicsVideo = True                    

                        # if augmenterType in RMSEs[subjectName][poseDetector][cameraSetup] and augmenterType in MEs[subjectName][poseDetector][cameraSetup] and not overwriteResults:
                        #     continue
                    
                        # TODO
                        if not augmenterTypes[augmenterType]['run'] and not overwriteResults:
                            continue
    
                        if not processingType in RMSEs[subjectName][poseDetector][cameraSetup][augmenterType]:
                            RMSEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates, index=trials)
                        if not processingType in RMSEs['all'][poseDetector][cameraSetup][augmenterType]:
                            RMSEs['all'][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                        for motion in motions:
                            if not processingType in RMSEs[motion][poseDetector][cameraSetup][augmenterType]:
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                                
                        if not processingType in MAEs[subjectName][poseDetector][cameraSetup][augmenterType]:
                            MAEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates, index=trials)
                        if not processingType in MAEs['all'][poseDetector][cameraSetup][augmenterType]:
                            MAEs['all'][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                        for motion in motions:
                            if not processingType in MAEs[motion][poseDetector][cameraSetup][augmenterType]:
                                MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)

                        if not processingType in MEs[subjectName][poseDetector][cameraSetup][augmenterType]:
                            MEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates, index=trials)
                        if not processingType in MEs['all'][poseDetector][cameraSetup][augmenterType]:
                            MEs['all'][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                        for motion in motions:
                            if not processingType in MEs[motion][poseDetector][cameraSetup][augmenterType]:
                                MEs[motion][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                        
                        if (subjectName in cases_to_exclude_trials and
                            trial[:-4] in cases_to_exclude_trials[subjectName]):
                            # print('Exclude {} - {}'.format(subjectName, trial[:-4]))
                            continue
                        
                        # if (subjectName in cases_to_exclude_syncing and 
                        #     poseDetector in cases_to_exclude_syncing[subjectName] and 
                        #     cameraSetup in cases_to_exclude_syncing[subjectName][poseDetector] and 
                        #     trial[:-4] in cases_to_exclude_syncing[subjectName][poseDetector][cameraSetup]):
                        #     print('Exclude {} - {} - {} - {}'.format(subjectName, poseDetector, cameraSetup, trial[:-4]))
                        #     continue
                        
                        if (subjectName in cases_to_exclude_algo and 
                            poseDetector in cases_to_exclude_algo[subjectName] and 
                            cameraSetup in cases_to_exclude_algo[subjectName][poseDetector] and 
                            augmenterType in cases_to_exclude_algo[subjectName][poseDetector][cameraSetup] and 
                            trial[:-4] in cases_to_exclude_algo[subjectName][poseDetector][cameraSetup][augmenterType]):
                            print('Exclude (algo) {} - {} - {} - {} - {}'.format(subjectName, poseDetector, cameraSetup, augmenterType, trial[:-4]))
                            continue
                        
                        in_vec = False
                        for case_to_exclude_paper in cases_to_exclude_paper:                                
                            if case_to_exclude_paper in trial[:-4].lower():
                                # print('Exclude {}'.format(trial))  
                                in_vec = True
                        if in_vec:
                            continue
                        
                        
                        if addBiomechanicsMocap:
                            mocapDir = os.path.join(osDir, 'Mocap_updated', 'AddBiomechanics', 'IK', addBiomechanicsMocapModel)                              
                        else:
                            mocapDir = os.path.join(osDir, 'Mocap_updated', 'IK', genericModel4ScalingName[:-5])                              
                            
                        pathTrial = os.path.join(mocapDir, trial)
                        trial_mocap_df = storage2df(pathTrial, coordinates)
                            
                        if addBiomechanicsVideo:
                            cameraSetupDir = os.path.join(
                                poseDetectorDir, cameraSetup, augmenterType, 'AddBiomechanics', 'IK', 
                                addBiomechanicsVideoModel)
                        else:
                            cameraSetupDir = os.path.join(
                                poseDetectorDir, cameraSetup, augmenterType, 'IK', 
                                genericModel4ScalingName[:-5])                            
                            
                        # Used to use same augmenter for all for offset, not sure why
                        if augmenterType == 'pose_updated':
                            cameraSetupMarkerDir = os.path.join(
                                poseDetectorMarkerDir, cameraSetup, 'v0.1_updated')
                        else:
                            cameraSetupMarkerDir = os.path.join(
                                poseDetectorMarkerDir, cameraSetup, augmenterType)
                        
                        trial_video = trial[:-4] + '_videoAndMocap.mot'
                        trial_marker = trial[:-4] + '_videoAndMocap.trc'
                        pathTrial_video = os.path.join(cameraSetupDir, trial_video)
                        pathTrial_marker = os.path.join(cameraSetupMarkerDir, trial_marker)                        
                        
                        if not os.path.exists(pathTrial_video):
                            continue
                        
                        trial_video_df = storage2df(pathTrial_video, coordinates)
                        
                        # Convert to numpy
                        trial_mocap_np = trial_mocap_df.to_numpy()
                        trial_video_np = trial_video_df.to_numpy()
                        
                        # Extract start and end time from video
                        time_start_video = trial_video_np[0, 0]
                        time_end_video = trial_video_np[-1, 0]

                        # Extract start and end time from moca[]
                        time_start_mocap = trial_mocap_np[0, 0]
                        time_end_mocap = trial_mocap_np[-1, 0]
                        
                        # Pick the time vector that is the most restrictive
                        time_start = max(time_start_video, time_start_mocap)
                        time_end = min(time_end_video, time_end_mocap)
                        
                        # For DJ trials, we trimmed the OpenSim IK ones, but not the addb ones
                        # Let's load a reference OpenSim IK one to get the same time vector
                        if addBiomechanicsMocap and addBiomechanicsVideo and 'DJ' in trial:
                            mocapDir_temp = os.path.join(osDir, 'Mocap_updated', 'IK', genericModel4ScalingName[:-5])
                            pathTrial_temp = os.path.join(mocapDir_temp, trial)
                            trial_mocap_df_temp = storage2df(pathTrial_temp, coordinates)
                            trial_mocap_np_temp = trial_mocap_df_temp.to_numpy()
                            time_start_mocap_temp = trial_mocap_np_temp[0, 0]
                            time_end_mocap_temp = trial_mocap_np_temp[-1, 0]
                            time_start = max(time_start, time_start_mocap_temp)
                            time_end = min(time_end, time_end_mocap_temp)
                        
                        # Find corresponding indices in mocap data
                        idx_start_mocap = np.argwhere(trial_mocap_np[:, 0] == time_start)[0][0]
                        idx_end_mocap = np.argwhere(trial_mocap_np[:, 0] == time_end)[0][0]
                        # Select mocap data based on video-based time vector
                        trial_mocap_np_adj = trial_mocap_np[idx_start_mocap:idx_end_mocap+1, :]

                        # Find corresponding indices in video data
                        idx_start_video = np.argwhere(trial_video_np[:, 0] == time_start)[0][0]
                        idx_end_video = np.argwhere(trial_video_np[:, 0] == time_end)[0][0]
                        # Select video data based on video-based time vector
                        trial_video_np_adj = trial_video_np[idx_start_video:idx_end_video+1, :]               
                        
                        # Compute RMSEs and MEs
                        y_true = trial_mocap_np_adj[:, 1:]
                        y_pred = trial_video_np_adj[:, 1:]
                        
                        c_rmse = []
                        c_mae = []
                        c_me = []
                        
                        # If translational degree of freedom, adjust for offset
                        # Compute offset from trc file.
                        c_trc = dm.TRCFile(pathTrial_marker)
                        c_trc_m1 = c_trc.marker('Neck')
                        c_trc_m1_offsetRemoved = c_trc.marker('Neck_offsetRemoved')
                        # Verify same offset for different markers.
                        c_trc_m2 = c_trc.marker('L_knee_study')
                        c_trc_m2_offsetRemoved = c_trc.marker('L_knee_study_offsetRemoved')                    
                        c_trc_m1_offset = np.mean(c_trc_m1-c_trc_m1_offsetRemoved, axis=0)
                        c_trc_m2_offset = np.mean(c_trc_m2-c_trc_m2_offsetRemoved, axis=0)
                        assert (np.all(np.round(c_trc_m1_offset,2)==np.round(c_trc_m2_offset,2))), 'Problem offset'
                        
                        for count1 in range(y_true.shape[1]):
                            c_coord = coordinates[count1]
                            # If translational degree of freedom, adjust for offset
                            if c_coord in coordinates_tr:
                                if c_coord == 'pelvis_tx':
                                    y_pred_adj = y_pred[:, count1] - c_trc_m1_offset[0]/1000
                                elif c_coord == 'pelvis_ty':
                                    y_pred_adj = y_pred[:, count1] - c_trc_m1_offset[1]/1000
                                elif c_coord == 'pelvis_tz':
                                    y_pred_adj = y_pred[:, count1] - c_trc_m1_offset[2]/1000
                                y_true_adj = y_true[:, count1]
                            else:
                                # Convert to degrees if addBiomechanics
                                if addBiomechanicsMocap:
                                    y_true_adj = np.rad2deg(y_true[:, count1])
                                else:
                                    y_true_adj = y_true[:, count1]
                                if addBiomechanicsVideo:
                                    y_pred_adj = np.rad2deg(y_pred[:, count1])
                                else:
                                    y_pred_adj = y_pred[:, count1]
                                    
                            # RMSE
                            value = mean_squared_error(
                                y_true_adj, y_pred_adj, squared=False)
                            RMSEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType].loc[trials[count], c_coord] = value
                            c_rmse.append(value)
                            # MAE                        
                            value2 = mean_absolute_error(
                                y_true_adj, y_pred_adj)
                            MAEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType].loc[trials[count], c_coord] = value2
                            c_mae.append(value2)
                            # ME
                            # COmpute mean error between y_true_adj and y_pred_adj
                            value3 = np.mean(y_true_adj - y_pred_adj)
                            MEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType].loc[trials[count], c_coord] = value3
                            c_me.append(value3)
                            
                            if value > 20:
                                print('{} - {} - {} - {} - {} - {} - {} had RMSE of {}'.format(subjectName, poseDetector, cameraSetup, augmenterType, processingType, trial[:-4], c_coord, value))
                            
                        c_name = subjectName + '_' +  trial[:-4]
                        RMSEs['all'][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_rmse
                        MAEs['all'][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_mae
                        MEs['all'][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_me
                        
                        for motion in motions:
                            if motion in trial:
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_rmse
                                MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_mae
                                MEs[motion][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_me 

        count += 1
        
if saveAndOverwriteResults:
    np.save(os.path.join(outputDir, 'RMSEs.npy'), RMSEs)
    np.save(os.path.join(outputDir, 'MAEs.npy'), MAEs)
    np.save(os.path.join(outputDir, 'MEs.npy'), MEs)   
        
# %% Plots per coordinate: RMSEs
all_motions = ['all'] + motions
bps, means_RMSEs, medians_RMSEs, stds_RMSEs = {}, {}, {}, {}
for motion in all_motions:
    bps[motion], means_RMSEs[motion], medians_RMSEs[motion], stds_RMSEs[motion] = {}, {}, {}, {}
    if plots:
        fig, axs = plt.subplots(5, 3, sharex=True)    
        fig.suptitle(motion)    
    for count, coordinate in enumerate(coordinates_lr):
        c_data = {}
        for processingType in processingTypes:
            for augmenterType in augmenterTypes:
                for poseDetector in poseDetectors:
                    for cameraSetup in cameraSetups:                
                        if coordinate[-2:] == '_l':
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType] = (                    
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist() +                     
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate[:-2] + '_r'].tolist())
                            coordinate_title = coordinate[:-2]
                        else:
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType+ '_' + processingType] = (
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist())
                            coordinate_title = coordinate                    
        if plots:
            ax = axs.flat[count]
            bps[motion][coordinate] = ax.boxplot(c_data.values())
            ax.set_title(coordinate_title)
            xticks = list(range(1, len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)*len(processingTypes)+1))
            ax.set_xticks(xticks)
            ax.set_xticklabels(c_data.keys(), rotation = 90)
            ax.set_ylim(0, 20)
            ax.axhline(y=5, color='r', linestyle='--')
            ax.axhline(y=10, color='b', linestyle='--')
        
        means_RMSEs[motion][coordinate] = [np.mean(c_data[a]) for a in c_data]
        medians_RMSEs[motion][coordinate] = [np.median(c_data[a]) for a in c_data]
        stds_RMSEs[motion][coordinate] = [np.std(c_data[a]) for a in c_data]
        
# %% Print out csv files with results: median RMSEs
setups = []
for processingType in processingTypes:
    for augmenterType in augmenterTypes:
        for poseDetector in poseDetectors:
            for cameraSetup in cameraSetups:        
                setups.append(poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType)

suffixRMSE = ''
if fixed_markers:
    suffixRMSE = '_fixed'    
        
with open(os.path.join(outputDir,'RMSEs{}_medians.csv'.format(suffixRMSE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['motion-type', '', 'setup']
    for label in coordinates_lr:
        
        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label            
        
        topRow.extend([label_adj,''])
    _ = csvWriter.writerow(topRow)
    secondRow = ['', '', '']
    secondRow.extend(["median-RMSE",''] * len(coordinates_lr))
    secondRow.extend(["min-rot","max-rot","mean-rot","std-rot","","min-tr","max-tr","mean-tr","std-tr"])
    _ = csvWriter.writerow(secondRow)
    for idxMotion, motion in enumerate(all_motions):
        c_bp = medians_RMSEs[motion]
        for idxSetup, setup in enumerate(setups):
            if idxSetup == 0:
                RMSErow = [motion, '', setup]
            else:
                RMSErow = ['', '', setup]
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]
                RMSErow.extend(['%.2f' %c_coord[idxSetup], ''])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1                    
            # Add min, max, mean, std
            RMSErow.extend(['%.2f' %np.round(np.min(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.max(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.mean(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.std(temp_med_rot),1), ''])
            RMSErow.extend(['%.2f' %np.round(np.min(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.max(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.mean(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.std(temp_med_tr*1000),1)]) 
            _ = csvWriter.writerow(RMSErow)
            
# %% Print out csv files with results: mean RMSEs
suffixRMSE = ''
if fixed_markers:
    suffixRMSE = '_fixed'    
        
with open(os.path.join(outputDir,'RMSEs{}_means.csv'.format(suffixRMSE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['motion-type', '', 'setup']
    for label in coordinates_lr:
        
        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label            
        
        topRow.extend([label_adj,''])
    _ = csvWriter.writerow(topRow)
    secondRow = ['', '', '']
    secondRow.extend(["mean-RMSE",''] * len(coordinates_lr))
    secondRow.extend(["min-rot","max-rot","mean-rot","std-rot","","min-tr","max-tr","mean-tr","std-tr"])
    _ = csvWriter.writerow(secondRow)
    means_RMSE_summary, mins_RMSE_summary, maxs_RMSE_summary = {}, {}, {}
    mins_RMSE_summary_labels, maxs_RMSE_summary_labels = {}, {}
    for idxMotion, motion in enumerate(all_motions):
        means_RMSE_summary[motion], mins_RMSE_summary[motion], maxs_RMSE_summary[motion] = {}, {}, {}
        mins_RMSE_summary_labels[motion], maxs_RMSE_summary_labels[motion] = {}, {}
        c_bp = means_RMSEs[motion]
        for idxSetup, setup in enumerate(setups):
            means_RMSE_summary[motion][setup] = {}
            mins_RMSE_summary[motion][setup] = {}
            maxs_RMSE_summary[motion][setup] = {}
            mins_RMSE_summary_labels[motion][setup] = {}
            maxs_RMSE_summary_labels[motion][setup] = {}
            if idxSetup == 0:
                RMSErow = [motion, '', setup]
            else:
                RMSErow = ['', '', setup]
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            coordinates_temp_rot = []
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]              
                RMSErow.extend(['%.2f' %c_coord[idxSetup], ''])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                        coordinates_temp_rot.append(coordinate)
                        coordinates_temp_rot.append(coordinate)
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                        coordinates_temp_rot.append(coordinate)
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1      
            # Add min, max, mean
            RMSErow.extend(['%.2f' %np.round(np.min(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.max(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.mean(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.std(temp_med_rot),1), ''])
            RMSErow.extend(['%.2f' %np.round(np.min(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.max(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.mean(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.std(temp_med_tr*1000),1)]) 
            _ = csvWriter.writerow(RMSErow)
            means_RMSE_summary[motion][setup]['rotation'] = np.round(np.mean(temp_med_rot),1)
            means_RMSE_summary[motion][setup]['translation'] = np.round(np.mean(temp_med_tr*1000),1)
            
            mins_RMSE_summary[motion][setup]['rotation'] = np.round(np.min(temp_med_rot),1)
            mins_RMSE_summary[motion][setup]['translation'] = np.round(np.min(temp_med_tr*1000),1)
            mins_RMSE_summary_labels[motion][setup]['rotation'] = coordinates_temp_rot[np.argmin(temp_med_rot)]
            mins_RMSE_summary_labels[motion][setup]['translation'] = coordinates_lr_tr[np.argmin(temp_med_tr)]
            
            maxs_RMSE_summary[motion][setup]['rotation'] = np.round(np.max(temp_med_rot),1)
            maxs_RMSE_summary[motion][setup]['translation'] = np.round(np.max(temp_med_tr*1000),1)
            maxs_RMSE_summary_labels[motion][setup]['rotation'] = coordinates_temp_rot[np.argmax(temp_med_rot)]
            maxs_RMSE_summary_labels[motion][setup]['translation'] = coordinates_lr_tr[np.argmax(temp_med_tr)]

# %% Plots per coordinate: MAEs
all_motions = ['all'] + motions
bps, means_MAEs, medians_MAEs = {}, {}, {}
for motion in all_motions:
    bps[motion], means_MAEs[motion], medians_MAEs[motion] = {}, {}, {}
    if plots:
        fig, axs = plt.subplots(5, 3, sharex=True)    
        fig.suptitle(motion)    
    for count, coordinate in enumerate(coordinates_lr):
        c_data = {}
        for processingType in processingTypes:
            for augmenterType in augmenterTypes:
                for poseDetector in poseDetectors:
                    for cameraSetup in cameraSetups:                
                        if coordinate[-2:] == '_l':
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType] = (                    
                                MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist() +                     
                                MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate[:-2] + '_r'].tolist())
                            coordinate_title = coordinate[:-2]
                        else:
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType] = (
                                MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist())
                            coordinate_title = coordinate                    
        if plots:
            ax = axs.flat[count]
            bps[motion][coordinate] = ax.boxplot(c_data.values())        
            ax.set_title(coordinate_title)
            xticks = list(range(1, len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)*len(processingTypes)+1))
            ax.set_xticks(xticks)
            ax.set_xticklabels(c_data.keys(), rotation = 90)
            ax.set_ylim(0, 20)
            ax.axhline(y=5, color='r', linestyle='--')
            ax.axhline(y=10, color='b', linestyle='--')        
        means_MAEs[motion][coordinate] = [np.mean(c_data[a]) for a in c_data]  
        medians_MAEs[motion][coordinate] = [np.median(c_data[a]) for a in c_data]  
        
# %% Print out csv files with results: median MAEs
suffixMAE = ''
if fixed_markers:
    suffixMAE = '_fixed'    
        
with open(os.path.join(outputDir,'MAEs{}_medians.csv'.format(suffixMAE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['motion-type', '', 'setup']
    for label in coordinates_lr:
        
        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label            
        
        topRow.extend([label_adj,''])
    _ = csvWriter.writerow(topRow)
    secondRow = ['', '', '']
    secondRow.extend(["median-MAE",''] * len(coordinates_lr))
    secondRow.extend(["min-rot","max-rot","mean-rot","std-rot","","min-tr","max-tr","mmean-tr","std-tr"])
    _ = csvWriter.writerow(secondRow)
    for idxMotion, motion in enumerate(all_motions):
        c_bp = medians_MAEs[motion]
        for idxSetup, setup in enumerate(setups):
            if idxSetup == 0:
                MAErow = [motion, '', setup]
            else:
                MAErow = ['', '', setup]   
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]
                MAErow.extend(['%.2f' %c_coord[idxSetup], ''])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1                       
            # Add min, max, mean
            MAErow.extend(['%.2f' %np.round(np.min(temp_med_rot),1)])
            MAErow.extend(['%.2f' %np.round(np.max(temp_med_rot),1)])
            MAErow.extend(['%.2f' %np.round(np.mean(temp_med_rot),1)])
            MAErow.extend(['%.2f' %np.round(np.std(temp_med_rot),1), ''])
            MAErow.extend(['%.2f' %np.round(np.min(temp_med_tr*1000),1)])
            MAErow.extend(['%.2f' %np.round(np.max(temp_med_tr*1000),1)])
            MAErow.extend(['%.2f' %np.round(np.mean(temp_med_tr*1000),1)])
            MAErow.extend(['%.2f' %np.round(np.std(temp_med_tr*1000),1)]) 
            _ = csvWriter.writerow(MAErow)
            
# %% Print out csv files with results: mean MAEs
suffixRMSE = ''
if fixed_markers:
    suffixRMSE = '_fixed'    
        
with open(os.path.join(outputDir,'MAEs{}_means.csv'.format(suffixRMSE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['motion-type', '', 'setup']
    for label in coordinates_lr:
        
        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label            
        
        topRow.extend([label_adj,''])
    _ = csvWriter.writerow(topRow)
    secondRow = ['', '', '']
    secondRow.extend(["mean-MAE",''] * len(coordinates_lr))
    secondRow.extend(["min-rot","max-rot","mean-rot","std-rot","","min-tr","max-tr","mean-tr","std-tr"])
    _ = csvWriter.writerow(secondRow)
    
    means_summary, mins_summary, maxs_summary = {}, {}, {}
    # all_summary = np.zeros((len(all_motions)*len(setups),len(coordinates_lr_rot)))
    # c_all = 0
    for idxMotion, motion in enumerate(all_motions):
        means_summary[motion], mins_summary[motion], maxs_summary[motion] = {}, {}, {}
        c_bp = means_MAEs[motion]
        for idxSetup, setup in enumerate(setups):
            means_summary[motion][setup] = {}
            mins_summary[motion][setup] = {}
            maxs_summary[motion][setup] = {}
            if idxSetup == 0:
                MAErow = [motion, '', setup]
            else:
                MAErow = ['', '', setup]
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]              
                MAErow.extend(['%.2f' %c_coord[idxSetup], ''])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1                    
            # Add min, max, mean
            MAErow.extend(['%.2f' %np.round(np.min(temp_med_rot),1)])
            MAErow.extend(['%.2f' %np.round(np.max(temp_med_rot),1)])
            MAErow.extend(['%.2f' %np.round(np.mean(temp_med_rot),1)])
            MAErow.extend(['%.2f' %np.round(np.std(temp_med_rot),1), ''])
            MAErow.extend(['%.2f' %np.round(np.min(temp_med_tr*1000),1)])
            MAErow.extend(['%.2f' %np.round(np.max(temp_med_tr*1000),1)])
            MAErow.extend(['%.2f' %np.round(np.mean(temp_med_tr*1000),1)])
            MAErow.extend(['%.2f' %np.round(np.std(temp_med_tr*1000),1)]) 
            _ = csvWriter.writerow(MAErow)
            means_summary[motion][setup]['rotation'] = np.round(np.mean(temp_med_rot),1)
            means_summary[motion][setup]['translation'] = np.round(np.mean(temp_med_tr*1000),1)
            
            mins_summary[motion][setup]['rotation'] = np.round(np.min(temp_med_rot),1)
            mins_summary[motion][setup]['translation'] = np.round(np.min(temp_med_tr*1000),1)
            
            maxs_summary[motion][setup]['rotation'] = np.round(np.max(temp_med_rot),1)
            maxs_summary[motion][setup]['translation'] = np.round(np.max(temp_med_tr*1000),1)
            
            # all_summary[c_all, :] = temp_med_rot
            # c_all += 1

# %% Plots per coordinate: MEs
all_motions = ['all'] + motions
bps, means_MEs, medians_MEs = {}, {}, {}
for motion in all_motions:
    bps[motion], means_MEs[motion], medians_MEs[motion] = {}, {}, {}
    if plots:
        fig, axs = plt.subplots(5, 3, sharex=True)    
        fig.suptitle(motion)    
    for count, coordinate in enumerate(coordinates_lr):
        c_data = {}
        for processingType in processingTypes:
            for augmenterType in augmenterTypes:
                for poseDetector in poseDetectors:
                    for cameraSetup in cameraSetups:                
                        if coordinate[-2:] == '_l':
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType] = (                    
                                MEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist() +                     
                                MEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate[:-2] + '_r'].tolist())
                            coordinate_title = coordinate[:-2]
                        else:
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType+ '_' + processingType] = (
                                MEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist())
                            coordinate_title = coordinate                    
        if plots:
            ax = axs.flat[count]
            bps[motion][coordinate] = ax.boxplot(c_data.values())
            ax.set_title(coordinate_title)
            xticks = list(range(1, len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)*len(processingTypes)+1))
            ax.set_xticks(xticks)
            ax.set_xticklabels(c_data.keys(), rotation = 90)
            ax.set_ylim(0, 20)
            ax.axhline(y=5, color='r', linestyle='--')
            ax.axhline(y=10, color='b', linestyle='--')
        
        means_MEs[motion][coordinate] = [np.mean(c_data[a]) for a in c_data]
        medians_MEs[motion][coordinate] = [np.median(c_data[a]) for a in c_data]

        # # Compute stastical differences between entries of c_data
        # # Compute p-values for normality test
        # p_values = []
        # methods = []
        # for a in c_data:
        #     _, p_value = stats.shapiro(c_data[a])
        #     p_values.append(p_value)
        #     methods.append(a)
        # # Get indices of pairs of method
        # from itertools import combinations
        # idx_pairs = list(combinations(range(len(c_data)), 2))
        # alpha = 0.01  # significance level
        # for idx_pair in idx_pairs:
        #     if p_values[idx_pair[0]] > alpha and p_values[idx_pair[1]] > alpha:
        #         # print("The data for both methods appears to be normally distributed.")
        #         # Perform t-test
        #         _, p_value = stats.ttest_ind(c_data[methods[idx_pair[0]]], c_data[methods[idx_pair[1]]])
        #         if p_value < alpha:
        #             print("{} - {}: t-test, The difference between the two methods is statistically significant".format(motion, coordinate))
        #     else:
        #         # print("The data for one or both methods does not appear to be normally distributed.")
        #         # Perform Mann-Whitney U test
        #         _, p_value = stats.mannwhitneyu(c_data[methods[idx_pair[0]]], c_data[methods[idx_pair[1]]])
        #         if p_value < alpha:
        #             print("{} - {}: Mann-Whitney U test, The difference between the two methods is statistically significant".format(motion, coordinate))
            
# %% Print out csv files with results: mean MAEs and RMSEs - table formatted for paper
activity_names = {'walking': 'Walking',
                  'DJ': 'Drop jump',
                  'squats': 'Squatting',
                  'STS': 'Sit-to-stand'}

def getPoseName(setup):    
    if 'mmpose' in setup:
        poseName = 'HRNet'
    elif 'openpose' in setup.lower():
        if 'generic' in setup.lower():
            poseName = 'Low resolution OpenPose'
        else:
            poseName = 'High resolution OpenPose'
    else:
        raise ValueError('Pose detector not recognized')
    return poseName

def getCameraConfig(setup):    
    if '2-cameras' in setup.lower():
        configName = '2-cameras'
    elif '3-cameras' in setup.lower():
        configName = '3-cameras'
    elif '5-cameras' in setup.lower():
        configName = '5-cameras'        
    else:
        raise ValueError('Camera config not recognized')       
    return configName

def getPoseConfig(setup): 
    if '_pose' in setup.lower():
        poseName = 'Video keypoints'
    else:
        poseName = 'Anatomical markers'
    return poseName

suffixRMSE = ''
if fixed_markers:
    suffixRMSE = '_fixed'    

# MAEs
with open(os.path.join(outputDir,'MAEs{}_means_paper.csv'.format(suffixRMSE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['Activity', 'Markers', 'Pose detector', 'Camera configuration']
    for label in coordinates_lr:
        
        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label            
        
        topRow.extend([label_adj])
    topRow.extend(["","min rotations","max rotations","mean rotations","std rotations","","min translations","max translations","mean translations","std translations"])        
    _ = csvWriter.writerow(topRow)
    for idxMotion, motion in enumerate(all_motions):
        if 'all' in motion:
            continue        
        c_bp = means_MAEs[motion]
        for idxSetup, setup in enumerate(setups):            
            activity_name = activity_names[motion]
            pose_name = getPoseName(setup)
            config_name = getCameraConfig(setup)
            marker_name = getPoseConfig(setup)            
            if idxSetup == 0:
                MAErow = [activity_name, marker_name, pose_name, config_name]
            else:
                MAErow = ['', marker_name, pose_name, config_name]
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]
                if coordinate in coordinates_lr_tr:
                    MAErow.extend(['%.1f' %(c_coord[idxSetup]*1000)])
                else:
                    MAErow.extend(['%.1f' %c_coord[idxSetup]])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1                    
            # Add min, max, mean
            MAErow.extend(['','%.1f' %np.round(np.min(temp_med_rot),1)])
            MAErow.extend(['%.1f' %np.round(np.max(temp_med_rot),1)])
            MAErow.extend(['%.1f' %np.round(np.mean(temp_med_rot),1)])
            MAErow.extend(['%.1f' %np.round(np.std(temp_med_rot),1), ''])
            MAErow.extend(['%.1f' %np.round(np.min(temp_med_tr*1000),1)])
            MAErow.extend(['%.1f' %np.round(np.max(temp_med_tr*1000),1)])
            MAErow.extend(['%.1f' %np.round(np.mean(temp_med_tr*1000),1)])
            MAErow.extend(['%.1f' %np.round(np.std(temp_med_tr*1000),1)]) 
            _ = csvWriter.writerow(MAErow)
   
# RMSEs
with open(os.path.join(outputDir,'RMSEs{}_means_paper.csv'.format(suffixRMSE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['Activity', 'Markers', 'Pose detector', 'Camera configuration']
    for label in coordinates_lr:
        
        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label            
        
        topRow.extend([label_adj])
    topRow.extend(["","min rotations","max rotations","mean rotations","std rotations","","min translations","max translations","mean translations","std translations"])        
    _ = csvWriter.writerow(topRow)
    for idxMotion, motion in enumerate(all_motions):
        if 'all' in motion:
            continue        
        c_bp = means_RMSEs[motion]
        c_bp_std = stds_RMSEs[motion]
        for idxSetup, setup in enumerate(setups):            
            activity_name = activity_names[motion]
            pose_name = getPoseName(setup)
            config_name = getCameraConfig(setup)
            marker_name = getPoseConfig(setup)            
            if idxSetup == 0:
                MAErow = [activity_name, marker_name, pose_name, config_name]
            else:
                MAErow = ['', marker_name, pose_name, config_name]
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]
                c_coord_std = c_bp_std[coordinate]
                if coordinate in coordinates_lr_tr:                    
                    value1 = '%.1f' % (c_coord[idxSetup] * 1000)
                    value2 = '%.1f' % (c_coord_std[idxSetup] * 1000)
                    result_string = f"{value1} +/- {value2}"                    
                    MAErow.extend([result_string])
                else:
                    value1 = '%.1f' % (c_coord[idxSetup])
                    value2 = '%.1f' % (c_coord_std[idxSetup])
                    result_string = f"{value1} +/- {value2}"                    
                    MAErow.extend([result_string])
                    # MAErow.extend(['%.1f' %c_coord[idxSetup]])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1                    
            # Add min, max, mean
            MAErow.extend(['','%.1f' %np.round(np.min(temp_med_rot),1)])
            MAErow.extend(['%.1f' %np.round(np.max(temp_med_rot),1)])
            MAErow.extend(['%.1f' %np.round(np.mean(temp_med_rot),1)])
            MAErow.extend(['%.1f' %np.round(np.std(temp_med_rot),1), ''])
            MAErow.extend(['%.1f' %np.round(np.min(temp_med_tr*1000),1)])
            MAErow.extend(['%.1f' %np.round(np.max(temp_med_tr*1000),1)])
            MAErow.extend(['%.1f' %np.round(np.mean(temp_med_tr*1000),1)])
            MAErow.extend(['%.1f' %np.round(np.std(temp_med_tr*1000),1)]) 
            _ = csvWriter.writerow(MAErow)

# %% Means across activities
# MAEs
means_summary_rot, means_summary_tr = {}, {}
mins_summary_rot, mins_summary_tr = {}, {}
maxs_summary_rot, maxs_summary_tr = {}, {}
for setup in setups:
    means_summary_rot[setup], means_summary_tr[setup] = [], []
    mins_summary_rot[setup], mins_summary_tr[setup] = [], []
    maxs_summary_rot[setup], maxs_summary_tr[setup] = [], []
    for motion in motions:
        means_summary_rot[setup].append(means_summary[motion][setup]['rotation'])
        means_summary_tr[setup].append(means_summary[motion][setup]['translation'])
        mins_summary_rot[setup].append(mins_summary[motion][setup]['rotation'])
        mins_summary_tr[setup].append(mins_summary[motion][setup]['translation'])
        maxs_summary_rot[setup].append(maxs_summary[motion][setup]['rotation'])
        maxs_summary_tr[setup].append(maxs_summary[motion][setup]['translation'])
        
# RMSEs
means_RMSE_summary_rot, means_RMSE_summary_tr = {}, {}
mins_RMSE_summary_rot, mins_RMSE_summary_tr = {}, {}
maxs_RMSE_summary_rot, maxs_RMSE_summary_tr = {}, {}
mins_RMSE_summary_label_rot, mins_RMSE_summary_label_tr = {}, {}
maxs_RMSE_summary_label_rot, maxs_RMSE_summary_label_tr = {}, {}

for setup in setups:
    means_RMSE_summary_rot[setup], means_RMSE_summary_tr[setup] = [], []
    mins_RMSE_summary_rot[setup], mins_RMSE_summary_tr[setup] = [], []
    maxs_RMSE_summary_rot[setup], maxs_RMSE_summary_tr[setup] = [], []
    mins_RMSE_summary_label_rot[setup], mins_RMSE_summary_label_tr[setup] = [], []
    maxs_RMSE_summary_label_rot[setup], maxs_RMSE_summary_label_tr[setup] = [], []
    for motion in motions:
        means_RMSE_summary_rot[setup].append(means_RMSE_summary[motion][setup]['rotation'])
        means_RMSE_summary_tr[setup].append(means_RMSE_summary[motion][setup]['translation'])
        mins_RMSE_summary_rot[setup].append(mins_RMSE_summary[motion][setup]['rotation'])
        mins_RMSE_summary_tr[setup].append(mins_RMSE_summary[motion][setup]['translation'])
        maxs_RMSE_summary_rot[setup].append(maxs_RMSE_summary[motion][setup]['rotation'])
        maxs_RMSE_summary_tr[setup].append(maxs_RMSE_summary[motion][setup]['translation'])
        
        mins_RMSE_summary_label_rot[setup].append(mins_RMSE_summary_labels[motion][setup]['rotation'])
        mins_RMSE_summary_label_tr[setup].append(mins_RMSE_summary_labels[motion][setup]['translation'])
        maxs_RMSE_summary_label_rot[setup].append(maxs_RMSE_summary_labels[motion][setup]['rotation'])
        maxs_RMSE_summary_label_tr[setup].append(maxs_RMSE_summary_labels[motion][setup]['translation'])
    
print("Rotations - MAEs")           
c_mean_rot_all = np.zeros((len(setups),))
c_std_rot_all = np.zeros((len(setups),))
c_min_rot_all = np.zeros((len(setups),))
c_max_rot_all = np.zeros((len(setups),))
for c_s, setup in enumerate(setups):
    c_mean_rot_all[c_s] = np.round(np.mean(np.asarray( means_summary_rot[setup])),1)
    c_std_rot_all[c_s] = np.round(np.std(np.asarray( means_summary_rot[setup])),1)  
    c_min_rot_all[c_s] = np.round(np.min(np.asarray( mins_summary_rot[setup])),1)
    c_max_rot_all[c_s] = np.round(np.max(np.asarray( maxs_summary_rot[setup])),1)
    print("{}: {} +/- {} [{} {}]".format(setup, c_mean_rot_all[c_s], c_std_rot_all[c_s], c_min_rot_all[c_s], c_max_rot_all[c_s]))
# c_mean_rot_diff = c_mean_rot_all[9:] - c_mean_rot_all[:9] 
# print('Max decrease with no augmenter - rotation - mmpose: {}'.format(np.round(np.max(c_mean_rot_diff[:3]),1)))
# print('Max decrease with no augmenter - rotation - openpose high res: {}'.format(np.round(np.max(c_mean_rot_diff[3:6]),1)))
# print('Max decrease with no augmenter - rotation - openpose low res: {}'.format(np.round(np.max(c_mean_rot_diff[6:]),1)))
# print('Max decrease with no augmenter - rotation - all: {}'.format(np.round(np.max(c_mean_rot_diff),1)))
# print('Mean decrease with no augmenter - rotation - mmpose: {}'.format(np.round(np.mean(c_mean_rot_diff[:3]),1)))
# print('Mean decrease with no augmenter - rotation - openpose high res: {}'.format(np.round(np.mean(c_mean_rot_diff[3:6]),1)))
# print('Mean decrease with no augmenter - rotation - openpose low res: {}'.format(np.round(np.mean(c_mean_rot_diff[6:]),1)))
# print('Mean decrease with no augmenter - rotation - all: {}'.format(np.round(np.mean(c_mean_rot_diff),1)))


print("")
print("Translations - MAEs")
c_mean_tr_all = np.zeros((len(setups),))
c_std_tr_all = np.zeros((len(setups),))
c_min_tr_all = np.zeros((len(setups),))
c_max_tr_all = np.zeros((len(setups),))
for c_s, setup in enumerate(setups):
    c_mean_tr_all[c_s] = np.round(np.mean(np.asarray( means_summary_tr[setup])),1)
    c_std_tr_all[c_s] = np.round(np.std(np.asarray( means_summary_tr[setup])),1)
    c_min_tr_all[c_s] = np.round(np.min(np.asarray( mins_summary_tr[setup])),1)
    c_max_tr_all[c_s] = np.round(np.max(np.asarray( maxs_summary_tr[setup])),1)
    print("{}: {} +/- {} [{} {}]".format(setup, c_mean_tr_all[c_s], c_std_tr_all[c_s], c_min_tr_all[c_s], c_max_tr_all[c_s]))
# c_mean_tr_diff = c_mean_tr_all[9:] - c_mean_tr_all[:9]
# print('Max decrease with no augmenter - translation - mmpose: {}'.format(np.round(np.max(c_mean_tr_diff[:3]),1)))
# print('Max decrease with no augmenter - translation - openpose high res: {}'.format(np.round(np.max(c_mean_tr_diff[3:6]),1)))
# print('Max decrease with no augmenter - translation - openpose low res: {}'.format(np.round(np.max(c_mean_tr_diff[6:]),1)))
# print('Max decrease with no augmenter - translation - all: {}'.format(np.round(np.max(c_mean_tr_diff),1)))
# print('Mean decrease with no augmenter - translation - mmpose: {}'.format(np.round(np.mean(c_mean_tr_diff[:3]),1)))
# print('Mean decrease with no augmenter - translation - openpose high res: {}'.format(np.round(np.mean(c_mean_tr_diff[3:6]),1)))
# print('Mean decrease with no augmenter - translation - openpose low res: {}'.format(np.round(np.mean(c_mean_tr_diff[6:]),1)))
# print('Mean decrease with no augmenter - translation - all: {}'.format(np.round(np.mean(c_mean_tr_diff),1)))

# RMSEs
print("")
print("Rotations - RMSEs")           
c_mean_rot_all_RMSE = np.zeros((len(setups),))
c_std_rot_all_RMSE = np.zeros((len(setups),))
c_min_rot_all_RMSE = np.zeros((len(setups),))
c_max_rot_all_RMSE = np.zeros((len(setups),))
c_min_rot_all_RMSE_lables, c_max_rot_all_RMSE_lables = [], []
for c_s, setup in enumerate(setups):
    c_mean_rot_all_RMSE[c_s] = np.round(np.mean(np.asarray( means_RMSE_summary_rot[setup])),1)
    c_std_rot_all_RMSE[c_s] = np.round(np.std(np.asarray( means_RMSE_summary_rot[setup])),1)  
    c_min_rot_all_RMSE[c_s] = np.round(np.min(np.asarray( mins_RMSE_summary_rot[setup])),1)
    c_max_rot_all_RMSE[c_s] = np.round(np.max(np.asarray( maxs_RMSE_summary_rot[setup])),1)    
    c_min_label = mins_RMSE_summary_label_rot[setup][np.argmin(np.asarray( mins_RMSE_summary_rot[setup]))]
    c_max_label = maxs_RMSE_summary_label_rot[setup][np.argmax(np.asarray( maxs_RMSE_summary_rot[setup]))]
    c_min_rot_all_RMSE_lables.append(c_min_label)
    c_max_rot_all_RMSE_lables.append(c_max_label)
    
    print("{}: {} +/- {} [{} {}] [{} {}]".format(setup, c_mean_rot_all_RMSE[c_s], c_std_rot_all_RMSE[c_s], c_min_rot_all_RMSE[c_s], c_max_rot_all_RMSE[c_s], c_min_label, c_max_label))
# c_mean_rot_diff_RMSE = c_mean_rot_all_RMSE[9:] - c_mean_rot_all_RMSE[:9] 
# print('Max decrease with no augmenter - rotation - mmpose: {}'.format(np.round(np.max(c_mean_rot_diff_RMSE[:3]),1)))
# print('Max decrease with no augmenter - rotation - openpose high res: {}'.format(np.round(np.max(c_mean_rot_diff_RMSE[3:6]),1)))
# print('Max decrease with no augmenter - rotation - openpose low res: {}'.format(np.round(np.max(c_mean_rot_diff_RMSE[6:]),1)))
# print('Max decrease with no augmenter - rotation - all: {}'.format(np.round(np.max(c_mean_rot_diff_RMSE),1)))
# print('Mean decrease with no augmenter - rotation - mmpose: {}'.format(np.round(np.mean(c_mean_rot_diff_RMSE[:3]),1)))
# print('Mean decrease with no augmenter - rotation - openpose high res: {}'.format(np.round(np.mean(c_mean_rot_diff_RMSE[3:6]),1)))
# print('Mean decrease with no augmenter - rotation - openpose low res: {}'.format(np.round(np.mean(c_mean_rot_diff_RMSE[6:]),1)))
# print('Mean decrease with no augmenter - rotation - all: {}'.format(np.round(np.mean(c_mean_rot_diff_RMSE),1)))


print("")
print("Translations - RMSEs")
c_mean_tr_all_RMSE = np.zeros((len(setups),))
c_std_tr_all_RMSE = np.zeros((len(setups),))
c_min_tr_all_RMSE = np.zeros((len(setups),))
c_max_tr_all_RMSE = np.zeros((len(setups),))
c_min_tr_all_RMSE_lables, c_max_tr_all_RMSE_lables = [], []
for c_s, setup in enumerate(setups):
    c_mean_tr_all_RMSE[c_s] = np.round(np.mean(np.asarray( means_RMSE_summary_tr[setup])),1)
    c_std_tr_all_RMSE[c_s] = np.round(np.std(np.asarray( means_RMSE_summary_tr[setup])),1)
    c_min_tr_all_RMSE[c_s] = np.round(np.min(np.asarray( mins_RMSE_summary_tr[setup])),1)
    c_max_tr_all_RMSE[c_s] = np.round(np.max(np.asarray( maxs_RMSE_summary_tr[setup])),1)
    c_min_label = mins_RMSE_summary_label_tr[setup][np.argmin(np.asarray( mins_RMSE_summary_tr[setup]))]
    c_max_label = maxs_RMSE_summary_label_tr[setup][np.argmax(np.asarray( maxs_RMSE_summary_tr[setup]))]
    c_min_tr_all_RMSE_lables.append(c_min_label)
    c_max_tr_all_RMSE_lables.append(c_max_label)
    print("{}: {} +/- {} [{} {}] [{} {}]".format(setup, c_mean_tr_all_RMSE[c_s], c_std_tr_all_RMSE[c_s], c_min_tr_all_RMSE[c_s], c_max_tr_all_RMSE[c_s], c_min_label, c_max_label))
# c_mean_tr_diff_RMSE = c_mean_tr_all_RMSE[9:] - c_mean_tr_all_RMSE[:9]
# print('Max decrease with no augmenter - translation - mmpose: {}'.format(np.round(np.max(c_mean_tr_diff_RMSE[:3]),1)))
# print('Max decrease with no augmenter - translation - openpose high res: {}'.format(np.round(np.max(c_mean_tr_diff_RMSE[3:6]),1)))
# print('Max decrease with no augmenter - translation - openpose low res: {}'.format(np.round(np.max(c_mean_tr_diff_RMSE[6:]),1)))
# print('Max decrease with no augmenter - translation - all: {}'.format(np.round(np.max(c_mean_tr_diff_RMSE),1)))
# print('Mean decrease with no augmenter - translation - mmpose: {}'.format(np.round(np.mean(c_mean_tr_diff_RMSE[:3]),1)))
# print('Mean decrease with no augmenter - translation - openpose high res: {}'.format(np.round(np.mean(c_mean_tr_diff_RMSE[3:6]),1)))
# print('Mean decrease with no augmenter - translation - openpose low res: {}'.format(np.round(np.mean(c_mean_tr_diff_RMSE[6:]),1)))
# print('Mean decrease with no augmenter - translation - all: {}'.format(np.round(np.mean(c_mean_tr_diff_RMSE),1)))

# # %% Benchmark
# # TODO need to make more general
# with open(os.path.join(outputDir,'MAEs{}_benchmark_means{}.csv'.format(suffixRMSE,suffix_files)), 'w', newline='') as csvfile:
#     csvWriter = csv.writer(csvfile)
#     topRow = ['Setup', '', 'Rot - ref', 'Rot - new', 'Rot-diff', 'Tr - ref', 'Tr - new', 'Tr - diff']
#     _ = csvWriter.writerow(topRow)
#     for cs, setup in enumerate(setups):
#         if cs >= len(setups)/2:
#             continue
#         MAErow = [setup, '', '%.1f' %c_mean_rot_all[cs], '%.1f' %c_mean_rot_all[cs+9], '%.1f' %c_mean_rot_diff[cs], 
#                   '%.1f' %c_mean_tr_all[cs], '%.1f' %c_mean_tr_all[cs+9], '%.1f' %c_mean_tr_diff[cs]]
#         _ = csvWriter.writerow(MAErow) 

# # %% Classify coordinates
# # idx_sort = np.argsort(all_summary, axis=1)
# # coor_sort = []
# # for c in range(idx_sort.shape[0]):
# #     temp_list = []
# #     for cc in range(len(coordinates_lr_rot)):
# #         temp_list.append(coordinates_lr_rot[idx_sort[c, cc]])
# #     coor_sort.append(temp_list)
    
# # %% Detailed effect of augmenter
# def getSummary(idx_setup_sel, activity_names, coordinates_lr, means_MAEs):
#     summary_sel = np.zeros((len(activity_names), len(coordinates_lr)))
#     for c, coordinate in enumerate(coordinates_lr):
#         for a, activity in enumerate(activity_names):    
#             summary_sel[a,c] = means_MAEs[activity][coordinate][idx_setup_sel]
            
#     return summary_sel

# setup_sel = 'mmpose_0.8_2-cameras_pose'
# idx_setup_sel = setups.index(setup_sel)
# summary_sel_pose = getSummary(idx_setup_sel, activity_names, coordinates_lr, means_MAEs)

# setup_sel = 'mmpose_0.8_2-cameras_separateLowerUpperBody_OpenPose'
# idx_setup_sel = setups.index(setup_sel)
# summary_sel_augmenter = getSummary(idx_setup_sel, activity_names, coordinates_lr, means_MAEs)

# summary_sel_diff = summary_sel_pose - summary_sel_augmenter
# bad_coordinates = ['pelvis_tilt', 'hip_flexion_l', 'lumbar_extension']
# idx_bad_coordinates = [coordinates_lr.index(bad_coordinate) for bad_coordinate in bad_coordinates]

# # Difference between pose and augmenter.
# bad_values_diff = np.zeros((len(bad_coordinates)*summary_sel_diff.shape[0],))
# count = 0
# for i in range(summary_sel_diff.shape[0]):
#     for j in idx_bad_coordinates:
#         bad_values_diff[count,] = summary_sel_diff[i, j]
#         count += 1
# range_bad_values_diff = [np.round(np.min(bad_values_diff),1), np.round(np.max(bad_values_diff),1)]

# # Pose errors.
# bad_values_pose = np.zeros((len(bad_coordinates)*summary_sel_diff.shape[0],))
# count = 0
# for i in range(summary_sel_diff.shape[0]):
#     for j in idx_bad_coordinates:
#         bad_values_pose[count,] = summary_sel_pose[i, j]
#         count += 1
# range_bad_values_pose = [np.round(np.min(bad_values_pose),1), np.round(np.max(bad_values_pose),1)]


# Plots
# means_RMSEs
# setups

# %% RMSEs
# Get len(cameraSetups) color-blind frienly colors.
colors = sns.color_palette('colorblind', len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)*len(processingTypes))

# Copy means_RMSEs to means_RMSEs_copy.
means_RMSEs_copy = copy.deepcopy(means_RMSEs)
stds_RMSEs_copy = copy.deepcopy(stds_RMSEs)
# We do not want the mean across all trials, but rather the across the mean of
# each motion type. Remove field 'all' from means_RMSEs_copy
means_RMSEs_copy.pop('all')
stds_RMSEs_copy.pop('all')
motions = list(means_RMSEs_copy.keys())
# Create a new field 'mean' in means_RMSEs_copy
means_RMSEs_copy['mean'], stds_RMSEs_copy['mean'] = {}, {}
# Compute mean across all motions for each coordinate.
for coordinate in means_RMSEs_copy[motions[0]]:
    means_RMSEs_copy['mean'][coordinate], stds_RMSEs_copy['mean'][coordinate] = [], []
    for i in range(len(means_RMSEs_copy[motions[0]][coordinate])):
        means_RMSEs_copy['mean'][coordinate].append(np.mean([means_RMSEs_copy[motion][coordinate][i] for motion in motions], axis=0))
        stds_RMSEs_copy['mean'][coordinate].append(np.std([means_RMSEs_copy[motion][coordinate][i] for motion in motions], axis=0))
motions.append('mean')

# Exclude coordinates_tr from means_RMSEs_copy
means_RMSEs_copy_tr = copy.deepcopy(means_RMSEs_copy)
stds_RMSEs_copy_tr = copy.deepcopy(stds_RMSEs_copy)
for motion in means_RMSEs_copy:
    for coordinate in coordinates_tr:
        means_RMSEs_copy[motion].pop(coordinate)
        stds_RMSEs_copy[motion].pop(coordinate)
for motion in means_RMSEs_copy_tr:
    for coordinate in coordinates_lr_rot:
        means_RMSEs_copy_tr[motion].pop(coordinate)
        stds_RMSEs_copy_tr[motion].pop(coordinate)
# Compute mean, this should match means_RMSE_summary_rot. There is still a mismatch
# between means_RMSE_summary_rot, where the mean is across the mean of each motion,
# whereas here the mean is across the mean coordinates. The means should match, but
# not the std.
for motion in means_RMSEs_copy:
    # Add field mean that contains the mean of the RMSEs for all coordinates.
    # Stack lists from all fiedls of means_RMSEs_copy[motion] in one numpy array.
    # Count twice the bilateral coordinates.
    c_stack = np.zeros((len(coordinates_lr_rot) + len(coordinates_bil), len(setups)))
    count = 0
    for i, coordinate in enumerate(coordinates_lr_rot):
        c_stack[count, :] = means_RMSEs_copy[motion][coordinate]
        count += 1
        if coordinate[-2:] == '_l':
            c_stack[count, :] = means_RMSEs_copy[motion][coordinate]
            count += 1
    means_RMSEs_copy[motion]['mean'] = list(np.mean(c_stack, axis=0))
    stds_RMSEs_copy[motion]['mean'] = list(np.std(c_stack, axis=0))
    
for motion in means_RMSEs_copy_tr:
    # Add field mean that contains the mean of the RMSEs for all coordinates.
    # Stack lists from all fiedls of means_RMSEs_copy[motion] in one numpy array.
    c_stack = np.zeros((len(coordinates_tr), len(setups)))
    count = 0
    for i, coordinate in enumerate(coordinates_tr):
        c_stack[count, :] = means_RMSEs_copy_tr[motion][coordinate]
        count += 1
    means_RMSEs_copy_tr[motion]['mean'] = list(np.mean(c_stack, axis=0))
    stds_RMSEs_copy_tr[motion]['mean'] = list(np.std(c_stack, axis=0))
    
# Create the x-tick labels for all subplots.
xtick_labels = list(means_RMSEs[motions[0]].keys())
# Remove pelvis_tx, pelvis_ty and pelivs_tz from xtick_labels.
xtick_labels = [xtick_label for xtick_label in xtick_labels if xtick_label not in coordinates_tr] + ['mean']
# remove _l at the end of the xtick_labels if present
xtick_labels_labels = [xtick_label[:-2] if xtick_label[-2:] == '_l' else xtick_label for xtick_label in xtick_labels]
# xtick_values = ['pelvis_tilt', 'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l', 'lumbar_extension', 'mean']
xtick_values = ['mean']

for cameraSetup in cameraSetups:
    fig, axs = plt.subplots(len(means_RMSEs_copy.keys()), 1, figsize=(10, 5))
    # Get indices of setups for the camera setup.
    idx_setups = [i for i, setup in enumerate(setups) if cameraSetup in setup]
    bar_width = 0.8/len(idx_setups)
    # Create list of integers with that has as many elements as there are idx_setups. The list has values with
    # a step of 1 and is centered on 0.
    x = np.arange(len(idx_setups)) - (len(idx_setups)-1)/2
    # Loop over subplots in axs.
    for a, ax in enumerate(axs):
        ax.set_title(motions[a], y=1.0, pad=-14, fontweight='bold')
        ax.set_ylabel('RMSE (deg)')
        ax.set_xticks(np.arange(len(xtick_labels)))
        if a == len(axs)-1:
            axs[a].set_xticklabels(xtick_labels)
        else:
            axs[a].set_xticklabels([])        
        # For each field in means_RMSEs['all'], plot bars with the values of the field for each idx_setups.
        for i, field in enumerate(xtick_labels):
            for j, idx_setup in enumerate(idx_setups):
                ax.bar(i+x[j]*bar_width, means_RMSEs_copy[motions[a]][field][idx_setup], bar_width, yerr=stds_RMSEs_copy[motions[a]][field][idx_setup], color=colors[j], ecolor='black', alpha=0.5, edgecolor='white', capsize=2)
                # Add text with the value of the bar.
                # if i == len(xtick_labels)-1:
                if field in xtick_values:
                    ax.text(i+x[j]*bar_width, means_RMSEs_copy[motions[a]][field][idx_setup], np.round(means_RMSEs_copy[motions[a]][field][idx_setup], 1), ha='center', va='bottom')    
        # Add legend with idx_setups
        # leg_t = [setups[idx_setup] for idx_setup in idx_setups]
        leg_t = [setups_t[idx_setup] for idx_setup in idx_setups]
        # Get what is after the last '_' in leg_t.
        # leg_t = [leg_t[i].split('_')[-1] for i in range(len(leg_t))]
        # ax.legend(leg_t)
        # Make legend horizontal and top left
        if a == 0:
            ax.legend(leg_t, loc='upper left', bbox_to_anchor=(0, 1.2), ncol=len(leg_t), frameon=False)
        plt.show()
        # Use same y-limits for all subplots.
        ax.set_ylim([0, 15])
        # Use 3 y-ticks [0, 5, 10]
        ax.set_yticks(np.arange(0, 20, 5))
        # Remove upper and right axes.
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    # Set the x-tick labels for the last subplot only.
    axs[-1].set_xticks(np.arange(len(xtick_labels)))
    axs[-1].set_xticklabels(xtick_labels_labels)
    # Align y-labels.
    fig.align_ylabels(axs)
    # Rotate x-tick labels.
    fig.autofmt_xdate(rotation=45)
    # plt.tight_layout()

    # %% Plots only means
    
    # Settings
    barWidth = 0.15
    fontsize_labels = 20
    fontsize_title = 20
    colors = sns.color_palette('colorblind', len(setups))
    num_bars = len(setups)
    positions = [np.arange(len(motions)) + i * barWidth for i in range(num_bars)]    
    
    # Rotations
    values = [[means_RMSEs_copy[motion]['mean'][i] for motion in motions] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy[motion]['mean'][i] for motion in motions] for i in range(num_bars)]    
    # Make the plot
    plt.figure(figsize=(10, 5))
    for i in range(num_bars):
        plt.bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', label=setups_t[i], align='center', alpha=0.5, ecolor='black', capsize=10)
    # Add xticks on the middle of the group bars
    plt.xticks(np.arange(len(motions)) + (barWidth * (num_bars - 1)) / 2, motions, fontweight='bold') 
    # Add ylabel
    plt.ylabel('Root Mean Squared Error (deg)', fontweight='bold', fontsize=fontsize_labels)
    # Increase fontsize of labels
    plt.tick_params(axis='both', which='major', labelsize=fontsize_labels)
    plt.ylim([0, 25])
    plt.yticks(np.arange(0, 30, 5))
    # Add title
    plt.title('Joint kinematic errors (mean +/- std; 18 degrees of freedom)', fontweight='bold', fontsize=fontsize_title)
    # Create legend
    plt.legend(loc='upper left', fontsize=fontsize_labels)
    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Remove box around legend
    plt.gca().get_legend().get_frame().set_linewidth(0.0)
    plt.tight_layout()
    plt.show()    
    
    # Translations in cm
    values = [[means_RMSEs_copy_tr[motion]['mean'][i]*100 for motion in motions] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy_tr[motion]['mean'][i]*100 for motion in motions] for i in range(num_bars)]    
    # Make the plot
    plt.figure(figsize=(10, 5))
    for i in range(num_bars):
        plt.bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', label=setups_t[i], align='center', alpha=0.5, ecolor='black', capsize=10)
    # Add xticks on the middle of the group bars
    plt.xticks(np.arange(len(motions)) + (barWidth * (num_bars - 1)) / 2, motions, fontweight='bold') 
    # Add ylabel
    plt.ylabel('Root Mean Squared Error (cm)', fontweight='bold', fontsize=fontsize_labels)
    # Increase fontsize of labels
    plt.tick_params(axis='both', which='major', labelsize=fontsize_labels)
    plt.ylim([0, 5])
    plt.yticks(np.arange(0, 6, 1))
    # Add title
    plt.title('Pelvis translation errors (mean +/- std; 3 degrees of freedom)', fontweight='bold', fontsize=fontsize_title)
    # Create legend
    plt.legend(loc='upper left', fontsize=fontsize_labels)
    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Remove box around legend
    plt.gca().get_legend().get_frame().set_linewidth(0.0)
    plt.tight_layout()
    plt.show()
    
    # %% Merging both plots

    # Your data and parameters here
    barWidth = 0.15
    fontsize_labels = 16
    fontsize_title = 18
    fontsize_legend = 16
    colors = sns.color_palette('colorblind', len(setups))
    num_bars = len(setups)
    positions = [np.arange(len(motions)) + i * barWidth for i in range(num_bars)]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Create a 1x2 grid of subplots (one row, two columns)
    
    # Rotations subplot
    values = [[means_RMSEs_copy[motion]['mean'][i] for motion in motions] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy[motion]['mean'][i] for motion in motions] for i in range(num_bars)]  
    for i in range(num_bars):
        axs[0].bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', label=setups_t[i], align='center', alpha=0.5, ecolor='black', capsize=5)
    
    axs[0].set_xticks(np.arange(len(motions)) + (barWidth * (num_bars - 1)) / 2)
    axs[0].set_xticklabels(motions, fontweight='bold')
    axs[0].set_ylabel('Root Mean Squared Error (deg)', fontweight='bold', fontsize=fontsize_labels)
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_labels)
    axs[0].set_ylim([0, 12])
    axs[0].set_yticks(np.arange(0, 16, 4))
    axs[0].set_title('Joint rotations (18 degrees of freedom)', fontweight='bold', fontsize=fontsize_title)
    # axs[0].legend(loc='upper left', fontsize=fontsize_labels)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    # axs[0].get_legend().get_frame().set_linewidth(0.0)
    
    # Translations subplot
    values = [[means_RMSEs_copy_tr[motion]['mean'][i]*100 for motion in motions] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy_tr[motion]['mean'][i]*100 for motion in motions] for i in range(num_bars)]   
    for i in range(num_bars):
        axs[1].bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', align='center', alpha=0.5, ecolor='black', capsize=5)
    
    axs[1].set_xticks(np.arange(len(motions)) + (barWidth * (num_bars - 1)) / 2)
    axs[1].set_xticklabels(motions, fontweight='bold')
    axs[1].set_ylabel('Root Mean Squared Error (cm)', fontweight='bold', fontsize=fontsize_labels)
    axs[1].tick_params(axis='both', which='major', labelsize=fontsize_labels)
    axs[1].set_ylim([0, 3])
    axs[1].set_yticks(np.arange(0, 4, 1))
    axs[1].set_title('Pelvis translations (3 degrees of freedom)', fontweight='bold', fontsize=fontsize_title)
    # axs[1].legend(loc='upper left', fontsize=fontsize_labels)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    # axs[1].get_legend().get_frame().set_linewidth(0.0)
    
    legend = fig.legend(loc='lower center', fontsize=fontsize_legend, ncol=num_bars)
    legend.set_frame_on(False)
    # plt.subplots_adjust(bottom=2)
    
    # plt.tight_layout()
    plt.show()
    
    # %% Means only

    # Your data and parameters here
    barWidth = 0.15
    fontsize_labels = 16
    fontsize_title = 18
    fontsize_legend = 16
    colors = sns.color_palette('colorblind', len(setups))
    num_bars = len(setups)
    positions = [np.arange(1) + i * barWidth for i in range(num_bars)]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Create a 1x2 grid of subplots (one row, two columns)
    
    # Rotations subplot
    motions_mean = ['mean']
    values = [[means_RMSEs_copy[motion]['mean'][i] for motion in motions_mean] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy[motion]['mean'][i] for motion in motions_mean] for i in range(num_bars)]  
    for i in range(num_bars):
        axs[0].bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', label=setups_t[i], align='center', alpha=0.5, ecolor='black', capsize=5)
    
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Root Mean Squared Error (deg)', fontweight='bold', fontsize=fontsize_labels)
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_labels)
    axs[0].set_ylim([0, 12])
    axs[0].set_yticks(np.arange(0, 16, 4))
    axs[0].set_title('Joint rotations (18 degrees of freedom)', fontweight='bold', fontsize=fontsize_title)
    # axs[0].legend(loc='upper left', fontsize=fontsize_labels)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    # axs[0].get_legend().get_frame().set_linewidth(0.0)
    
    # Translations subplot
    values = [[means_RMSEs_copy_tr[motion]['mean'][i]*100 for motion in motions_mean] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy_tr[motion]['mean'][i]*100 for motion in motions_mean] for i in range(num_bars)]   
    for i in range(num_bars):
        axs[1].bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', align='center', alpha=0.5, ecolor='black', capsize=5)
    
    axs[1].set_xticks([])
    axs[1].set_xticklabels([])
    axs[1].set_ylabel('Root Mean Squared Error (cm)', fontweight='bold', fontsize=fontsize_labels)
    axs[1].tick_params(axis='both', which='major', labelsize=fontsize_labels)
    axs[1].set_ylim([0, 3])
    axs[1].set_yticks(np.arange(0, 4, 1))
    axs[1].set_title('Pelvis translations (3 degrees of freedom)', fontweight='bold', fontsize=fontsize_title)
    # axs[1].legend(loc='upper left', fontsize=fontsize_labels)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    # axs[1].get_legend().get_frame().set_linewidth(0.0)
    
    legend = fig.legend(loc='lower center', fontsize=fontsize_legend, ncol=num_bars)
    legend.set_frame_on(False)
    # plt.subplots_adjust(bottom=2)
    
    # plt.tight_layout()
    plt.show()
    
    # %% Rotations Means only

    # Your data and parameters here
    barWidth = 0.01
    fontsize_labels = 10
    fontsize_title = 10
    fontsize_legend = 8
    colors = sns.color_palette('colorblind', len(setups))
    num_bars = len(setups)
    positions = [np.arange(1) + i * 1.25*barWidth for i in range(num_bars)]
    
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [8, 1]}, figsize=(3.5, 3.5))    
    
    # Rotations subplot
    motions_mean = ['mean']
    values = [[means_RMSEs_copy[motion]['mean'][i] for motion in motions_mean] for i in range(num_bars)]    
    stds = [[stds_RMSEs_copy[motion]['mean'][i] for motion in motions_mean] for i in range(num_bars)]  
    for i in range(num_bars):
        axs[0].bar(positions[i], values[i], yerr=stds[i], color=colors[i], width=barWidth, edgecolor='white', label=setups_t[i], align='center', alpha=0.5, ecolor='black', capsize=5)
    
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('RMSE (deg)', fontweight='bold', fontsize=fontsize_labels)
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize_labels)
    axs[0].set_ylim([0, 12])
    axs[0].set_yticks(np.arange(0, 16, 4))
    axs[0].set_title('Accuracy task', fontweight='bold', fontsize=fontsize_title)    
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    legend = fig.legend(loc='upper center', fontsize=fontsize_legend, ncol=2, bbox_to_anchor=(0.5, 1.5), bbox_transform=axs[1].transAxes)
    legend.set_frame_on(False)
    
    axs[1].axis('off')
    
    # plt.subplots_adjust(bottom=1) 
    # legend = fig.legend(loc='lower center', fontsize=fontsize_legend, ncol=2)
    # legend.set_frame_on(False)
    plt.tight_layout()
    plt.show()
    
    if saveAndOverwriteResults:
        np.save(os.path.join(outputDir, 'means_RMSEs.npy'), means_RMSEs_copy) 
        np.save(os.path.join(outputDir, 'stds_RMSEs.npy'), stds_RMSEs_copy) 



# # %% Further analysis
# # For each coordinate in means_RMSEs_copy[motion][coordinate], report
# # if the best entry is at least 1deg than the second best entry.
# threshold=1.5
# # Loop over motions.
# for motion in motions:
#     # Loop over coordinates.
#     for coordinate in means_RMSEs_copy[motion].keys():
#         # Get the values of the coordinate for each idx_setup.
#         values = [means_RMSEs_copy[motion][coordinate][idx_setup] for idx_setup in idx_setups[1:]]
#         # Get the indices of the sorted values.
#         # idx_sorted = np.argsort(values)
#         # Get the difference between the best and second best values.
#         diff = values[1] - values[0]
#         # If the difference is greater than 1deg, print the coordinate and the setups.
#         if diff < -threshold:
#             print('Better: {} - {}'.format(motion, coordinate))
#         if diff > threshold:
#             print('Worse: {} - {}'.format(motion, coordinate))

#     # %% Means only
#     # Same figure as above but with only one subplot corresponding to the motion 'walking'.
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     ax.set_title('Joint kinematic errors - Means per coordinate', fontweight='bold', fontsize=fontsize_title)
#     ax.set_ylabel('Root Mean Squared Error (deg)', fontweight='bold', fontsize=fontsize_labels)
#     ax.set_xticks(np.arange(len(xtick_labels)))
#     ax.set_xticklabels(xtick_labels_labels)
#     # For each field in means_RMSEs['all'], plot bars with the values of the field for each idx_setups.
#     for i, field in enumerate(xtick_labels):
#         for j, idx_setup in enumerate(idx_setups):
#             ax.bar(i+x[j]*bar_width, means_RMSEs_copy['mean'][field][idx_setup], bar_width, yerr=stds_RMSEs_copy['mean'][field][idx_setup], color=colors[j], ecolor='black', alpha=0.5, edgecolor='white', capsize=2)
#             # Add text with the value of the bar.
#             if i == len(xtick_labels)-1:
#                 ax.text(i+x[j]*bar_width-0.08, means_RMSEs_copy['mean'][field][idx_setup], int(np.round(means_RMSEs_copy['mean'][field][idx_setup], 0)), ha='center', va='bottom')
#     # Add legend with idx_setups
#     # leg_t = [setups[idx_setup] for idx_setup in idx_setups]
#     leg_t = [setups_t[idx_setup] for idx_setup in idx_setups]
#     # Get what is after the last '_' in leg_t.
#     # leg_t = [leg_t[i].split('_')[-1] for i in range(len(leg_t))]
#     ax.legend(leg_t, loc='upper left', fontsize=fontsize_labels)
#     plt.gca().get_legend().get_frame().set_linewidth(0.0)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.tick_params(axis='both', which='major', labelsize=fontsize_labels)
#     plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
#     plt.tight_layout()
#     plt.show()
    
# # %% MEs
# # Get len(cameraSetups) color-blind frienly colors.
# colors = sns.color_palette('colorblind', len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)*len(processingTypes))
# motions = list(means_MEs.keys())
# # Create the x-tick labels for all subplots.
# xtick_labels = list(means_MEs[motions[0]].keys())
# # Remove pelvis_tx, pelvis_ty and pelivs_tz from xtick_labels.
# xtick_labels = [xtick_label for xtick_label in xtick_labels if xtick_label not in coordinates_tr] + ['mean']
# # remove _l at the end of the xtick_labels if present
# xtick_labels_labels = [xtick_label[:-2] if xtick_label[-2:] == '_l' else xtick_label for xtick_label in xtick_labels]

# # Copy means_MEs to means_MEs_copy.
# means_MEs_copy = copy.deepcopy(means_MEs)
# # Exclude coordinates_tr from means_MEs_copy
# for motion in means_MEs_copy:
#     for coordinate in coordinates_tr:
#         means_MEs_copy[motion].pop(coordinate)

# # Compute mean, this should match means_RMSE_summary_rot
# for motion in means_MEs_copy:
#     # Add field mean that contains the mean of the MEs for all coordinates.
#     # Stack lists from all fiedls of means_MEs_copy[motion] in one numpy array.
#     # Count twice the bilateral coordinates.
#     c_stack = np.zeros((len(coordinates_lr_rot) + len(coordinates_bil), len(setups)))
#     count = 0
#     for i, coordinate in enumerate(coordinates_lr_rot):
#         c_stack[count, :] = means_MEs_copy[motion][coordinate]
#         count += 1
#         if coordinate[-2:] == '_l':
#             c_stack[count, :] = means_MEs_copy[motion][coordinate]
#             count += 1
#     means_MEs_copy[motion]['mean'] = list(np.mean(c_stack, axis=0))

# for cameraSetup in cameraSetups:

#     # Create a figure with 1 column and as many columns as fields in means_MEs.
#     fig, axs = plt.subplots(len(means_MEs_copy.keys()), 1, figsize=(10, 5*len(means_MEs_copy.keys())))
#     fig.suptitle(cameraSetup)

#     # Get indices of setups for the camera setup.
#     idx_setups = [i for i, setup in enumerate(setups) if cameraSetup in setup]
#     bar_width = 0.8/len(idx_setups)

#     # Create list of integers with that has as many elements as there are idx_setups. The list has values with
#     # a step of 1 and is centered on 0.
#     x = np.arange(len(idx_setups)) - (len(idx_setups)-1)/2

#     # Loop over subplots in axs.
#     for a, ax in enumerate(axs):
#         ax.set_title(motions[a])
#         ax.set_ylabel('RMSE (deg)')
#         ax.set_xticks(np.arange(len(xtick_labels)))
#         if a == len(axs)-1:
#             axs[a].set_xticklabels(xtick_labels)
#         else:
#             axs[a].set_xticklabels([])        
        
#         # For each field in means_MEs['all'], plot bars with the values of the field for each idx_setups.
#         for i, field in enumerate(xtick_labels):
#             for j, idx_setup in enumerate(idx_setups):
#                 ax.bar(i+x[j]*bar_width, means_MEs_copy[motions[a]][field][idx_setup], bar_width, color=colors[j])
#                 # Add text with the value of the bar.
#                 if i == len(xtick_labels)-1:
#                     ax.text(i+x[j]*bar_width, means_MEs_copy[motions[a]][field][idx_setup], np.round(means_MEs_copy[motions[a]][field][idx_setup], 1), ha='center', va='bottom')    
#         # Add legend with idx_setups
#         leg_t = [setups[idx_setup] for idx_setup in idx_setups]
#         # Get what is after the last '_' in leg_t.
#         # leg_t = [leg_t[i].split('_')[-1] for i in range(len(leg_t))]
#         ax.legend(leg_t)        
#         plt.show()
        
#     # Set the x-tick labels for the last subplot only.
#     axs[-1].set_xticks(np.arange(len(xtick_labels)))
#     axs[-1].set_xticklabels(xtick_labels_labels)

#     # Same figure as above but with only one subplot corresponding to the motion 'walking'.
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     ax.set_title(motions[1])
#     ax.set_ylabel('RMSE (deg)')
#     ax.set_xticks(np.arange(len(xtick_labels)))
#     ax.set_xticklabels(xtick_labels_labels)
#     # For each field in means_MEs['all'], plot bars with the values of the field for each idx_setups.
#     for i, field in enumerate(xtick_labels):
#         for j, idx_setup in enumerate(idx_setups):
#             ax.bar(i+x[j]*bar_width, means_MEs_copy[motions[1]][field][idx_setup], bar_width, color=colors[j])
#             # Add text with the value of the bar.
#             if i == len(xtick_labels)-1:
#                 ax.text(i+x[j]*bar_width, means_MEs_copy[motions[1]][field][idx_setup], np.round(means_MEs_copy[motions[1]][field][idx_setup], 1), ha='center', va='bottom')
#     # Add legend with idx_setups
#     leg_t = [setups[idx_setup] for idx_setup in idx_setups]
#     # Get what is after the last '_' in leg_t.
#     # leg_t = [leg_t[i].split('_')[-1] for i in range(len(leg_t))]
#     ax.legend(leg_t)
#     plt.show()

    
    


