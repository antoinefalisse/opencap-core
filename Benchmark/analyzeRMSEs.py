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

plots = False
suffix_files = ''

# %% User inputs.
subjects = ['subject' + str(i) for i in range(2, 12)]

poseDetectors = ['OpenPose_1x1008_4scales']
cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
augmenterTypes = ['v0.7']
augmenterTypeOffset = 'v0.7'

# Cases to exclude to make sure we have the same number of trials per subject
cases_to_exclude_trials = {'subject2': ['walkingTS3'],
                           'subject6': ['walkingTO1']}
# Cases to exclude because of failed syncing (mocap vs opencap)
cases_to_exclude_syncing = {}
cases_to_exclude_syncing = {
    'subject3': {'OpenPose': {'2-cameras': ['walking1', 'walkingTI1', 'walkingTI2', 'walkingTO1', 'walkingTO2', 'walkingTS3', 'walkingTS4']}}}
# Cases to exclude because of failed algorithm (opencap)
cases_to_exclude_algo = {
    'subject2': {'OpenPose_generic': {'3-cameras': ['walkingTS1', 'walkingTS2', 'walkingTS4', 'DJ1', 'DJ2', 'DJ3', 'DJAsym1', 'DJAsym4', 'DJAsym5'],
                                      '5-cameras': ['walkingTS2']}},
    'subject3': {'OpenPose_generic': {'2-cameras': ['walkingTS2', 'walkingTS4']}}}

# Cases to exclude for paper
cases_to_exclude_paper = ['static', 'stsasym', 'stsfast', 'walkingti', 'walkingto']

fixed_markers = False # False should be default (better results)

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

# %%
scriptDir = os.getcwd()
repoDir = os.path.dirname(scriptDir)
mainDir = getDataDirectory(False)
dataDir = os.path.join(mainDir)

if not os.path.exists(os.path.join(dataDir, 'Data', 'RMSEs.npy')): 
    RMSEs = {}
else:  
    RMSEs = np.load(os.path.join(dataDir, 'Data', 'RMSEs.npy'),
                    allow_pickle=True).item()    
if not os.path.exists(os.path.join(dataDir, 'Data', 'MAEs.npy')): 
    MAEs = {}
else:  
    MAEs = np.load(os.path.join(dataDir, 'Data', 'MAEs.npy'),
                    allow_pickle=True).item()

RMSEs = {}
RMSEs['all'] = {}
MAEs = {}
MAEs['all'] = {}
for motion in motions:
    RMSEs[motion] = {}
    MAEs[motion] = {}
    
for subjectName in subjects:
    RMSEs[subjectName] = {}
    MAEs[subjectName] = {}
    print('\nProcessing {}'.format(subjectName))
    if fixed_markers:
        osDir = os.path.join(dataDir, 'Data', subjectName, 'OpenSimData_fixed')
    else:
        osDir = os.path.join(dataDir, 'Data', subjectName, 'OpenSimData')
    markerDir = os.path.join(dataDir, 'Data', subjectName, 'MarkerData')
    mocapDir = os.path.join(osDir, 'Mocap', 'IK', genericModel4ScalingName[:-5])
    
    trials = []
    for trial in os.listdir(mocapDir):
        if not trial[-3:] == 'mot':
            continue
        trials.append(trial[:-4])
    count = 0
    for trial in os.listdir(mocapDir):
        if not trial[-3:] == 'mot':
            continue
        pathTrial = os.path.join(mocapDir, trial)
        trial_mocap_df = storage2df(pathTrial, coordinates)
        
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
                    
                for augmenterType in augmenterTypes:
                    if not augmenterType in RMSEs[subjectName][poseDetector][cameraSetup]:
                        RMSEs[subjectName][poseDetector][cameraSetup][augmenterType] = pd.DataFrame(columns=coordinates, index=trials)
                    if not augmenterType in RMSEs['all'][poseDetector][cameraSetup]:
                        RMSEs['all'][poseDetector][cameraSetup][augmenterType] = pd.DataFrame(columns=coordinates)
                    for motion in motions:
                        if not augmenterType in RMSEs[motion][poseDetector][cameraSetup]:
                            RMSEs[motion][poseDetector][cameraSetup][augmenterType] = pd.DataFrame(columns=coordinates)
                            
                    if not augmenterType in MAEs[subjectName][poseDetector][cameraSetup]:
                        MAEs[subjectName][poseDetector][cameraSetup][augmenterType] = pd.DataFrame(columns=coordinates, index=trials)
                    if not augmenterType in MAEs['all'][poseDetector][cameraSetup]:
                        MAEs['all'][poseDetector][cameraSetup][augmenterType] = pd.DataFrame(columns=coordinates)
                    for motion in motions:
                        if not augmenterType in MAEs[motion][poseDetector][cameraSetup]:
                            MAEs[motion][poseDetector][cameraSetup][augmenterType] = pd.DataFrame(columns=coordinates)
                    
                    if (subjectName in cases_to_exclude_trials and
                        trial[:-4] in cases_to_exclude_trials[subjectName]):
                        print('Exclude {} - {}'.format(subjectName, trial[:-4]))
                        continue
                    
                    if (subjectName in cases_to_exclude_syncing and 
                        poseDetector in cases_to_exclude_syncing[subjectName] and 
                        cameraSetup in cases_to_exclude_syncing[subjectName][poseDetector] and 
                        trial[:-4] in cases_to_exclude_syncing[subjectName][poseDetector][cameraSetup]):
                        print('Exclude {} - {} - {} - {}'.format(subjectName, poseDetector, cameraSetup, trial[:-4]))
                        continue
                    
                    if (subjectName in cases_to_exclude_algo and 
                        poseDetector in cases_to_exclude_algo[subjectName] and 
                        cameraSetup in cases_to_exclude_algo[subjectName][poseDetector] and 
                        trial[:-4] in cases_to_exclude_algo[subjectName][poseDetector][cameraSetup]):
                        print('Exclude {} - {} - {} - {}'.format(subjectName, poseDetector, cameraSetup, trial[:-4]))
                        continue
                    
                    in_vec = False
                    for case_to_exclude_paper in cases_to_exclude_paper:                                
                        if case_to_exclude_paper in trial[:-4].lower():
                            print('Exclude {}'.format(trial))  
                            in_vec = True
                    if in_vec:
                        continue
                        
                    cameraSetupDir = os.path.join(
                        poseDetectorDir, cameraSetup, augmenterType, 'IK', 
                        genericModel4ScalingName[:-5])
                    cameraSetupMarkerDir = os.path.join(
                        poseDetectorMarkerDir, cameraSetup, augmenterTypeOffset)
                    
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
                    time_start = trial_video_np[0, 0]
                    time_end = trial_video_np[-1, 0]
                    # Find corresponding indices in mocap data
                    idx_start = np.argwhere(trial_mocap_np[:, 0] == time_start)[0][0]
                    idx_end = np.argwhere(trial_mocap_np[:, 0] == time_end)[0][0]
                    # Select mocap data based on video-based time vector
                    trial_mocap_np_adj = trial_mocap_np[idx_start:idx_end+1, :]                
                    
                    # Compute RMSEs and MAEs
                    y_true = trial_mocap_np_adj[:, 1:]
                    y_pred = trial_video_np[:, 1:]
                    
                    c_rmse = []
                    c_mae = []
                    
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
                                y_pred[:, count1] -= c_trc_m1_offset[0]/1000
                            elif c_coord == 'pelvis_ty':
                                y_pred[:, count1] -= c_trc_m1_offset[1]/1000
                            elif c_coord == 'pelvis_tz':
                                y_pred[:, count1] -= c_trc_m1_offset[2]/1000
                        # RMSE
                        value = mean_squared_error(
                            y_true[:, count1], y_pred[:, count1], squared=False)
                        RMSEs[subjectName][poseDetector][cameraSetup][augmenterType].loc[trials[count], c_coord] = value
                        c_rmse.append(value)
                        # MAE                        
                        value2 = mean_absolute_error(
                            y_true[:, count1], y_pred[:, count1])
                        MAEs[subjectName][poseDetector][cameraSetup][augmenterType].loc[trials[count], c_coord] = value2
                        c_mae.append(value2)
                        
                        if value > 20:
                            print('{} - {} - {} - {} - {} - {} had RMSE of {}'.format(subjectName, poseDetector, cameraSetup, augmenterType, trial[:-4], c_coord, value))
                        
                    c_name = subjectName + '_' +  trial[:-4]
                    RMSEs['all'][poseDetector][cameraSetup][augmenterType].loc[c_name] = c_rmse
                    MAEs['all'][poseDetector][cameraSetup][augmenterType].loc[c_name] = c_mae
                    
                    for motion in motions:
                        if motion in trial:
                            RMSEs[motion][poseDetector][cameraSetup][augmenterType].loc[c_name] = c_rmse
                            MAEs[motion][poseDetector][cameraSetup][augmenterType].loc[c_name] = c_mae  

        count += 1
        
# %% Plots per coordinate: RMSEs
all_motions = ['all'] + motions
bps, means_RMSEs, medians_RMSEs = {}, {}, {}
for motion in all_motions:
    bps[motion], means_RMSEs[motion], medians_RMSEs[motion] = {}, {}, {}
    if plots:
        fig, axs = plt.subplots(5, 3, sharex=True)    
        fig.suptitle(motion)    
    for count, coordinate in enumerate(coordinates_lr):
        c_data = {}
        for augmenterType in augmenterTypes:
            for poseDetector in poseDetectors:
                for cameraSetup in cameraSetups:                
                    if coordinate[-2:] == '_l':
                        c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType] = (                    
                            RMSEs[motion][poseDetector][cameraSetup][augmenterType][coordinate].tolist() +                     
                            RMSEs[motion][poseDetector][cameraSetup][augmenterType][coordinate[:-2] + '_r'].tolist())
                        coordinate_title = coordinate[:-2]
                    else:
                        c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType] = (
                            RMSEs[motion][poseDetector][cameraSetup][augmenterType][coordinate].tolist())
                        coordinate_title = coordinate                    
        if plots:
            ax = axs.flat[count]
            bps[motion][coordinate] = ax.boxplot(c_data.values())
            ax.set_title(coordinate_title)
            xticks = list(range(1, len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)+1))
            ax.set_xticks(xticks)
            ax.set_xticklabels(c_data.keys(), rotation = 90)
            ax.set_ylim(0, 20)
            ax.axhline(y=5, color='r', linestyle='--')
            ax.axhline(y=10, color='b', linestyle='--')
        
        means_RMSEs[motion][coordinate] = [np.mean(c_data[a]) for a in c_data]
        medians_RMSEs[motion][coordinate] = [np.median(c_data[a]) for a in c_data]
        
# %% Print out csv files with results: median RMSEs
setups = []
for augmenterType in augmenterTypes:
    for poseDetector in poseDetectors:
        for cameraSetup in cameraSetups:        
            setups.append(poseDetector + '_' + cameraSetup + '_' + augmenterType)
outputDir = os.path.join(dataDir, 'Results-paper-augmenterV2')

suffixRMSE = ''
if fixed_markers:
    suffixRMSE = '_fixed'    
        
with open(os.path.join(outputDir,'RMSEs{}_medians{}.csv'.format(suffixRMSE,suffix_files)), 'w', newline='') as csvfile:
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
        
with open(os.path.join(outputDir,'RMSEs{}_means{}.csv'.format(suffixRMSE,suffix_files)), 'w', newline='') as csvfile:
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
    for idxMotion, motion in enumerate(all_motions):
        means_RMSE_summary[motion], mins_RMSE_summary[motion], maxs_RMSE_summary[motion] = {}, {}, {}
        c_bp = means_RMSEs[motion]
        for idxSetup, setup in enumerate(setups):
            means_RMSE_summary[motion][setup] = {}
            mins_RMSE_summary[motion][setup] = {}
            maxs_RMSE_summary[motion][setup] = {}
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
            
            maxs_RMSE_summary[motion][setup]['rotation'] = np.round(np.max(temp_med_rot),1)
            maxs_RMSE_summary[motion][setup]['translation'] = np.round(np.max(temp_med_tr*1000),1)

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
        for augmenterType in augmenterTypes:
            for poseDetector in poseDetectors:
                for cameraSetup in cameraSetups:                
                    if coordinate[-2:] == '_l':
                        c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType] = (                    
                            MAEs[motion][poseDetector][cameraSetup][augmenterType][coordinate].tolist() +                     
                            MAEs[motion][poseDetector][cameraSetup][augmenterType][coordinate[:-2] + '_r'].tolist())
                        coordinate_title = coordinate[:-2]
                    else:
                        c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType] = (
                            MAEs[motion][poseDetector][cameraSetup][augmenterType][coordinate].tolist())
                        coordinate_title = coordinate                    
        if plots:
            ax = axs.flat[count]
            bps[motion][coordinate] = ax.boxplot(c_data.values())        
            ax.set_title(coordinate_title)
            xticks = list(range(1, len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)+1))
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
        
with open(os.path.join(outputDir,'MAEs{}_medians{}.csv'.format(suffixMAE,suffix_files)), 'w', newline='') as csvfile:
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
        
with open(os.path.join(outputDir,'MAEs{}_means{}.csv'.format(suffixRMSE,suffix_files)), 'w', newline='') as csvfile:
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
with open(os.path.join(outputDir,'MAEs{}_means_paper{}.csv'.format(suffixRMSE,suffix_files)), 'w', newline='') as csvfile:
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
with open(os.path.join(outputDir,'RMSEs{}_means_paper{}.csv'.format(suffixRMSE,suffix_files)), 'w', newline='') as csvfile:
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
for setup in setups:
    means_RMSE_summary_rot[setup], means_RMSE_summary_tr[setup] = [], []
    mins_RMSE_summary_rot[setup], mins_RMSE_summary_tr[setup] = [], []
    maxs_RMSE_summary_rot[setup], maxs_RMSE_summary_tr[setup] = [], []
    for motion in motions:
        means_RMSE_summary_rot[setup].append(means_RMSE_summary[motion][setup]['rotation'])
        means_RMSE_summary_tr[setup].append(means_RMSE_summary[motion][setup]['translation'])
        mins_RMSE_summary_rot[setup].append(mins_RMSE_summary[motion][setup]['rotation'])
        mins_RMSE_summary_tr[setup].append(mins_RMSE_summary[motion][setup]['translation'])
        maxs_RMSE_summary_rot[setup].append(maxs_RMSE_summary[motion][setup]['rotation'])
        maxs_RMSE_summary_tr[setup].append(maxs_RMSE_summary[motion][setup]['translation'])
    
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
for c_s, setup in enumerate(setups):
    c_mean_rot_all_RMSE[c_s] = np.round(np.mean(np.asarray( means_RMSE_summary_rot[setup])),1)
    c_std_rot_all_RMSE[c_s] = np.round(np.std(np.asarray( means_RMSE_summary_rot[setup])),1)  
    c_min_rot_all_RMSE[c_s] = np.round(np.min(np.asarray( mins_RMSE_summary_rot[setup])),1)
    c_max_rot_all_RMSE[c_s] = np.round(np.max(np.asarray( maxs_RMSE_summary_rot[setup])),1)
    print("{}: {} +/- {} [{} {}]".format(setup, c_mean_rot_all_RMSE[c_s], c_std_rot_all_RMSE[c_s], c_min_rot_all_RMSE[c_s], c_max_rot_all_RMSE[c_s]))
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
for c_s, setup in enumerate(setups):
    c_mean_tr_all_RMSE[c_s] = np.round(np.mean(np.asarray( means_RMSE_summary_tr[setup])),1)
    c_std_tr_all_RMSE[c_s] = np.round(np.std(np.asarray( means_RMSE_summary_tr[setup])),1)
    c_min_tr_all_RMSE[c_s] = np.round(np.min(np.asarray( mins_RMSE_summary_tr[setup])),1)
    c_max_tr_all_RMSE[c_s] = np.round(np.max(np.asarray( maxs_RMSE_summary_tr[setup])),1)
    print("{}: {} +/- {} [{} {}]".format(setup, c_mean_tr_all_RMSE[c_s], c_std_tr_all_RMSE[c_s], c_min_tr_all_RMSE[c_s], c_max_tr_all_RMSE[c_s]))
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