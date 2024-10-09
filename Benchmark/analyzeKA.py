"""
    This script computes and analyses the RMSEs between mocap-based on video-
    based kinematics.

    TOREAD: Paper results are using updated instead of updated_benchmark_reprojection
    Since it didn't change much to add the reprojection, we will stick to the updated ones
    to have consistency between IK and IK used for dynamic simulations
"""

import os
import sys
import numpy as np
import pandas as pd
sys.path.append("..") # utilities in child directory
from utils import getDataDirectory, storage2df
import matplotlib.pyplot as plt

part1 = False
saveResults = True
loadResults = True
part2 = False
part3 = True

dataDir = os.path.join(getDataDirectory(False), 'Data')
# subjects = ['subject' + str(i) for i in range(2, 12)]
# tasksKA = [None, 0.001, 0.005, 0.01, 0.1]
# motion_types = ['walking', 'STS', 'squat', 'DJ']
subjects = ['subject' + str(i) for i in range(10,11)]
tasksKA = [None, 0.005, 0.01]
motion_types = ['walking', 'STS', 'squat', 'DJ']
outputDir = os.path.join(getDataDirectory(False), 'Results-kam_benchmark')

# Part 1: data extraction
defaultModelName = 'LaiUhlrich2022_KA'
cases_to_exclude_trials = {'subject2': ['walkingTS3']}
cases_to_exclude_paper = ['static', 'stsasym', 'stsfast', 'walkingti', 'walkingto']
coordinates = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_l', 'knee_angle_r', 'knee_adduction_l', 'knee_adduction_r', 
    'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l', 'subtalar_angle_r',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
sides = ['r', 'l']

if part1:
    count_mocap = 0
    count_opencap = 0
    range_ka, ka = {}, {}
    range_ka['all'] = {}
    for subject in subjects:
        subjectDir = os.path.join(dataDir, subject)
        opensimDir = os.path.join(subjectDir, 'OpenSimData')
        mocapDir = os.path.join(opensimDir, 'Mocap_updated')
        mocapIKDir = os.path.join(mocapDir, 'IK')
        opencapDir = os.path.join(opensimDir, 'Video', 'mmpose_0.8', '2-cameras', 'v0.63_updated_benchmark_reprojection_kam')
        opencapIKDir = os.path.join(opencapDir, 'IK')
        range_ka[subject], ka[subject] = {}, {}
        for taskKA in tasksKA:
            modelName = defaultModelName
            if taskKA:
                modelName += '_KAtask_{}'.format(taskKA)
            mocapIKModelDir = os.path.join(mocapIKDir, modelName)
            opencapIKModelDir = os.path.join(opencapIKDir, modelName)
            range_ka[subject][taskKA], ka[subject][taskKA] = {}, {}
            range_ka['all'][taskKA] = {}
            trials = []
            for motion_type in motion_types:
                range_ka['all'][taskKA][motion_type] = {}
                range_ka['all'][taskKA][motion_type]['mocap'] = []
                range_ka['all'][taskKA][motion_type]['opencap'] = []        
            
            for trial in os.listdir(mocapIKModelDir):
                trialName = trial.replace('.mot', '')
                # Continue if not a motion file
                if not trial[-3:] == 'mot':
                    continue
                if 'baseline' in trial:
                    continue            
                if subject in cases_to_exclude_trials and trialName in cases_to_exclude_trials[subject]:
                    continue
                in_vec = False
                for case_to_exclude_paper in cases_to_exclude_paper:                                
                    if case_to_exclude_paper in trial[:-4].lower():
                        # print('Exclude {}'.format(trial))  
                        in_vec = True
                if in_vec:
                    continue
                
                mocapTrialPath = os.path.join(mocapIKModelDir, trial)
                trialNameOpenCap = trial.replace('.mot', '_videoAndMocap.mot')
                opencapTrialPath = os.path.join(opencapIKModelDir, trialNameOpenCap)
                # Check if the trial exists in the mocap directory
                if not os.path.exists(mocapTrialPath):
                    print('Mocap: Trial {} does not exist for subject {}'.format(trial, subject))
                    continue
                else:
                    count_mocap += 1
                # Check if the trial exists in the OpenCap directory
                if not os.path.exists(opencapTrialPath):
                    print('OpenCap: Trial {} does not exist for subject {}'.format(trial, subject))
                    continue
                else:
                    count_opencap += 1
                # Extract data
                mocap_data = storage2df(mocapTrialPath, coordinates)
                opencap_data = storage2df(opencapTrialPath, coordinates)
                rom_mocap_ka, rom_opencap_ka, ka[subject][taskKA][trialName] = {}, {}, {}
                for side in sides:
                    mocap_ka = mocap_data['knee_adduction_{}'.format(side)]
                    opencap_ka = opencap_data['knee_adduction_{}'.format(side)]                
                    ka[subject][taskKA][trialName][side] = {
                        'mocap': mocap_ka,
                        'opencap': opencap_ka,
                    }
                    rom_mocap_ka[side] = [np.min(mocap_ka), np.max(mocap_ka)]
                    rom_opencap_ka[side] = [np.min(opencap_ka), np.max(opencap_ka)]
                rom_mocap_ka['all'] = [np.min(rom_mocap_ka['r'] + rom_mocap_ka['l']), np.max(rom_mocap_ka['r'] + rom_mocap_ka['l'])]
                rom_opencap_ka['all'] = [np.min(rom_opencap_ka['r'] + rom_opencap_ka['l']), np.max(rom_opencap_ka['r'] + rom_opencap_ka['l'])]

                range_ka[subject][taskKA][trialName] = {
                    'mocap': rom_mocap_ka['all'],
                    'opencap': rom_opencap_ka['all'],
                }

                for motion_type in motion_types:
                    if motion_type.lower() in trialName.lower():
                        break
                    
                range_ka['all'][taskKA][motion_type]['mocap'].append(range_ka[subject][taskKA][trial.replace('.mot', '')]['mocap'])
                range_ka['all'][taskKA][motion_type]['opencap'].append(range_ka[subject][taskKA][trial.replace('.mot', '')]['opencap'])                    

    print('Number of mocap trials: {}'.format(count_mocap))
    print('Number of OpenCap trials: {}'.format(count_opencap))

    if saveResults:
        np.save(os.path.join(outputDir, 'range_ka.npy'), range_ka)
        np.save(os.path.join(outputDir, 'ka.npy'), ka)

if loadResults:
    range_ka = np.load(os.path.join(outputDir, 'range_ka.npy'), allow_pickle=True).item()
    ka = np.load(os.path.join(outputDir, 'ka.npy'), allow_pickle=True).item()

# Part 2: Data analysis
if part2:
    for motion_type in motion_types:
        for taskKA in tasksKA:    
            c_mocap = np.array(range_ka['all'][taskKA][motion_type]['mocap'])
            c_opencap = np.array(range_ka['all'][taskKA][motion_type]['opencap'])        
            range_ka['all'][taskKA][motion_type]['mocap_ext'] = [np.min(c_mocap, 0)[0], np.max(c_mocap, 0)[1]]
            range_ka['all'][taskKA][motion_type]['opencap_ext'] = [np.min(c_opencap, 0)[0], np.max(c_opencap, 0)[1]]
            range_ka['all'][taskKA][motion_type]['mocap_mean'] = [np.mean(c_mocap, 0)[0], np.mean(c_mocap, 0)[1]]
            range_ka['all'][taskKA][motion_type]['opencap_mean'] = [np.mean(c_opencap, 0)[0], np.mean(c_opencap, 0)[1]]
            
            print("Range of KA for {} task {} motion: Mocap: {} OpenCap: {}".format(taskKA, motion_type, np.round(range_ka['all'][taskKA][motion_type]['mocap_ext'], 1), np.round(range_ka['all'][taskKA][motion_type]['opencap_ext'], 1)))
        print('')
# Part 3: Plotting
linestyles = ['-', '--']
colors = ['b', 'r']

nSubplots = {'walking': 6, 'STS': 2, 'squat': 2, 'DJ': 6}


if part3:
    for subject in subjects:
        for motion_type in motion_types:
            # Create figure
            fig, axs = plt.subplots(2, nSubplots[motion_type], figsize=(10, 10))
            fig.suptitle('Subject {} - {}'.format(subject, motion_type))
            c_trial = 0
            for trial in list(ka[subject][tasksKA[0]].keys()):                
                if not motion_type.lower() in trial.lower():
                    continue                                
                for c_taskKA, taskKA in enumerate(tasksKA):
                    for c_side, side in enumerate(sides):                        
                        if c_taskKA == len(tasksKA) - 1 and c_trial == nSubplots[motion_type] - 1:                            
                            axs[c_side][c_trial].plot(ka[subject][taskKA][trial][side]['mocap'], color=colors[0], linestyle=linestyles[c_side], label='Mocap')
                            axs[c_side][c_trial].plot(ka[subject][taskKA][trial][side]['opencap'], color=colors[1], linestyle=linestyles[c_side], label='OpenCap')           
                        else:                        
                            axs[c_side][c_trial].plot(ka[subject][taskKA][trial][side]['mocap'], color=colors[0], linestyle=linestyles[c_side])
                            axs[c_side][c_trial].plot(ka[subject][taskKA][trial][side]['opencap'], color=colors[1], linestyle=linestyles[c_side])
                
                # If last plot, add legend
                if c_trial == nSubplots[motion_type] - 1:
                    axs[0][c_trial].legend()   
                    
                c_trial += 1
