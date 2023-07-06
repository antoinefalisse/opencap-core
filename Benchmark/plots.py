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

plots = False
saveAndOverwriteResults = True
overwriteResults = False

scriptDir = os.getcwd()
repoDir = os.path.dirname(scriptDir)
mainDir = getDataDirectory(False)
dataDir = os.path.join(mainDir)
outputDir = os.path.join(dataDir, 'Results-paper-augmenterV2')

# Effect of # cameras
# cases = {'1': {'poseDetector': 'mmpose_0.8', 'cameraSetup': '2-cameras', 'augmenterType': 'v0.54'},
#          '2': {'poseDetector': 'mmpose_0.8', 'cameraSetup': '3-cameras', 'augmenterType': 'v0.54'},
#          '3': {'poseDetector': 'mmpose_0.8', 'cameraSetup': '5-cameras', 'augmenterType': 'v0.54'}}
# mySuptitle = 'Effect of adding cameras using mmpose'
# setups_t = ['2-cameras', '3-cameras', '5-cameras']

# Effect of pose detector
cases = {'1': {'poseDetector': 'mmpose_0.8', 'cameraSetup': '2-cameras', 'augmenterType': 'v0.54'},
          '2': {'poseDetector': 'OpenPose_1x736', 'cameraSetup': '2-cameras', 'augmenterType': 'v0.45'},
          '3': {'poseDetector': 'OpenPose_1x1008_4scales', 'cameraSetup': '2-cameras', 'augmenterType': 'v0.45'}}
mySuptitle = 'Effect of using different pose detection models with 2-cameras'
setups_t = ['mmpose', 'OpenPose_1x736', 'OpenPose_1x1008_4scales']

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
processingType = 'IK_IK'

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
    

    
# %% RMSEs per case
all_motions = ['all'] + motions
bps, means_RMSEs, medians_RMSEs = {}, {}, {}
for motion in all_motions:
    bps[motion], means_RMSEs[motion], medians_RMSEs[motion] = {}, {}, {}
    if plots:
        fig, axs = plt.subplots(5, 3, sharex=True)    
        fig.suptitle(motion)    
    for count, coordinate in enumerate(coordinates_lr):
        c_data = {}
        for case in list(cases.keys()):
            augmenterType = cases[case]['augmenterType']
            poseDetector = cases[case]['poseDetector']
            cameraSetup = cases[case]['cameraSetup']                
            if coordinate[-2:] == '_l':
                c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType] = (                    
                    RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist() +                     
                    RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate[:-2] + '_r'].tolist())
                coordinate_title = coordinate[:-2]
            else:
                c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType+ '_' + processingType] = (
                    RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist())
                coordinate_title = coordinate        
        means_RMSEs[motion][coordinate] = [np.mean(c_data[a]) for a in c_data]
        medians_RMSEs[motion][coordinate] = [np.median(c_data[a]) for a in c_data]

# %% Plos
# Get len(cameraSetups) color-blind frienly colors.
colors = sns.color_palette('colorblind', len(cases))
# Copy means_RMSEs to means_RMSEs_copy.
means_RMSEs_copy = copy.deepcopy(means_RMSEs)
# We do want to the mean across all trials, but rather the across the mean of
# each motion type. Remove fiedl 'all' from means_RMSEs_copy
means_RMSEs_copy.pop('all')
motions = list(means_RMSEs_copy.keys())
# Create a new fiedl 'mean' in means_RMSEs_copy
means_RMSEs_copy['mean'] = {}
# Compute mean across all motions for each coordinate.
for coordinate in means_RMSEs_copy[motions[0]]:
    means_RMSEs_copy['mean'][coordinate] = []
    for i in range(len(means_RMSEs_copy[motions[0]][coordinate])):
        means_RMSEs_copy['mean'][coordinate].append(np.mean([means_RMSEs_copy[motion][coordinate][i] for motion in motions], axis=0))
motions.append('mean')
# Exclude coordinates_tr from means_RMSEs_copy
for motion in means_RMSEs_copy:
    for coordinate in coordinates_tr:
        means_RMSEs_copy[motion].pop(coordinate)
# Compute mean, this should match means_RMSE_summary_rot
for motion in means_RMSEs_copy:
    # Add field mean that contains the mean of the RMSEs for all coordinates.
    # Stack lists from all fiedls of means_RMSEs_copy[motion] in one numpy array.
    # Count twice the bilateral coordinates.
    c_stack = np.zeros((len(coordinates_lr_rot) + len(coordinates_bil), len(cases)))
    count = 0
    for i, coordinate in enumerate(coordinates_lr_rot):
        c_stack[count, :] = means_RMSEs_copy[motion][coordinate]
        count += 1
        if coordinate[-2:] == '_l':
            c_stack[count, :] = means_RMSEs_copy[motion][coordinate]
            count += 1
    means_RMSEs_copy[motion]['mean'] = list(np.mean(c_stack, axis=0))
    
# Create the x-tick labels for all subplots.
xtick_labels = list(means_RMSEs[motions[0]].keys())
# Remove pelvis_tx, pelvis_ty and pelivs_tz from xtick_labels.
xtick_labels = [xtick_label for xtick_label in xtick_labels if xtick_label not in coordinates_tr] + ['mean']
# remove _l at the end of the xtick_labels if present
xtick_labels_labels = [xtick_label[:-2] if xtick_label[-2:] == '_l' else xtick_label for xtick_label in xtick_labels]

xtick_values = ['pelvis_tilt', 'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l', 'lumbar_extension', 'mean']

# Create a figure with 1 column and as many columns as fields in means_RMSEs.
fig, axs = plt.subplots(len(means_RMSEs_copy.keys()), 1, figsize=(10, 5*len(means_RMSEs_copy.keys())))
fig.suptitle(mySuptitle)

# Get indices of setups for the camera setup.
# idx_setups = [i for i, setup in enumerate(setups) if cameraSetup in setup]
bar_width = 0.8/len(cases)

# Create list of integers with that has as many elements as there are idx_setups. The list has values with
# a step of 1 and is centered on 0.
x = np.arange(len(cases)) - (len(cases)-1)/2

# Loop over subplots in axs.
for a, ax in enumerate(axs):
    ax.set_title(motions[a], y=1.0, pad=-14)
    ax.set_ylabel('RMSE (deg)')
    ax.set_xticks(np.arange(len(xtick_labels)))
    if a == len(axs)-1:
        axs[a].set_xticklabels(xtick_labels)
    else:
        axs[a].set_xticklabels([])        
    
    # For each field in means_RMSEs['all'], plot bars with the values of the field for each idx_setups.
    for i, field in enumerate(xtick_labels):
        for j, idx_setup in enumerate(cases):
            ax.bar(i+x[j]*bar_width, means_RMSEs_copy[motions[a]][field][j], bar_width, color=colors[j])
            # Add text with the value of the bar.
            # if i == len(xtick_labels)-1:
            if field in xtick_values:
                ax.text(i+x[j]*bar_width, means_RMSEs_copy[motions[a]][field][j], np.round(means_RMSEs_copy[motions[a]][field][j], 1), ha='center', va='bottom')    
    # Add legend with idx_setups
    # leg_t = [setups[idx_setup] for idx_setup in idx_setups]
    leg_t = [setups_t[idx_setup] for idx_setup, _ in enumerate(cases)]
    # Get what is after the last '_' in leg_t.
    # leg_t = [leg_t[i].split('_')[-1] for i in range(len(leg_t))]
    # ax.legend(leg_t)
    # Make legend horizontal and top left
    if a == 0:
        ax.legend(leg_t, loc='upper left', bbox_to_anchor=(0, 1.2), ncol=len(leg_t), frameon=False)
    plt.show()

    # Use same y-limits for all subplots.
    ax.set_ylim([0, 10])
    # Use 3 y-ticks [0, 5, 10]
    ax.set_yticks(np.arange(0, 15, 5))
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