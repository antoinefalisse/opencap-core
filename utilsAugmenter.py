import os
import numpy as np
import utilsDataman
import copy
import tensorflow as tf
from utils import TRC2numpy
import json

from scipy.interpolate import interp1d

# TODO: not convinced this is robust. Hard to be able to upsamplde and downsample
# and keep the same time vector. Might need to be more careful with this.
def resample_array(array, target_hz):
    
    data = array[:, 1:]
    time_vector = array[:, 0]    
    original_hz = np.round(1 / np.mean(np.diff(time_vector)), 2)
    
    # Adjust time vector to correct for TRC rounding.    
    start = time_vector[0]
    num_values = time_vector.shape[0]
    step = 1/original_hz    
    time_vector_adj = np.linspace(start, start + (num_values - 1) * step, num=num_values)

    # Calculate the target time vector within the range of the original time vector    
    target_time_vector = np.arange(time_vector_adj[0], time_vector_adj[-1] + 1/target_hz, 1/target_hz)
    if target_time_vector[-1] > time_vector_adj[-1]:
        target_time_vector = target_time_vector[:-1]        
    if target_time_vector[-1] != time_vector_adj[-1]:
        raise ValueError('Issue')
        
    # Interpolate the data at the target time vector
    interp_func = interp1d(time_vector_adj, data, axis=0)
    resampled_data = interp_func(target_time_vector)

    # Create the resampled array
    resampled_array = np.column_stack((target_time_vector, resampled_data))

    return resampled_array

def augmentTRC(pathInputTRCFile, subject_mass, subject_height,
               pathOutputTRCFile, augmenterDir, augmenterModelName="LSTM",
               augmenter_model='v0.7', offset=True):
    
    # This is by default - might need to be adjusted in the future.
    featureHeight = True
    featureWeight = True
    
    # Augmenter types
    if augmenter_model == 'v0.0':
        from utils import getOpenPoseMarkers_fullBody
        feature_markers_full, response_markers_full = getOpenPoseMarkers_fullBody()         
        augmenterModelType_all = [augmenter_model]
        feature_markers_all = [feature_markers_full]
        response_markers_all = [response_markers_full]            
    elif augmenter_model == 'v0.1' or augmenter_model == 'v0.2' or augmenter_model == 'v0.20' or augmenter_model == 'v0.21' or augmenter_model == 'v0.22':
        # Lower body           
        augmenterModelType_lower = '{}_lower'.format(augmenter_model)
        from utils import getOpenPoseMarkers_lowerExtremity
        feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity()
        # Upper body
        augmenterModelType_upper = '{}_upper'.format(augmenter_model)
        from utils import getMarkers_upperExtremity_noPelvis
        feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis()        
        augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
        feature_markers_all = [feature_markers_lower, feature_markers_upper]
        response_markers_all = [response_markers_lower, response_markers_upper]
    # elif augmenter_model == 'v0.3' or augmenter_model == 'v0.4' or augmenter_model == 'v0.5' or augmenter_model == 'v0.6' or augmenter_model == 'v0.7' or augmenter_model == 'v0.8' or augmenter_model == 'v0.9' or augmenter_model == 'v0.10':
    else:
        if augmenter_model == 'v0.13':
            # Lower body           
            augmenterModelType_lower = '{}_lower'.format(augmenter_model)
            from utils import getOpenPoseMarkers_lowerExtremity4
            feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity4()
            # Feet          
            augmenterModelType_feet = '{}_feet'.format(augmenter_model)
            from utils import getOpenPoseMarkers_feet
            feature_markers_feet, response_markers_feet = getOpenPoseMarkers_feet()
             # Upper body
            augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            from utils import getMarkers_upperExtremity_noPelvis2
            feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()        
            augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_feet, augmenterModelType_upper]
            feature_markers_all = [feature_markers_lower, feature_markers_feet, feature_markers_upper]
            response_markers_all = [response_markers_lower, response_markers_feet, response_markers_upper]
        elif augmenter_model == 'v0.44':
            # Lower body           
            augmenterModelType_lower = '{}_lower'.format(augmenter_model)
            from utils import getOpenPoseMarkers_lowerExtremity5
            feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity5()
            # Feet          
            augmenterModelType_feet = '{}_feet'.format(augmenter_model)
            from utils import getOpenPoseMarkers_feet
            feature_markers_feet, response_markers_feet = getOpenPoseMarkers_feet()
             # Upper body
            augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            from utils import getMarkers_upperExtremity_noPelvis2
            feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()        
            augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_feet, augmenterModelType_upper]
            feature_markers_all = [feature_markers_lower, feature_markers_feet, feature_markers_upper]
            response_markers_all = [response_markers_lower, response_markers_feet, response_markers_upper]
        else:
            if augmenter_model == 'v0.12' or augmenter_model == 'v0.14':
                # Lower body           
                augmenterModelType_lower = '{}_lower'.format(augmenter_model)
                from utils import getOpenPoseMarkers_lowerExtremity3
                feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity3()
            else:
                # Lower body           
                augmenterModelType_lower = '{}_lower'.format(augmenter_model)
                from utils import getOpenPoseMarkers_lowerExtremity2
                feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity2()
            # Upper body
            augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            from utils import getMarkers_upperExtremity_noPelvis2
            feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()        
            augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
            feature_markers_all = [feature_markers_lower, feature_markers_upper]
            response_markers_all = [response_markers_lower, response_markers_upper]
    # else:
    #     raise ValueError('Augmenter model not recognized.')
    
    print('Augmenter model: {} - {}'.format(augmenterModelName, augmenter_model))

    if augmenter_model == 'v0.57' or augmenter_model == 'v0.58' or augmenter_model == 'v0.60':
        upsampling = True
        if augmenter_model == 'v0.57':
            upsample_sf = 240
        elif augmenter_model == 'v0.58' or augmenter_model == 'v0.60':
            upsample_sf = 120
    else:
        upsampling = False
    
    # %% Process data.
    # Import TRC file
    trc_file = utilsDataman.TRCFile(pathInputTRCFile)   
    
    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    outputs_all = {}
    n_response_markers_all = 0
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        outputs_all[idx_augm] = {}
        feature_markers = feature_markers_all[idx_augm]
        response_markers = response_markers_all[idx_augm]
        
        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
        
        # %% Pre-process inputs.
        # Step 1: import .trc file with OpenPose marker trajectories.  
        trc_data = TRC2numpy(pathInputTRCFile, feature_markers)        
        if upsampling:
            trc_data_sf = resample_array(trc_data, upsample_sf)
        else:
            trc_data_sf = trc_data  
        trc_data_data = trc_data_sf[:,1:]
        
        # Step 2: Normalize with reference marker position.
        with open(os.path.join(augmenterModelDir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        referenceMarker = metadata['reference_marker']
        referenceMarker_data = trc_file.marker(referenceMarker)
        if upsampling:      
            referenceTime = trc_file.time
            referenceMarker_data_in = np.concatenate((referenceTime[:,None], 
                                                      referenceMarker_data), axis=1)            
            referenceMarker_data_all = resample_array(referenceMarker_data_in, upsample_sf)
            time_sf = referenceMarker_data_all[:,0]
            referenceMarker_data_sf = referenceMarker_data_all[:,1:]
        else:
            referenceMarker_data_sf = referenceMarker_data   
        norm_trc_data_data = np.zeros((trc_data_data.shape[0],
                                       trc_data_data.shape[1]))
        for i in range(0,trc_data_data.shape[1],3):
            norm_trc_data_data[:,i:i+3] = (trc_data_data[:,i:i+3] - 
                                           referenceMarker_data_sf)
            
        # Step 3: Normalize with subject's height.
        norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
        norm2_trc_data_data = norm2_trc_data_data / subject_height
        
        # Step 4: Add remaining features.
        inputs = copy.deepcopy(norm2_trc_data_data)
        if featureHeight:    
            inputs = np.concatenate(
                (inputs, subject_height*np.ones((inputs.shape[0],1))), axis=1)
        if featureWeight:    
            inputs = np.concatenate(
                (inputs, subject_mass*np.ones((inputs.shape[0],1))), axis=1)
            
        # Step 5: Pre-process data
        pathMean = os.path.join(augmenterModelDir, "mean.npy")
        pathSTD = os.path.join(augmenterModelDir, "std.npy")
        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean
        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std 
            
        # Step 6: Reshape inputs if necessary (eg, LSTM)
        if augmenterModelName == "LSTM":
            inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
            
        # %% Load model and weights, and predict outputs.
        if augmenterModelName == "LSTM":
            json_file = open(os.path.join(augmenterModelDir, "model.json"), 'r')
            pretrainedModel_json = json_file.read()
            json_file.close()
            model = tf.keras.models.model_from_json(pretrainedModel_json)
            model.load_weights(os.path.join(augmenterModelDir, "weights.h5"))  
            outputs = model.predict(inputs)
        elif augmenterModelName == "Transformer":
            print('Pre-loading {}.'.format(augmenterModelDir))
            augmenter_instance_reloaded = tf.saved_model.load(augmenterModelDir)
            print('Predicting outputs.')
            outputs_temp = augmenter_instance_reloaded(inputs)
            print('Done predicting outputs.')
            outputs = outputs_temp.numpy()
        elif augmenterModelName == "Linear":
            json_file = open(os.path.join(augmenterModelDir, "model.json"), 'r')
            pretrainedModel_json = json_file.read()
            json_file.close()
            model = tf.keras.models.model_from_json(pretrainedModel_json)
            model.load_weights(os.path.join(augmenterModelDir, "weights.h5"))  
            outputs = model.predict(inputs)
        
        # %% Post-process outputs.
        # Step 1: Reshape if necessary (eg, LSTM)
        if augmenterModelName == "LSTM":
            outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))
            
        # Step 2: Un-normalize with subject's height.
        unnorm_outputs = outputs * subject_height
        
        # Step 2: Un-normalize with reference marker position.
        unnorm2_outputs = np.zeros((unnorm_outputs.shape[0],
                                    unnorm_outputs.shape[1]))
        for i in range(0,unnorm_outputs.shape[1],3):
            unnorm2_outputs[:,i:i+3] = (unnorm_outputs[:,i:i+3] + 
                                        referenceMarker_data_sf)
            
        if upsampling:     
            unnorm2_outputs_in = np.concatenate((time_sf[:,None],   
                                                   unnorm2_outputs), axis=1)
            # TODO: select sf_resample based on sf data           
            unnorm3_outputs = resample_array(unnorm2_outputs_in, 60)[:,1:]
        else:
            unnorm3_outputs = unnorm2_outputs            
            
        # %% Add markers to .trc file.
        for c, marker in enumerate(response_markers):
            x = unnorm3_outputs[:,c*3]
            y = unnorm3_outputs[:,c*3+1]
            z = unnorm3_outputs[:,c*3+2]
            trc_file.add_marker(marker, x, y, z)
            
        # %% Gather data for computing minimum y-position.
        outputs_all[idx_augm]['response_markers'] = response_markers   
        outputs_all[idx_augm]['response_data'] = unnorm3_outputs
        n_response_markers_all += len(response_markers)
        
    # %% Extract minimum y-position across response markers. This is used
    # to align feet and floor when visualizing.
    responses_all_conc = np.zeros((unnorm3_outputs.shape[0],
                                   n_response_markers_all*3))
    idx_acc_res = 0
    for idx_augm in outputs_all:
        idx_acc_res_end = (idx_acc_res + 
                           (len(outputs_all[idx_augm]['response_markers']))*3)
        responses_all_conc[:,idx_acc_res:idx_acc_res_end] = (
            outputs_all[idx_augm]['response_data'])
        idx_acc_res = idx_acc_res_end
    # Minimum y-position across response markers.
    min_y_pos = np.min(responses_all_conc[:,1::3])
        
    # %% If offset
    if offset:
        trc_file.offset('y', -(min_y_pos-0.01))
        
    # %% Return augmented .trc file   
    trc_file.write(pathOutputTRCFile)
    
    return min_y_pos
