'''
    This script takes both video and mocap TRC and GRF files (mocap only) and
    runs through scaling, IK, and ID (mocap only).
'''

import os
import sys
sys.path.append("..") # utilities in child directory
sys.path.append("../opensimPipeline") # utilities in opensimPipeline directory
import numpy as np
import shutil

from utils import importMetadata, getDataDirectory, storage2df, storage2numpy
from utilsOpenSim import runScaleTool, runIKTool, getScaleTimeRange, runIDTool

# runMocap = False
# runVideoAugmenter = True
# runVideoPose = False

# subjects = ['subject' + str(i) for i in range(2,12)]
# poseDetectors = ['OpenPose_1x1008_4scales']
# cameraSetups = ['5-cameras', '3-cameras', '2-cameras']
# augmenterTypes = ['v0.7']

# %% Process data.
# scriptDir = os.getcwd()
# repoDir = os.path.dirname(scriptDir)
# mainDir = getDataDirectory(False)
# dataDir = os.path.join(mainDir, 'Data')
# opensimPipelineDir = os.path.join(repoDir, 'opensimPipeline')
# for subjectName in subjects:

def runOpenSimPipeline(dataDir, opensimPipelineDir, subjectName, poseDetectors, cameraSetups, augmenterTypes, runMocap=False, runVideoAugmenter=True, runVideoPose=False, withTrackingMarkers=True, allVideoOnly=False):

    # Filter frequencies for ID.
    filterFrequencies = {'walking': 12, 'default':30}

    # Should be False. True was for a sensitivity analysis.
    fixed_markers = False # False should be default (better results) 

    # %% Setup filenames.
    genericModel4ScalingName = 'LaiArnoldModified2017_poly_withArms_weldHand.osim'
    genericSetupFile4ScalingNameMocap = 'Setup_scaling_RajagopalModified2016_withArms_KA_Mocap.xml'
    genericSetupFile4ScalingNameVideo = 'Setup_scaling_RajagopalModified2016_withArms_KA.xml'
    genericSetupFile4ScalingNameVideoOpenPose = 'Setup_scaling_RajagopalModified2016_withArms_KA_openpose.xml'
    genericSetupFile4ScalingNameVideoMMpose = 'Setup_scaling_RajagopalModified2016_withArms_KA_mmpose.xml'
    genericSetupFile4IKNameMocap = 'Setup_IK_Mocap.xml'
    if withTrackingMarkers:
        genericSetupFile4IKNameVideo = 'Setup_IK.xml'
    else:
        genericSetupFile4IKNameVideo = 'Setup_IK_augmenter_noTracking.xml'
    genericSetupFile4IKNameVideoOpenPose = 'Setup_IK_openpose.xml'
    genericSetupFile4IKNameVideoMMpose = 'Setup_IK_mmpose.xml'
    genericSetupFile4IDName = 'Setup_ID.xml'
    genericSetupFile4ExternalLoadsName = 'Setup_ExternalLoads.xml'

    # %% Force headers.
    headers_force = [
        'R_ground_force_vx', 'R_ground_force_vy', 'R_ground_force_vz', 
        'R_ground_force_px', 'R_ground_force_py', 'R_ground_force_pz',
        'R_ground_torque_x', 'R_ground_torque_y', 'R_ground_torque_z',
        'L_ground_force_vx', 'L_ground_force_vy', 'L_ground_force_vz',
        'L_ground_force_px', 'L_ground_force_py', 'L_ground_force_pz',
        'L_ground_torque_x', 'L_ground_torque_y', 'L_ground_torque_z']

    print('\nProcessing {}'.format(subjectName))
    markerDir = os.path.join(dataDir, 'Data', subjectName, 'MarkerData')
    forceDir = os.path.join(dataDir, 'Data', subjectName, 'ForceData')
    mocapDir = os.path.join(markerDir, 'Mocap')
    videoDir = os.path.join(markerDir, 'Video')
    if fixed_markers:
        osDir = os.path.join(dataDir, 'Data', subjectName, 'OpenSimData_fixed')
    else:
        osDir = os.path.join(dataDir, 'Data', subjectName, 'OpenSimData')
    
    sessionMetadata = importMetadata(os.path.join(dataDir, 'Data', subjectName,
                                                  'sessionMetadata.yaml'))
    
    # %% Mocap
    if runMocap:
        
        # Scaling
        pathTRCFile4Scaling = os.path.join(mocapDir, 'static1.trc')
        if not os.path.exists(pathTRCFile4Scaling):
            raise ValueError("no scaling file")
            
        scaledModelName = genericModel4ScalingName[:-5] + '_scaled'
        outputScaledModelDir = os.path.join(
            osDir, 'Mocap', 'Model', genericModel4ScalingName[:-5])
        pathScaledModel = os.path.join(outputScaledModelDir,
                                        scaledModelName + '.osim')
        pathGenericModel4Scaling = os.path.join(
            opensimPipelineDir, 'Models', genericModel4ScalingName)
        pathGenericSetupFile4Scaling = os.path.join(
            opensimPipelineDir, 'Scaling', genericSetupFile4ScalingNameMocap)
        os.makedirs(outputScaledModelDir, exist_ok=True) 
        
        timeRange4Scaling = getScaleTimeRange(
            pathTRCFile4Scaling, thresholdPosition=0.007,
            thresholdTime=0.1, isMocap=True)
            
        runScaleTool(pathGenericSetupFile4Scaling,
                      pathGenericModel4Scaling, sessionMetadata['mass_kg'],
                      pathTRCFile4Scaling, timeRange4Scaling, 
                      outputScaledModelDir,
                      subjectHeight=sessionMetadata['height_m'],
                      fixed_markers=fixed_markers)
        
        # IK
        for TRCFile4IKName in os.listdir(mocapDir):
            pathTRCFile4IK = os.path.join(mocapDir, TRCFile4IKName)
            
            if 'static1' in pathTRCFile4IK:
                continue            
            if not TRCFile4IKName[-3:] == 'trc':
                continue
            
            pathOutputFolder4IK = os.path.join(
                osDir, 'Mocap', 'IK', genericModel4ScalingName[:-5])
            
            # For the DJs, the time interval for IK is based on the GRFs.
            # Load force data
            if 'DJ' in TRCFile4IKName:
                pathForceFile = os.path.join(forceDir, TRCFile4IKName[:-4] + 
                                             '_forces.mot')
                forceData = storage2df(pathForceFile, headers_force)
                # Select all vertical force vectors.
                verticalForces = np.concatenate(( 
                    np.expand_dims(forceData['R_ground_force_vy'].to_numpy(), axis=1),
                    np.expand_dims(forceData['L_ground_force_vy'].to_numpy(), axis=1)), axis=1)
                sumVerticalForces = np.sum(verticalForces, axis=1)
                diffForces = np.diff(sumVerticalForces)
                startIdx = np.argwhere(diffForces)[0][0] + 1
                endIdx = np.argwhere(diffForces[startIdx-1:] == 0)[0][0] + startIdx - 2
                startTime = np.round(forceData.iloc[startIdx]['time'], 2)
                endTime = np.round(forceData.iloc[endIdx]['time'], 2)                
                startTime_adj = startTime - 0.3
                endTime_adj = endTime + 0.03                
                timeRange4IK = [startTime_adj, endTime_adj]                
            else:
                timeRange4IK = [] # leave empty to select entire trial    
            
            os.makedirs(pathOutputFolder4IK, exist_ok=True)          
            pathGenericSetupFile4IK = os.path.join(
                opensimPipelineDir, 'IK', genericSetupFile4IKNameMocap)
            
            runIKTool(pathGenericSetupFile4IK, pathScaledModel, 
                      pathTRCFile4IK, pathOutputFolder4IK,
                      timeRange=timeRange4IK)
            
            
        # ID
        for MOTFile4IDName in os.listdir(pathOutputFolder4IK):
            
            if not MOTFile4IDName[-3:] == 'mot':
                continue
        
            timeRange4ID = [] # leave empty to select time range from IK
            trialName = MOTFile4IDName[:-4]
            forcesFileName = trialName + '_forces.mot'
            lowpassCutoffFrequency = [
                val for key,val in filterFrequencies.items() 
                if key.lower() in trialName.lower()]
            if not lowpassCutoffFrequency:
                lowpassCutoffFrequency = filterFrequencies['default']
            else:
                lowpassCutoffFrequency = lowpassCutoffFrequency[0]
            print('Lowpass cutoff frequency = {}'.format(
                lowpassCutoffFrequency))
        
            pathOutputFolder4ID = os.path.join(osDir, 'Mocap', 'ID',
                                           genericModel4ScalingName[:-5]) 
            os.makedirs(pathOutputFolder4ID, exist_ok=True)            
            pathIKFile = os.path.join(pathOutputFolder4IK, trialName + '.mot')            
            pathGRFFile = os.path.join(forceDir, forcesFileName)                      
            pathGenericSetupFile4ID = os.path.join(opensimPipelineDir, 'ID',
                                                   genericSetupFile4IDName)
            pathGenericSetupFile4EL = os.path.join(
                opensimPipelineDir, 'ID', genericSetupFile4ExternalLoadsName) 

            if not timeRange4ID:
                ik_file = storage2numpy(pathIKFile)
                time_ik = ik_file['time']
                timeRange4ID = [time_ik[0], time_ik[-1]]
            
            runIDTool(pathGenericSetupFile4ID, pathGenericSetupFile4EL,
                      pathGRFFile, pathScaledModel, pathIKFile, timeRange4ID,
                      pathOutputFolder4ID, IKFileName=trialName,
                      filteringFrequency=lowpassCutoffFrequency)            
        
    # %% Video - augmenter
    if runVideoAugmenter:
        
        for poseDetector in poseDetectors:
            poseDetectorDir = os.path.join(videoDir, poseDetector)
            for cameraSetup in cameraSetups:            
                cameraSetupDir = os.path.join(poseDetectorDir, cameraSetup)
                
                for augmenterType in augmenterTypes:
                    
                
                    
                    if allVideoOnly:
                        augmenterTypeDir = os.path.join(cameraSetupDir, augmenterType + '_allVideoOnly')
                        
                        # Scaling
                        # We do not want to re-run scaling here, but copy the
                        # model from allVideoOnly=False. The rationale is that
                        # we are just interested in generating the full video motion
                        # and not just the part synced with mocap, that way we
                        # can generate simualtions with start and end buffers,
                        # which should provide better results in the part of interest.
                        # We want to use the same model as the one we use for
                        # other analyses (RMSE on joint angles) and therefore
                        # here copy the scaled model from the other directory.
                        refScaledModelDir = os.path.join(
                            osDir, 'Video', poseDetector, cameraSetup, augmenterType,
                            'Model', genericModel4ScalingName[:-5])
                        pathGenericModel4Scaling = os.path.join(
                            opensimPipelineDir, 'Models', genericModel4ScalingName)
                        _, scaledModelNameA = os.path.split(pathGenericModel4Scaling)
                        scaledModelName = scaledModelNameA[:-5] + "_scaled"
                        pathScaledModel = os.path.join(refScaledModelDir,
                                                        scaledModelName + '.osim')
                        scaledModelDir = os.path.join(
                            osDir, 'Video', poseDetector, cameraSetup, augmenterType + '_allVideoOnly',
                            'Model', genericModel4ScalingName[:-5])
                        os.makedirs(scaledModelDir, exist_ok=True)
                        pathScaledModelEnd = os.path.join(scaledModelDir,
                                                        scaledModelName + '.osim')
                        shutil.copy2(pathScaledModel, pathScaledModelEnd)
                            
                        # IK
                        for TRCFile4IKName in os.listdir(augmenterTypeDir):
                            pathTRCFile4IK = os.path.join(augmenterTypeDir,
                                                          TRCFile4IKName)
                            print("Processing: {}".format(pathTRCFile4IK))
                            
                            if 'static1' in pathTRCFile4IK:
                                continue
                            
                            if not TRCFile4IKName[-3:] == 'trc':
                                continue
                            
                            timeRange4IK = [] # leave empty to select entire trial
                            
                            pathOutputFolder4IK = os.path.join(
                                osDir, 'Video', poseDetector, cameraSetup, augmenterType + '_allVideoOnly',
                                'IK', genericModel4ScalingName[:-5])
                            
                            os.makedirs(pathOutputFolder4IK, exist_ok=True)          
                            pathGenericSetupFile4IK = os.path.join(
                                opensimPipelineDir, 'IK', genericSetupFile4IKNameVideo)
                            
                            if os.path.exists(pathScaledModelEnd):
                                runIKTool(pathGenericSetupFile4IK, pathScaledModelEnd, 
                                          pathTRCFile4IK, pathOutputFolder4IK,
                                          timeRange=timeRange4IK)
                            else:
                                continue                            
                            
                    else:
                        
                        augmenterTypeDir = os.path.join(cameraSetupDir, augmenterType)
                        
                        # Scaling
                        pathTRCFile4Scaling = os.path.join(augmenterTypeDir, 
                                                            'static1_videoAndMocap.trc')
                        if not os.path.exists(pathTRCFile4Scaling):
                            raise ValueError("no scaling file")
                            
                        scaledModelName = genericModel4ScalingName[:-5] + '_scaled'
                        outputScaledModelDir = os.path.join(
                            osDir, 'Video', poseDetector, cameraSetup, augmenterType,
                            'Model', genericModel4ScalingName[:-5])
                        pathScaledModel = os.path.join(outputScaledModelDir,
                                                        scaledModelName + '.osim')
                        pathGenericModel4Scaling = os.path.join(
                            opensimPipelineDir, 'Models', genericModel4ScalingName)
                        pathGenericSetupFile4Scaling = os.path.join(
                            opensimPipelineDir, 'Scaling',
                            genericSetupFile4ScalingNameVideo)
                        os.makedirs(outputScaledModelDir, exist_ok=True) 
                        
                        if poseDetector == 'OpenPose_default' and cameraSetup == '5-cameras' and augmenterType == 'v0.45':
                            thresholdPosition = 0.008
                        else:
                            thresholdPosition = 0.007
                        thresholdTime = 0.1
                        timeRange4Scaling = getScaleTimeRange(
                            pathTRCFile4Scaling, thresholdPosition=thresholdPosition,
                            thresholdTime=thresholdTime, isMocap=False)
                            
                        if timeRange4Scaling:
                            print("Scaling model")
                            runScaleTool(pathGenericSetupFile4Scaling,
                                        pathGenericModel4Scaling, sessionMetadata['mass_kg'],
                                        pathTRCFile4Scaling, timeRange4Scaling, 
                                        outputScaledModelDir,
                                        subjectHeight=sessionMetadata['height_m'],
                                        fixed_markers=fixed_markers,
                                        withTrackingMarkers=withTrackingMarkers)                    
                    
                        # IK
                        for TRCFile4IKName in os.listdir(augmenterTypeDir):
                            pathTRCFile4IK = os.path.join(augmenterTypeDir,
                                                          TRCFile4IKName)
                            print("Processing: {}".format(pathTRCFile4IK))
                            
                            if 'static1' in pathTRCFile4IK:
                                continue
                            
                            if not TRCFile4IKName[-3:] == 'trc':
                                continue
                            
                            # For the DJs, the time interval for IK is based on the GRFs.
                            # Load force data
                            if 'DJ' in TRCFile4IKName:
                                pathForceFile = os.path.join(forceDir, TRCFile4IKName[:-18] + 
                                                              '_forces.mot')
                                forceData = storage2df(pathForceFile, headers_force)
                                # Select all vertical force vectors.
                                verticalForces = np.concatenate(( 
                                    np.expand_dims(forceData['R_ground_force_vy'].to_numpy(), axis=1),
                                    np.expand_dims(forceData['L_ground_force_vy'].to_numpy(), axis=1)), axis=1)
                                sumVerticalForces = np.sum(verticalForces, axis=1)
                                diffForces = np.diff(sumVerticalForces)
                                startIdx = np.argwhere(diffForces)[0][0] + 1
                                endIdx = np.argwhere(diffForces[startIdx-1:] == 0)[0][0] + startIdx - 2
                                startTime = np.round(forceData.iloc[startIdx]['time'], 2)
                                endTime = np.round(forceData.iloc[endIdx]['time'], 2)                        
                                startTime_adj = startTime - 0.3
                                endTime_adj = endTime + 0.03                        
                                timeRange4IK = [startTime_adj, endTime_adj]                        
                            else:
                                timeRange4IK = [] # leave empty to select entire trial
                            
                            pathOutputFolder4IK = os.path.join(
                                osDir, 'Video', poseDetector, cameraSetup, augmenterType,
                                'IK', genericModel4ScalingName[:-5])
                            
                            os.makedirs(pathOutputFolder4IK, exist_ok=True)          
                            pathGenericSetupFile4IK = os.path.join(
                                opensimPipelineDir, 'IK', genericSetupFile4IKNameVideo)
                            
                            if os.path.exists(pathScaledModel):
                                runIKTool(pathGenericSetupFile4IK, pathScaledModel, 
                                          pathTRCFile4IK, pathOutputFolder4IK,
                                          timeRange=timeRange4IK)
                            else:
                                continue
                        
    # %% Video - pose markers directly
    if runVideoPose:    
        for poseDetector in poseDetectors:
            poseDetectorDir = os.path.join(videoDir, poseDetector)
            for cameraSetup in cameraSetups:            
                cameraSetupDir = os.path.join(poseDetectorDir, cameraSetup)      
                # No need for augmented markers, they are w/ pose markers. 
                augmenterType = augmenterTypes[0]
                augmenterTypeDir = os.path.join(cameraSetupDir, augmenterType)        
                # Scaling
                pathTRCFile4Scaling = os.path.join(augmenterTypeDir, 
                                                   'static1_videoAndMocap.trc')
                if not os.path.exists(pathTRCFile4Scaling):
                    raise ValueError("no scaling file")
                    
                scaledModelName = genericModel4ScalingName[:-5] + '_scaled'
                outputScaledModelDir = os.path.join(
                    osDir, 'Video', poseDetector, cameraSetup, 'pose',
                    'Model', genericModel4ScalingName[:-5])
                pathScaledModel = os.path.join(outputScaledModelDir,
                                                scaledModelName + '.osim')
                pathGenericModel4Scaling = os.path.join(
                    opensimPipelineDir, 'Models', genericModel4ScalingName)
                if 'openpose' in poseDetector.lower():
                    pathGenericSetupFile4Scaling = os.path.join(
                        opensimPipelineDir, 'Scaling',
                        genericSetupFile4ScalingNameVideoOpenPose)
                elif 'mmpose' in poseDetector.lower():
                    pathGenericSetupFile4Scaling = os.path.join(
                        opensimPipelineDir, 'Scaling',
                        genericSetupFile4ScalingNameVideoMMpose)
                else:
                    raise ValueError("pose detector not supported")
                os.makedirs(outputScaledModelDir, exist_ok=True) 
                
                thresholdPosition = 0.007
                if poseDetector == 'OpenPose' and subjectName == 'subject2':
                    if cameraSetup == '5-cameras':
                        thresholdPosition = 0.008                        
                    elif cameraSetup == '3-cameras':
                        thresholdPosition = 0.023
                elif poseDetector == 'OpenPose_generic' and subjectName == 'subject2':
                    if cameraSetup == '5-cameras':
                        thresholdPosition = 0.021
                    elif cameraSetup == '3-cameras':
                        thresholdPosition = 0.009
                    elif cameraSetup == '2-cameras':
                        thresholdPosition = 0.011
                elif poseDetector == 'OpenPose_generic' and subjectName == 'subject3':
                    if cameraSetup == '3-cameras':
                        thresholdPosition = 0.009
                    elif cameraSetup == '2-cameras':
                        thresholdPosition = 0.010
                elif poseDetector == 'OpenPose_generic' and subjectName == 'subject4':
                    if cameraSetup == '2-cameras':
                        thresholdPosition = 0.008
                elif poseDetector == 'OpenPose_generic' and subjectName == 'subject5':
                    if cameraSetup == '2-cameras':
                        thresholdPosition = 0.009
                elif poseDetector == 'OpenPose_generic' and subjectName == 'subject6':
                    if cameraSetup == '5-cameras':
                        thresholdPosition = 0.008
                    elif cameraSetup == '2-cameras':
                        thresholdPosition = 0.011
                elif poseDetector == 'OpenPose_generic' and subjectName == 'subject7':
                    if cameraSetup == '2-cameras':
                        thresholdPosition = 0.008
                timeRange4Scaling = getScaleTimeRange(
                    pathTRCFile4Scaling, thresholdPosition=thresholdPosition,
                    thresholdTime=0.1, withOpenPoseMarkers=True, isMocap=False)
                    
                if timeRange4Scaling:
                    runScaleTool(
                        pathGenericSetupFile4Scaling, pathGenericModel4Scaling, 
                        sessionMetadata['mass_kg'], pathTRCFile4Scaling, 
                        timeRange4Scaling, outputScaledModelDir,
                        subjectHeight=sessionMetadata['height_m'],
                        fixed_markers=fixed_markers)                    
                
                # IK
                for TRCFile4IKName in os.listdir(augmenterTypeDir):
                    pathTRCFile4IK = os.path.join(augmenterTypeDir,
                                                  TRCFile4IKName)
                    
                    if 'static1' in pathTRCFile4IK:
                        continue
                    
                    if not TRCFile4IKName[-3:] == 'trc':
                        continue
                    
                    # For the DJs, the time interval for IK is based on the GRFs.
                    # Load force data
                    if 'DJ' in TRCFile4IKName:
                        pathForceFile = os.path.join(forceDir, TRCFile4IKName[:-18] + 
                                                      '_forces.mot')
                        forceData = storage2df(pathForceFile, headers_force)
                        # Select all vertical force vectors.
                        verticalForces = np.concatenate(( 
                            np.expand_dims(forceData['R_ground_force_vy'].to_numpy(), axis=1),
                            np.expand_dims(forceData['L_ground_force_vy'].to_numpy(), axis=1)), axis=1)
                        sumVerticalForces = np.sum(verticalForces, axis=1)
                        diffForces = np.diff(sumVerticalForces)
                        startIdx = np.argwhere(diffForces)[0][0] + 1
                        endIdx = np.argwhere(diffForces[startIdx-1:] == 0)[0][0] + startIdx - 2
                        startTime = np.round(forceData.iloc[startIdx]['time'], 2)
                        endTime = np.round(forceData.iloc[endIdx]['time'], 2)                        
                        startTime_adj = startTime - 0.3
                        endTime_adj = endTime + 0.03                        
                        timeRange4IK = [startTime_adj, endTime_adj]                        
                    else:
                        timeRange4IK = [] # leave empty to select entire trial
                    
                    pathOutputFolder4IK = os.path.join(
                        osDir, 'Video', poseDetector, cameraSetup, 'pose',
                        'IK', genericModel4ScalingName[:-5])
                    
                    os.makedirs(pathOutputFolder4IK, exist_ok=True)
                    
                    if 'openpose' in poseDetector.lower():
                        pathGenericSetupFile4IK = os.path.join(
                            opensimPipelineDir, 'IK', 
                            genericSetupFile4IKNameVideoOpenPose)
                    elif 'mmpose' in poseDetector.lower():
                        pathGenericSetupFile4IK = os.path.join(
                            opensimPipelineDir, 'IK', 
                            genericSetupFile4IKNameVideoMMpose)
                    else:
                        raise ValueError("pose detector not supported")
                    
                    if os.path.exists(pathScaledModel):
                        runIKTool(pathGenericSetupFile4IK, pathScaledModel, 
                                  pathTRCFile4IK, pathOutputFolder4IK,
                                  timeRange=timeRange4IK)
                    else:
                        continue
