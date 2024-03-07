"""
    This script:
        Syncs mocap and video marker data.
        Computes MPJEs between mocap and video marker data and outputs the
        result in a report.
"""

# import 
import sys
import os
import glob
import csv

sys.path.append('./..')
import utils as ut
import utilsDataman as dm
import utilsChecker
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import copy

# %% Function definitions
def computeMarkerDifferences(trialName,mocapDir,videoTrcDir,markersMPJE,
    mocapOriginPath, r_fromMarker_toVideoOrigin_inLab,
    mocapFiltFreq, R_video_opensim, R_opensim_xForward, 
    saveProcessedMocapData=False, overwriteMarkerDataProcessed=False, 
    overwriteForceDataProcessed=False, overwritevideoAndMocap=False,
    overwritevideo=False, augmenterModelName='LSTM'):
    print(trialName)

    # Dict with frequencies for loading appropriate force data.
    filtFrequencies = {
        'walking': '_filt6Hz',
        'running': '_filt12Hz',
        'squats': '_filt4Hz',
        'STS': '_filt4Hz',
        'DJ': '_filt30Hz',
        'static': ''}
    
    # Mocap directory
    mocapTrcPath = os.path.join(mocapDir,trialName + '.trc')
    mocapTRC = dm.TRCFile(mocapTrcPath)
    mocapMkrNames = mocapTRC.marker_names
    mocapMkrNamesLower = [mkr.lower() for mkr in mocapMkrNames]
    mocapData = ut.TRC2numpy(mocapTrcPath,mocapMkrNames)[:,1:]
    mocapTime = mocapTRC.time
    
    r_fromLabOrigin_toVideoOrigin_inLab = (np.mean(
        ut.TRC2numpy(mocapOriginPath,['origin'])[:,1:],axis=0) + 
        r_fromMarker_toVideoOrigin_inLab) # add marker radius to y in mm
    
    # Force directory
    forceDir = os.path.join(os.path.dirname(mocapDir), 'ForceData')
    filt_suffix = None
    for motion_type in filtFrequencies:
        if motion_type in trialName:
            filt_suffix = filtFrequencies[motion_type]
    if filt_suffix == None:
        raise ValueError('motion_type not recognized')
    forceMotPath = os.path.join(forceDir,trialName + '_forces' + filt_suffix + '.mot')
    if '0001' in mocapDir:
        headers_force = ['R_ground_force_vx', 'R_ground_force_vy', 'R_ground_force_vz', 
                         'R_ground_force_px', 'R_ground_force_py', 'R_ground_force_pz',
                         'R_ground_torque_x', 'R_ground_torque_y', 'R_ground_torque_z',
                         'L_ground_force_vx', 'L_ground_force_vy', 'L_ground_force_vz',
                         'L_ground_force_px', 'L_ground_force_py', 'L_ground_force_pz',
                         'L_ground_torque_x', 'L_ground_torque_y', 'L_ground_torque_z',
                         '3_ground_force_vx', '3_ground_force_vy', '3_ground_force_vz',
                         '3_ground_force_px', '3_ground_force_py', '3_ground_force_pz',
                         '3_ground_torque_x', '3_ground_torque_y', '3_ground_torque_z']
    elif '0002' in mocapDir:
        headers_force = ['R_ground_force_vx', 'R_ground_force_vy', 'R_ground_force_vz', 
                         'R_ground_force_px', 'R_ground_force_py', 'R_ground_force_pz',
                         'R_ground_torque_x', 'R_ground_torque_y', 'R_ground_torque_z',
                         'L_ground_force_vx', 'L_ground_force_vy', 'L_ground_force_vz',
                         'L_ground_force_px', 'L_ground_force_py', 'L_ground_force_pz',
                         'L_ground_torque_x', 'L_ground_torque_y', 'L_ground_torque_z']
        
    forceData = ut.storage2df(forceMotPath, headers_force).to_numpy()[:,1:]
    forceTime = ut.storage2df(forceMotPath, headers_force).to_numpy()[:,0]
    
    # Video-trc directory
    videoTrcPath = os.path.join(videoTrcDir,trialName + '_{}.trc'.format(augmenterModelName))
    videoTRC = dm.TRCFile(videoTrcPath)
    videoMkrNames = videoTRC.marker_names
    videoMkrNamesLower = [mkr.lower() for mkr in videoMkrNames]
    videoData = ut.TRC2numpy(videoTrcPath,videoMkrNames)[:,1:]
    newTime = np.arange(videoTRC.time[0],np.round(videoTRC.time[-1]+1/mocapTRC.camera_rate,6),1/mocapTRC.camera_rate)
    vidInterpFxn = scipy.interpolate.interp1d(videoTRC.time,videoData,axis=0,fill_value='extrapolate')
    videoData = vidInterpFxn(newTime)
    if videoTRC.units == 'm': videoData = videoData*1000
    videoTime = newTime
    
    # Filter mocap data
    if mocapFiltFreq is not None:
        mocapData = utilsChecker.filter3DPointsButterworth(mocapData,mocapFiltFreq,mocapTRC.camera_rate,order=4)
        videoData = utilsChecker.filter3DPointsButterworth(videoData,mocapFiltFreq,mocapTRC.camera_rate,order=4)
        
    # Copy filtered mocap data
    mocapData_all = copy.deepcopy(mocapData)
    
    # Rotate camera data and subtract origin
    R_lab_opensim = R.from_euler('x',-90,degrees=True)
    r_labOrigin_videoOrigin_inOpensim = R_lab_opensim.apply(r_fromLabOrigin_toVideoOrigin_inLab)
    
    for iMkr in range(videoTRC.num_markers):
        videoData[:,iMkr*3:iMkr*3+3] = R_video_opensim.apply(videoData[:,iMkr*3:iMkr*3+3]) + r_labOrigin_videoOrigin_inOpensim
    
    # Select sync algorithm that minimizes the MPJEs
    markersSync = markersMPJE        
    lag_markerError_sumabs, success_sumabs = syncMarkerError(
        mocapData, videoData, markersSync, mocapMkrNamesLower, videoMkrNamesLower)
    lag_markerError_norm, success_norm = syncMarkerError(
        mocapData, videoData, markersSync, mocapMkrNamesLower, videoMkrNamesLower, method='norm')
    lag_verticalVelocity = syncVerticalVelocity(
        mocapData, videoData, mocapTRC, videoTRC)
    
    if success_sumabs:
        outputMPJE_markerError_abs = getMPJEs(lag_markerError_sumabs, trialName, videoTime, mocapTime, mocapTRC, mocapData, 
                     videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE)
    else:
        outputMPJE_markerError_abs = {}
        outputMPJE_markerError_abs['MPJE_offsetRemoved_mean'] = 1e6
    if success_norm:
        outputMPJE_markerError_norm = getMPJEs(lag_markerError_norm, trialName, videoTime, mocapTime, mocapTRC, mocapData, 
                     videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE)
    else:
        outputMPJE_markerError_norm = {}
        outputMPJE_markerError_norm['MPJE_offsetRemoved_mean'] = 1e6
    outputMPJE_verticalVelocity = getMPJEs(lag_verticalVelocity, trialName, videoTime, mocapTime, mocapTRC, mocapData, 
                 videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE)
    outputMPJE_all = np.array([outputMPJE_markerError_abs['MPJE_offsetRemoved_mean'],
                               outputMPJE_markerError_norm['MPJE_offsetRemoved_mean'],
                               outputMPJE_verticalVelocity['MPJE_offsetRemoved_mean']])    
    idx_min = np.argmin(outputMPJE_all)
    
    
    if idx_min == 0:
        MPJE_mean = outputMPJE_markerError_abs['MPJE_mean']
        MPJE_std = outputMPJE_markerError_abs['MPJE_std']
        MPJE_offsetRemoved_mean = outputMPJE_markerError_abs['MPJE_offsetRemoved_mean']
        MPJE_offsetRemoved_std = outputMPJE_markerError_abs['MPJE_offsetRemoved_std']
        MPJEvec = outputMPJE_markerError_abs['MPJEvec']
        MPJE_offVec = outputMPJE_markerError_abs['MPJE_offVec']
        videoDataOffsetRemoved = outputMPJE_markerError_abs['videoDataOffsetRemoved']
        syncTimeVec = outputMPJE_markerError_abs['syncTimeVec']
        mocapData = outputMPJE_markerError_abs['mocapData']
        videoData = outputMPJE_markerError_abs['videoData']
        videoDataAllOffsetRemoved = outputMPJE_markerError_abs['videoDataAllOffsetRemoved']
        timeVecVideoAll = outputMPJE_markerError_abs['timeVecVideoAll']
        videoDataAll = outputMPJE_markerError_abs['videoDataAll']
    elif idx_min == 1:
        MPJE_mean = outputMPJE_markerError_norm['MPJE_mean']
        MPJE_std = outputMPJE_markerError_norm['MPJE_std']
        MPJE_offsetRemoved_mean = outputMPJE_markerError_norm['MPJE_offsetRemoved_mean']
        MPJE_offsetRemoved_std = outputMPJE_markerError_norm['MPJE_offsetRemoved_std']
        MPJEvec = outputMPJE_markerError_norm['MPJEvec']
        MPJE_offVec = outputMPJE_markerError_norm['MPJE_offVec']
        videoDataOffsetRemoved = outputMPJE_markerError_norm['videoDataOffsetRemoved']
        syncTimeVec = outputMPJE_markerError_norm['syncTimeVec']
        mocapData = outputMPJE_markerError_norm['mocapData']
        videoData = outputMPJE_markerError_norm['videoData']
        videoDataAllOffsetRemoved = outputMPJE_markerError_norm['videoDataAllOffsetRemoved']
        timeVecVideoAll = outputMPJE_markerError_norm['timeVecVideoAll']
        videoDataAll = outputMPJE_markerError_abs['videoDataAll']
    elif idx_min == 2:
        MPJE_mean = outputMPJE_verticalVelocity['MPJE_mean']
        MPJE_std = outputMPJE_verticalVelocity['MPJE_std']
        MPJE_offsetRemoved_mean = outputMPJE_verticalVelocity['MPJE_offsetRemoved_mean']
        MPJE_offsetRemoved_std = outputMPJE_verticalVelocity['MPJE_offsetRemoved_std']
        MPJEvec = outputMPJE_verticalVelocity['MPJEvec']
        MPJE_offVec = outputMPJE_verticalVelocity['MPJE_offVec']
        videoDataOffsetRemoved = outputMPJE_verticalVelocity['videoDataOffsetRemoved']
        syncTimeVec = outputMPJE_verticalVelocity['syncTimeVec']
        mocapData = outputMPJE_verticalVelocity['mocapData']
        videoData = outputMPJE_verticalVelocity['videoData']
        videoDataAllOffsetRemoved = outputMPJE_verticalVelocity['videoDataAllOffsetRemoved']
        timeVecVideoAll = outputMPJE_verticalVelocity['timeVecVideoAll']
        videoDataAll = outputMPJE_verticalVelocity['videoDataAll']
        
    # write TRC with combined data
    outData = np.concatenate((mocapData,videoData,videoDataOffsetRemoved),axis=1)
    outVideoDataAll = np.concatenate((videoDataAll,videoDataAllOffsetRemoved),axis=1) 
    
    # rotate so x is always forward
    for iMkr in range(int(outData.shape[1]/3)):
        outData[:,iMkr*3:iMkr*3+3] = R_opensim_xForward.apply(outData[:,iMkr*3:iMkr*3+3])
        
    for iMkr in range(int(outVideoDataAll.shape[1]/3)):
        outVideoDataAll[:,iMkr*3:iMkr*3+3] = R_opensim_xForward.apply(outVideoDataAll[:,iMkr*3:iMkr*3+3])  
        
    # rotate original mocap data
    for iMkr in range(int(mocapData_all.shape[1]/3)):
        mocapData_all[:,iMkr*3:iMkr*3+3] = R_opensim_xForward.apply(mocapData_all[:,iMkr*3:iMkr*3+3])
        
    # rotate original force data (not needed) keep it here if needed later
    # for iMkr in range(int(forceData.shape[1]/3)):
    #     forceData[:,iMkr*3:iMkr*3+3] = R_opensim_xForward.apply(forceData[:,iMkr*3:iMkr*3+3])
    
    # videoMkrNames = [mkr + '_video' for mkr in videoMkrNames]
    videoMkrNamesNoOffset = [mkr + '_offsetRemoved' for mkr in videoMkrNames]
    outMkrNames = mocapMkrNames + videoMkrNames + videoMkrNamesNoOffset
    
    outputDir = os.path.join(videoTrcDir,'videoAndMocap')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    pathOutputFile = os.path.join(outputDir,trialName + '_videoAndMocap.trc')
    
    if not os.path.exists(pathOutputFile) or overwritevideoAndMocap:
        with open(pathOutputFile,"w") as f:
            ut.numpy2TRC(f, outData, outMkrNames, fc=mocapTRC.camera_rate, units="mm",t_start=syncTimeVec[0])

    # This is exporting the full synced video data, including the offset removed data. This is useful to
    # have a buffer when generating dynamics simulations.
    pathOutputFileVideoAll = os.path.join(outputDir,trialName + '_video.trc')
    if not os.path.exists(pathOutputFileVideoAll) or overwritevideo:
        with open(pathOutputFileVideoAll,"w") as f:
            ut.numpy2TRC(f, outVideoDataAll, videoMkrNames + videoMkrNamesNoOffset, fc=mocapTRC.camera_rate, units="mm",t_start=timeVecVideoAll[0])
      
    if saveProcessedMocapData and overwriteMarkerDataProcessed:
        outputDirMocap = os.path.join(os.path.dirname(mocapDir), 'MarkerDataProcessed')
        os.makedirs(outputDirMocap, exist_ok=True)
        pathOutputFileMocap = os.path.join(outputDirMocap,trialName + '.trc')
        with open(pathOutputFileMocap,"w") as f:    
            ut.numpy2TRC(f, mocapData_all, mocapMkrNames, fc=mocapTRC.camera_rate, units="mm",t_start=mocapTime[0])
            
    if saveProcessedMocapData and overwriteForceDataProcessed:
        outputDirMocapF = os.path.join(os.path.dirname(mocapDir), 'ForceDataProcessed')
        os.makedirs(outputDirMocapF, exist_ok=True)        
        pathOutputFileMocapF = os.path.join(outputDirMocapF,trialName + '_forces.mot')
        labels = ['time'] + headers_force
        forceData_all = np.concatenate((np.expand_dims(forceTime,axis=1), forceData), axis=1)        
        ut.numpy2storage(labels, forceData_all, pathOutputFileMocapF)
        
    
    
    return MPJE_mean, MPJE_std, MPJE_offsetRemoved_mean, MPJE_offsetRemoved_std, MPJEvec, MPJE_offVec

def writeMPJE(trialNames, videoTrcDir, 
              MPJE_mean, MPJE_std, MPJEvec, 
              MPJE_offsetRemoved_mean, MPJE_offsetRemoved_std, MPJE_offVec,
              headers):
    os.makedirs(videoTrcDir, exist_ok=True)
    with open(os.path.join(videoTrcDir,'MPJE.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        headers_all = ['trial'] + headers + ['mean'] + ['std']       
        _ = csvWriter.writerow(headers_all)
        _ = csvWriter.writerow([''])
        _ = csvWriter.writerow(['With offset'])
        for idxTrial,tName in enumerate(trialNames):            
            MPJErow = [tName]
            for c_m, marker in enumerate(headers):
                MPJErow.extend(['%.2f' %MPJEvec[idxTrial,c_m]])
            MPJErow.extend(['%.2f' %MPJE_mean[idxTrial], '%.2f' %MPJE_std[idxTrial]])
            _ = csvWriter.writerow(MPJErow)
        
        _ = csvWriter.writerow([''])
        _ = csvWriter.writerow(['Without offset'])
        for idxTrial,tName in enumerate(trialNames):            
            MPJErow = [tName]
            for c_m, marker in enumerate(headers):
                MPJErow.extend(['%.2f' %MPJE_offVec[idxTrial,c_m]])
            MPJErow.extend(['%.2f' %MPJE_offsetRemoved_mean[idxTrial], '%.2f' %MPJE_offsetRemoved_std[idxTrial]])
            _ = csvWriter.writerow(MPJErow)
            
def writeMPJE_perSession(trialNames,outputDir,MPJE_session,
                         MPJE_offsetRemoved_session,analysisNames,
                         csv_name):
    with open(os.path.join(outputDir,csv_name + '.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        topRow = ['trial']
        for label in analysisNames:
            topRow.extend([label,'',''])
        _ = csvWriter.writerow(topRow)
        secondRow = ['']
        secondRow.extend(["MPJE", "offsetRmvd",''] * len(analysisNames))
        _ = csvWriter.writerow(secondRow)
        for idxTrial,tName in enumerate(trialNames):
            MPJErow = [tName]
            for MPJE,MPJE_offsetRemoved in zip(MPJE_session,MPJE_offsetRemoved_session):
                MPJErow.extend(['%.2f' %MPJE[idxTrial], '%.2f' %MPJE_offsetRemoved[idxTrial], ''])
            _ = csvWriter.writerow(MPJErow)
            
def syncMarkerError(mocapData, videoData, markersSync, mocapMkrNamesLower, 
                    videoMkrNamesLower, method='sumabs'):
    
    mocapSubset = np.zeros((mocapData.shape[0],len(markersSync)*3))
    videoSubset = np.zeros((videoData.shape[0],len(markersSync)*3))
    for i,mkr in enumerate(markersSync):
        idxM = mocapMkrNamesLower.index(mkr.lower())
        idxV = videoMkrNamesLower.index(mkr.lower() + '_study')
        mocapSubset[:,i*3:i*3+3] = mocapData[:,idxM*3:idxM*3+3]
        videoSubset[:,i*3:i*3+3] = videoData[:,idxV*3:idxV*3+3]
        
    if mocapSubset.shape[0] >= videoSubset.shape[0]:
        lag = 0
        success  = False
        return lag, success
        
    mkrDist = np.zeros(videoSubset.shape[0]-mocapSubset.shape[0])
    nMocapSamps = mocapSubset.shape[0]
    i=0
    while i<len(mkrDist): # this assumes that the video is always longer than the mocap        
        # Use Euclidian instead
        if method=='norm':
            mkrNorm = np.zeros((mocapSubset.shape[0],len(markersSync)))
            for j in range(0,len(markersSync)):                
                mkrNorm[:,j] = np.linalg.norm(videoSubset[i:i+nMocapSamps,j*3:(j+1)*3] - mocapSubset[:,j*3:(j+1)*3], axis=1)
            mkrDist[i] = np.sum(np.mean(mkrNorm,axis=1))
        elif method=='sumabs':
            mkrDist[i] = np.sum(np.abs(videoSubset[i:i+nMocapSamps]-mocapSubset))
        i+=1   
    lag = -np.argmin(mkrDist)
    success = True
    # if (lag == 0 or np.min(mkrDist)/(len(markersSync)*mocapSubset.shape[0]) > 75):
    #     success = False
        
    return lag, success
    
def syncVerticalVelocity(mocapData, videoData, mocapTRC, videoTRC):
    
    mocapY = mocapData[:,np.arange(1,mocapTRC.num_markers*3,3)]
    videoYInds = np.arange(1,videoTRC.num_markers*3,3).tolist()
    # del videoYInds[15:18] # delete face markers
    # del videoYInds[0]
    # TODO: might make sense to only select augmented markers...
    # if videoTRC.num_markers == 63:
    #     videoYInds = videoYInds[23:]
    # else:
    #     raise ValueError("Assumption about number of marker is wrong")        
    videoY = videoData[:,videoYInds]
    d_mocapY = np.diff(mocapY,axis=0)
    d_videoY = np.diff(videoY,axis=0)
    d_mocapY_sum = np.sum(d_mocapY,axis=1) / np.max(np.sum(d_mocapY,axis=1))
    d_videoY_sum = np.sum(d_videoY,axis=1) / np.max(np.sum(d_videoY,axis=1))
    
    corVal,lag = utilsChecker.cross_corr(d_mocapY_sum,d_videoY_sum,visualize=False) # neg lag means video started before mocap
    
    return lag    
    
def getMPJEs(lag, trialName, videoTime, mocapTime, mocapTRC, mocapData, 
             videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC,
             markersMPJE):
    
    if lag > 0 and 'tatic' not in trialName:
        print('WARNING video starts {} frames after mocap.'.format(lag))
    
    # Sync based on lag computed
    # Make sure video data is longer than mocap data on both ends
    if len(videoTime) < len(mocapTime) and 'running' in trialName:
        MPJE = np.nan
        MPJE_offsetRemoved = np.nan
        print('{} too trimmed'.format(trialName))
        return MPJE, MPJE_offsetRemoved        
    videoTime = videoTime + lag/mocapTRC.camera_rate
    
    startTime = np.max([videoTime[0],mocapTime[0]])
    endTime = np.min([videoTime[-1],mocapTime[-1]])
    
    N = int(np.round(np.round((endTime - startTime),2) * mocapTRC.camera_rate))
    NVideoAll = int(np.round(np.round((videoTime[-1] - videoTime[0]),2) * mocapTRC.camera_rate))
    syncTimeVec = np.linspace(startTime, endTime, N+1)
    timeVecVideoAll = np.linspace(videoTime[0], videoTime[-1], NVideoAll+1)
    
    print('t_start = ' + str(syncTimeVec[0]) + 's')
    
    mocapInds = np.arange(len(syncTimeVec)) + np.argmin(np.abs(mocapTime-startTime))
    videoInds = np.arange(len(syncTimeVec)) + np.argmin(np.abs(videoTime-startTime))

    # Copy videoData before offset removal
    videoDataAll = copy.deepcopy(videoData)    
    
    mocapData = mocapData[mocapInds,:]
    videoData = videoData[videoInds,:]
       
    offsetMkrs = markersMPJE
    offsetVals = np.empty((len(offsetMkrs),3))
    for iMkr,mkr in enumerate(offsetMkrs):
        idxM = mocapMkrNamesLower.index(mkr.lower())
        idxV = videoMkrNamesLower.index(mkr.lower() + '_study')
        offsetVals[iMkr,:] = np.mean(videoData[:,idxV*3:idxV*3+3] - mocapData[:,idxM*3:idxM*3+3],axis=0)
    offsetMean = np.mean(offsetVals,axis=0)    
    videoDataOffsetRemoved = videoData - np.tile(offsetMean,(1,videoTRC.num_markers))
    videoDataAllOffsetRemoved = videoDataAll - np.tile(offsetMean,(1,videoTRC.num_markers))
    
    # compute per joint errors
    MPJEvec = np.empty(len(markersMPJE))
    MPJE_offVec = np.copy(MPJEvec)    
    for i,mkr in enumerate(markersMPJE):
        idxM = mocapMkrNamesLower.index(mkr.lower())
        idxV = videoMkrNamesLower.index(mkr.lower() + '_study')
        MPJEvec[i] = np.mean(np.linalg.norm(videoData[:,idxV*3:idxV*3+3]- mocapData[:,idxM*3:idxM*3+3],axis = 1))
        MPJE_offVec[i] = np.mean(np.linalg.norm(videoDataOffsetRemoved[:,idxV*3:idxV*3+3]- mocapData[:,idxM*3:idxM*3+3],axis = 1))
    MPJE_mean = np.mean(MPJEvec)
    MPJE_std = np.std(MPJEvec)
    MPJE_offsetRemoved_mean = np.mean(MPJE_offVec)
    MPJE_offsetRemoved_std = np.std(MPJE_offVec)
    
    outputMPJE = {}
    outputMPJE['MPJE_mean'] = MPJE_mean
    outputMPJE['MPJE_std'] = MPJE_std
    outputMPJE['MPJE_std'] = MPJE_std
    outputMPJE['MPJE_offsetRemoved_mean'] = MPJE_offsetRemoved_mean
    outputMPJE['MPJE_offsetRemoved_std'] = MPJE_offsetRemoved_std
    outputMPJE['MPJEvec'] = MPJEvec
    outputMPJE['MPJE_offVec'] = MPJE_offVec
    outputMPJE['videoDataOffsetRemoved'] = videoDataOffsetRemoved
    outputMPJE['syncTimeVec'] = syncTimeVec
    outputMPJE['mocapData'] = mocapData
    outputMPJE['videoData'] = videoData
    outputMPJE['videoDataAllOffsetRemoved'] = videoDataAllOffsetRemoved
    outputMPJE['timeVecVideoAll'] = timeVecVideoAll
    outputMPJE['videoDataAll'] = videoDataAll
    
    return outputMPJE

# # %% Quick post-processing
# # Some video trials are named walkingT0 instead of walkingTO, let's adjust.
# for subjectName in sessionDetails:
#     for sessionName in sessionDetails[subjectName]:
#         subjectVideoDir = os.path.join(dataDir,'Data',sessionName)
#         markerDataFolders = glob.glob(os.path.join(subjectVideoDir,'MarkerData' + '/*/'))
#         for markerDataFolder in markerDataFolders:
#             camComboFolders = glob.glob(os.path.join(markerDataFolder,'*/'))
#             for camComboFolder in camComboFolders:
#                 prepostFolders = glob.glob(os.path.join(camComboFolder,'*/'))
#                 for prepostFolder in prepostFolders:
#                     for file in os.listdir(prepostFolder):
#                         if 'alkingT0' in file:
#                             pathFile = os.path.join(prepostFolder, file)
#                             fileName = file.replace('alkingT0', 'alkingTO')
#                             pathFileNew = os.path.join(prepostFolder, fileName)
#                             os.rename(pathFile, pathFileNew)

# # %% Process data
# if not os.path.exists(os.path.join(dataDir, 'Results-paper', MPJEName + '.npy')): 
#     MPJEs = {}
# else:  
#     MPJEs = np.load(os.path.join(dataDir, 'Results-paper', MPJEName + '.npy'),
#                     allow_pickle=True).item()
    
# poseDetectors = ['mmpose_0.8', 'OpenPose']
# cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
# augmenters = ['fullBody_mmpose', 'fullBody_OpenPose', 'old',
#               'separateLowerUpperBody_OpenPose_noPelvis', 'separateLowerUpperBody_OpenPose_Pelvis']    
# for subjectName in sessionDetails:
#     for poseDetector in poseDetectors:
#         for cameraSetup in cameraSetups:
#             for augmenter in augmenters:
#                 if augmenter in MPJEs[subjectName][poseDetector][cameraSetup]:
#                     print('{}-{}-{}-{}'.format(subjectName, poseDetector, cameraSetup,  augmenter))
#                     print(len(MPJEs[subjectName][poseDetector][cameraSetup][augmenter]))


# saveProcessedMocapData = False # Save processed and non-trimmed mocap data.
# overwriteMarkerDataProcessed = False # Overwrite non-trimmed mocap data.
# overwriteForceDataProcessed = False # Overwrite non-trimmed force data.
# overwritevideoAndMocap = True # Overwrite processed synced video-mocap data.
# writeMPJE_condition = True # Write MPJE for specific condition
# writeMPJE_session = True # Write MPJE for session

def main_sync(dataDir, subjectName, c_sessions, poseDetectors, cameraSetups, augmenters, videoParameters, saveProcessedMocapData=False,
    overwriteMarkerDataProcessed=False, overwriteForceDataProcessed=False,
    overwritevideoAndMocap=False, overwritevideo=False, writeMPJE_condition=False, writeMPJE_session=False,
    csv_name='MPJE_fullSession_new', augmenterModelName='LSTM'):
    MPJEs = {}
    print('\n\nProcessing {}'.format(subjectName))
    for sessionName in c_sessions:
        print('\nProcessing {}'.format(sessionName))
        
        sessionInfoOverride = False # true if you want to do specific trials below
        if not sessionInfoOverride:
            if '0001' in sessionName:
                subSession = '0001'
            elif '0002' in sessionName:
                subSession = '0002'
            else:
                raise ValueError("Error sub-session")
                
            originTrialName = videoParameters[subSession]['originName'] 
            r_fromMarker_toVideoOrigin_inLab = videoParameters[subSession]['r_fromMarker_toVideoOrigin_inLab']
            R_video_opensim = videoParameters[subSession]['R_video_opensim']
            R_opensim_xForward = videoParameters[subSession]['R_opensim_xForward']
            mocapFiltFreq = videoParameters[subSession]['mocapFiltFreq']
        mocapDir = os.path.join(dataDir,'Data',sessionName,'mocap','MarkerData')
        mocapOriginPath = os.path.join(mocapDir,originTrialName)
             
        # Loop over MarkerData files for different numbers of cameras
        subjectVideoDir = os.path.join(dataDir,'Data',sessionName)
        # markerDataFolders = glob.glob(os.path.join(subjectVideoDir,'MarkerData' + '/*/'))
        # Force order markerDataFolders to facilitate comparison with old data
        # We want mmpose first and openpose second.
        # poseDetectors = ['OpenPose_1x1008_4scales']
        markerDataFolders = []
        for poseDetector in poseDetectors:
            markerDataFolders.append(os.path.join(subjectVideoDir,'MarkerData',poseDetector))
            
        analysisNames = []
        MPJE_session = []
        MPJE_offsetRemoved_session = []
        trial_names = []
        for markerDataFolder in markerDataFolders:         
            temp = markerDataFolder.split('\\')
            # poseDetector = temp[len(temp) - 2]
            poseDetector = temp[-1]
            
            # TODO
            # if not poseDetector == 'mmpose_0.8':
            #     continue
            
            if not poseDetector in MPJEs:
                MPJEs[poseDetector] = {}
            
            print('\nProcessing {}'.format(markerDataFolder))
            # camComboFolders = glob.glob(os.path.join(markerDataFolder,'*/'))
            # Force order camComboFolders to facilitate comparison with old data
            # We want 2-cameras, 3-cameras, 5-cameras
            # cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
            # cameraSetups = ['5-cameras']
            camComboFolders = []
            for cameraSetup in cameraSetups:
                camComboFolders.append(os.path.join(markerDataFolder,cameraSetup))
            for camComboFolder in camComboFolders:      
                temp2 = camComboFolder.split('\\')
                # cameraSetup = temp2[len(temp2) - 2]
                cameraSetup = temp2[-1]
                
                # TODO
                # if not cameraSetup == '5-cameras':
                #     continue
                
                if not cameraSetup in MPJEs[poseDetector]:
                    MPJEs[poseDetector][cameraSetup] = {}
                
                print('\nProcessing {}'.format(camComboFolder))
                # Video path
                
                augmenterComboFolders = []
                for augmenter in augmenters:
                    augmenterComboFolders.append(os.path.join(camComboFolder,'PostAugmentation_updated_' + augmenter))
                
                for postDir in augmenterComboFolders:
                    
                    # if not postDir == 'PostAugmentation_separateLowerUpperBody_OpenPose':
                    #     continue
                
                    # if not 'PostAugmentation_separateLowerUpperBody_OpenPose' in postDir:
                    #     continue
                    
                    if postDir == 'PostAugmentation':
                        postAugmentationType = 'old'
                    else:
                        # Get string after last underscore in postDIr
                        postAugmentationType = postDir.split('_')[-1]
                        
                    if not postAugmentationType in MPJEs[poseDetector][cameraSetup]:
                        MPJEs[poseDetector][cameraSetup][postAugmentationType] = {}
                
                    videoTrcDir = os.path.join(camComboFolder, postDir)
                    print('\nProcessing {}'.format(videoTrcDir))
                                
                    # Get trialnames - hard code, or get all in video directory, as long as in mocap directory
                    trialNames = [os.path.split(tName.replace('_{}'.format(augmenterModelName),''))[1][0:-4] for tName in glob.glob(videoTrcDir + '/*.trc')]
                    trialsToRemove = []
                    for tName in trialNames: # check if in mocap directory, if not, delete trial
                        if not os.path.exists(os.path.join(mocapDir,tName + '.trc')):
                            trialsToRemove.append(tName)
                    [trialNames.remove(tName) for tName in trialsToRemove]
                    # Sort trialnames so same order each time
                    trialNames.sort()
                    
                    if len(trialNames) == 0:
                        raise Exception('No matching trialnames. Check paths: ' + videoTrcDir + '\n  ' + mocapDir)
                    
                    # Markers for MPJE computation (add arms eventually)
                    markersMPJE = ['c7','r_shoulder','l_shoulder','r.ASIS','l.ASIS','r.PSIS','l.PSIS','r_knee',
                                   'l_knee','r_ankle','l_ankle','r_calc','l_calc','r_toe','l_toe','r_5meta','l_5meta']                
                    
                    # Compute and save MPJE                
                    MPJE_mean = np.zeros((len(trialNames)))
                    MPJE_offsetRemoved_mean = np.zeros((len(trialNames)))
                    MPJE_std = np.zeros((len(trialNames)))
                    MPJE_offsetRemoved_std = np.zeros((len(trialNames)))
                    MPJE_markers = np.zeros((len(trialNames), len(markersMPJE)))
                    MPJE_offsetRemoved_markers = np.zeros((len(trialNames), len(markersMPJE)))
                    for idxTrial,trialName in enumerate(trialNames):
                        
                        # TODO
                        # if not 'STSweakLegs1' in trialName:
                        #     continue
                        
                        try:
                            MPJE_mean[idxTrial], MPJE_std[idxTrial], MPJE_offsetRemoved_mean[idxTrial], MPJE_offsetRemoved_std[idxTrial], MPJE_markers[idxTrial, :], MPJE_offsetRemoved_markers[idxTrial, :] = computeMarkerDifferences(
                                trialName,mocapDir,videoTrcDir,markersMPJE,mocapOriginPath, r_fromMarker_toVideoOrigin_inLab, mocapFiltFreq, R_video_opensim, R_opensim_xForward, saveProcessedMocapData, overwriteMarkerDataProcessed, 
                                overwriteForceDataProcessed, overwritevideoAndMocap, augmenterModelName=augmenterModelName)
                            if not 'headers' in MPJEs:
                                MPJEs_headers = markersMPJE.copy()
                                MPJEs_headers.append('mean')
                                MPJEs_headers.append('std')                                
                                MPJEs['headers'] = MPJEs_headers                                
                            c_MPJEs = np.zeros((len(markersMPJE)+2,))
                            c_MPJEs[:len(markersMPJE),] = MPJE_offsetRemoved_markers[idxTrial, :]
                            c_MPJEs[len(markersMPJE),] = MPJE_offsetRemoved_mean[idxTrial]
                            c_MPJEs[-1,] = MPJE_offsetRemoved_std[idxTrial]
                                
                            MPJEs[poseDetector][cameraSetup][postAugmentationType][trialName] = c_MPJEs
                        except Exception as e:
                            print(e)
                            nan_vec = np.nan*np.ones((len(markersMPJE),))
                            MPJE_mean[idxTrial], MPJE_std[idxTrial], MPJE_offsetRemoved_mean[idxTrial], MPJE_offsetRemoved_std[idxTrial], MPJE_markers[idxTrial, :], MPJE_offsetRemoved_markers[idxTrial, :] = np.nan, np.nan, np.nan, np.nan, nan_vec, nan_vec
                    
                    # Write to file
                    outputDir = os.path.join(videoTrcDir,'videoAndMocap')
                    if writeMPJE_condition:
                        writeMPJE(trialNames, outputDir, MPJE_mean, MPJE_std, MPJE_markers,
                                  MPJE_offsetRemoved_mean, MPJE_offsetRemoved_std, MPJE_offsetRemoved_markers,
                                  markersMPJE)  
                    poseDetector = os.path.basename(os.path.normpath(markerDataFolder))
                    nCams = os.path.basename(os.path.normpath(camComboFolder))
                    analysisNames.append(poseDetector + '__' + nCams + '__' +
                                          postAugmentationType)
                    MPJE_session.append(MPJE_mean)
                    MPJE_offsetRemoved_session.append(MPJE_offsetRemoved_mean)
                    trial_names.append(trialNames)
        
        # Write all MPJE to session file
        markerDataDir = os.path.join(subjectVideoDir, 'MarkerData')
        if writeMPJE_session:
            writeMPJE_perSession(trialNames, markerDataDir, MPJE_session,
                                  MPJE_offsetRemoved_session, analysisNames,
                                  csv_name)
            
            # Check if npy file MPJE_all in markerDataDir
            if os.path.exists(os.path.join(markerDataDir, 'MPJE_all_updated.npy')):
                MPJE_all = np.load(os.path.join(markerDataDir, 'MPJE_all_updated.npy'), allow_pickle=True).item()
                # MPJE_all = {}
            else:
                MPJE_all = {}
            for analysisName, MPJE, MPJE_offsetRemoved, trial_name in zip(analysisNames, MPJE_session, MPJE_offsetRemoved_session, trial_names):
                # Split analysisName based on last three underscores
                analysisName = analysisName.split('__')[-3:]
                if not analysisName[0] in list(MPJE_all.keys()):
                    MPJE_all[analysisName[0]] = {}
                if not analysisName[1] in list(MPJE_all[analysisName[0]].keys()):
                    MPJE_all[analysisName[0]][analysisName[1]] = {}
                if not analysisName[2] in list(MPJE_all[analysisName[0]][analysisName[1]].keys()):
                    MPJE_all[analysisName[0]][analysisName[1]][analysisName[2]] = {}  
                MPJE_all[analysisName[0]][analysisName[1]][analysisName[2]]['MPJE'] = MPJE
                MPJE_all[analysisName[0]][analysisName[1]][analysisName[2]]['MPJE_offsetRemoved'] = MPJE_offsetRemoved
                MPJE_all[analysisName[0]][analysisName[1]][analysisName[2]]['trials'] = trial_name
            np.save(os.path.join(markerDataDir, 'MPJE_all_updated.npy'), MPJE_all)
            
    return MPJEs

# if saveMPJEs:
#     np.save(os.path.join(dataDir, 'Results-paper', MPJEName + '.npy'), MPJEs)
