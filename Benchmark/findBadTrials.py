import os
import sys
sys.path.append("./..")
import numpy as np
from utils import getDataDirectory

def get_subject(subjects_to_process):
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
            'Session20210910_0002': {}}}
    sessionDetails_out = {}
    sessionNames_out = []
    for subject in subjects_to_process:
        sessionDetails_out[subject] = {}
        for session in sessionDetails[subject]:
            sessionDetails_out[subject][session] = {}   
            sessionNames_out.append(session)
    
    return sessionNames_out, sessionDetails_out 

# cases_to_exclude_missing = {'Session20210816_0001': '43a6c17b-39b4-4de6-ac55-dd65839dc2db'}
cases_to_exclude_paper = ['static', 'stsasym', 'stsfast', 'walkingti', 'walkingto']
cases_to_exclude_trials = {'Session20210813_0002': ['walkingTS3']}
pose_detectors_to_exclude = ['OpenPose_default']
expected_nSensitivities = 54
for i in pose_detectors_to_exclude:
    if 'openpose' in i.lower():
        expected_nSensitivities -= 12
    elif 'mmpose' in i.lower():
        expected_nSensitivities -= 18

threshold = 51 # 5cm to spot sync errors

subjects_to_process = ['subject' + str(i) for i in range(2,12)]
sessionNames, sessionDetails = get_subject(subjects_to_process)

dataDir = getDataDirectory()

nTrials = {}

for count, sessionName in enumerate(sessionNames):
    # print('Processing session: ' + sessionName)
    nTrials[sessionName] = {}    
    markerDataDir = dataDir + '/Data/' + sessionName + '/MarkerData/'
    MPJE_all = np.load(os.path.join(markerDataDir, 'MPJE_all.npy'), allow_pickle=True).item()
    countTests = 0
    for pose_detector in list(MPJE_all.keys()):
        
        if pose_detector in pose_detectors_to_exclude:
            continue
        
        
        nTrials[sessionName][pose_detector] = {}
        for camera_setup in list(MPJE_all[pose_detector].keys()):
            nTrials[sessionName][pose_detector][camera_setup] = {}
            for augmenter in list(MPJE_all[pose_detector][camera_setup].keys()):
                nTrials[sessionName][pose_detector][camera_setup][augmenter] = 0
                for t, trial in enumerate(MPJE_all[pose_detector][camera_setup][augmenter]['trials']):
                    if not any(case in trial.lower() for case in cases_to_exclude_paper):

                        if sessionName in cases_to_exclude_trials:
                            if trial in cases_to_exclude_trials[sessionName]:
                                continue
                        
                        nTrials[sessionName][pose_detector][camera_setup][augmenter] += 1                        
                        if MPJE_all[pose_detector][camera_setup][augmenter]['MPJE_offsetRemoved'][t] > threshold:
                            a=1
                            print('Session: ' + sessionName + ', pose detector: ' + pose_detector + ', camera setup: ' + camera_setup + ', augmenter: ' + augmenter + ', trial: ' + trial + ', MPJE: ' + str(MPJE_all[pose_detector][camera_setup][augmenter]['MPJE_offsetRemoved'][t]))
                
                countTests += 1
                # print('Number of trials {} - {} - {} - {}: {}'.format(sessionName, pose_detector, camera_setup, augmenter, nTrials[sessionName][pose_detector][camera_setup][augmenter]))
                if '_0001' in  sessionName:
                    if nTrials[sessionName][pose_detector][camera_setup][augmenter] != 10:
                        print('Number of trials {} - {} - {} - {}: {}'.format(sessionName, pose_detector, camera_setup, augmenter, nTrials[sessionName][pose_detector][camera_setup][augmenter]))
                if '_0002' in  sessionName:
                    if nTrials[sessionName][pose_detector][camera_setup][augmenter] != 6:
                        print('Number of trials {} - {} - {} - {}: {}'.format(sessionName, pose_detector, camera_setup, augmenter, nTrials[sessionName][pose_detector][camera_setup][augmenter]))
    if countTests != expected_nSensitivities:
        print("Missing sensitivity analyses")
                        
                        
        




