import sys
sys.path.append("./..")
import os
import shutil

part1 = False
part2 = False
part3 = True

# %% Messy helper function
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

# %% Validation: please keep here, hack to get all trials at once.
subjects_to_process = ['subject' + str(i) for i in range(10,12)]
sessionNames, sessionDetails = get_subject(subjects_to_process)

pathDrive = 'C:/MyDriveSym/Projects/mobilecap/Data/'
pathLocal = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/Data/'

# Copy data from drive to local
if part1:
    for count, sessionName in enumerate(sessionNames):
        print('Processing session: ' + sessionName)

        pathSessionDrive = pathDrive + sessionName + '/'
        pathSessionLocal = pathLocal + sessionName + '/'
        os.makedirs(pathSessionLocal, exist_ok=True)

        # Copy metadata
        pathMetadataDrive = pathSessionDrive + 'sessionMetadata.yaml'
        pathMetadataLocal = pathSessionLocal + 'sessionMetadata.yaml'
        shutil.copy2(pathMetadataDrive, pathMetadataLocal)

        # Copy videos
        pathVideosDrive = pathSessionDrive + 'Videos/'
        pathVideosLocal = pathSessionLocal + 'Videos/'
        os.makedirs(pathVideosLocal, exist_ok=True)
        for i in range(5):
            pathCamDrive  = pathVideosDrive + 'Cam' + str(i) + '/'
            pathCamLocal  = pathVideosLocal + 'Cam' + str(i) + '/'
            pathInputMediaDrive = pathCamDrive + 'InputMedia/'
            pathInputMediaLocal = pathCamLocal + 'InputMedia/'
            for folder in os.listdir(pathInputMediaDrive):
                # check if folder is a folder
                if not os.path.isdir(pathInputMediaDrive + folder):
                    continue
                pathFolderDrive = pathInputMediaDrive + folder + '/'
                pathFolderLocal = pathInputMediaLocal + folder + '/'
                os.makedirs(pathFolderLocal, exist_ok=True)
                for file in os.listdir(pathFolderDrive):
                    if file.endswith('.avi') or file.endswith('.mov'):
                        pathFileDrive = pathFolderDrive + file
                        pathFileLocal = pathFolderLocal + file
                        shutil.copy2(pathFileDrive, pathFileLocal)
                    
            # Copy parameters
            pathParametersDrive = pathCamDrive + 'cameraIntrinsicsExtrinsics.pickle'
            pathParametersLocal = pathCamLocal + 'cameraIntrinsicsExtrinsics.pickle'
            shutil.copy2(pathParametersDrive, pathParametersLocal)

# Delete data on drive
if part2:
    for count, sessionName in enumerate(sessionNames):
        print('Processing session: ' + sessionName)
        pathSessionDrive = pathDrive + sessionName + '/'
        pathVideosDrive = pathSessionDrive + 'Videos/'
        for i in range(5):
            pathCamDrive  = pathVideosDrive + 'Cam' + str(i) + '/'
            pathJsonsDrive = pathCamDrive + 'OutputJsons_1x736/'
            pathOutputMediaDrive = pathCamDrive + 'OutputMedia_1x736/'
            pathPklDrive = pathCamDrive + 'OutputPkl_1x736/'
            # Delete folders
            if os.path.isdir(pathJsonsDrive):
                shutil.rmtree(pathJsonsDrive)
            if os.path.isdir(pathOutputMediaDrive):
                shutil.rmtree(pathOutputMediaDrive)
            if os.path.isdir(pathPklDrive):
                shutil.rmtree(pathPklDrive)

        pathMarkerDataDrive = pathSessionDrive + 'MarkerData/OpenPose_1x736/'
        # Check if folder exists
        if os.path.isdir(pathMarkerDataDrive):
            shutil.rmtree(pathMarkerDataDrive)

# Copy data back to drive
camera_setups = ['2-cameras', '3-cameras', '5-cameras']
if part3:
    for count, sessionName in enumerate(sessionNames):
        print('Processing session: ' + sessionName)

        pathSessionDrive = pathDrive + sessionName + '/'
        pathSessionLocal = pathLocal + sessionName + '/'
        os.makedirs(pathSessionLocal, exist_ok=True)

        # Copy video-related content
        pathVideosDrive = pathSessionDrive + 'Videos/'
        pathVideosLocal = pathSessionLocal + 'Videos/'
        for i in range(5):
            pathCamDrive  = pathVideosDrive + 'Cam' + str(i) + '/'
            pathCamLocal  = pathVideosLocal + 'Cam' + str(i) + '/'

            # pathJsonsDrive = pathCamDrive + 'OutputJsons_1x736/'
            pathOutputMediaDrive = pathCamDrive + 'OutputMedia_1x736/'
            pathPklDrive = pathCamDrive + 'OutputPkl_1x736/'

            # pathJsonsLocal = pathCamLocal + 'OutputJsons_1x736/'
            pathOutputMediaLocal = pathCamLocal + 'OutputMedia_1x736/'
            pathPklLocal = pathCamLocal + 'OutputPkl_1x736/'

            for folder in os.listdir(pathOutputMediaLocal):
                # check if folder is a folder
                if not os.path.isdir(pathOutputMediaLocal + folder):
                    continue
                pathFolderDrive = pathOutputMediaDrive + folder + '/'
                pathFolderLocal = pathOutputMediaLocal + folder + '/'
                os.makedirs(pathFolderDrive, exist_ok=True)
                for file in os.listdir(pathFolderLocal):
                    if file.endswith('.avi') or file.endswith('.mov'):
                        pathFileDrive = pathFolderDrive + file
                        pathFileLocal = pathFolderLocal + file
                        shutil.copy2(pathFileLocal, pathFileDrive)

            for folder in os.listdir(pathPklLocal):
                # check if folder is a folder
                if not os.path.isdir(pathPklLocal + folder):
                    continue
                pathFolderDrive = pathPklDrive + folder + '/'
                pathFolderLocal = pathPklLocal + folder + '/'
                os.makedirs(pathFolderDrive, exist_ok=True)
                for file in os.listdir(pathFolderLocal):
                    if file.endswith('.pkl') or file.endswith('.txt'):
                        pathFileDrive = pathFolderDrive + file
                        pathFileLocal = pathFolderLocal + file
                        shutil.copy2(pathFileLocal, pathFileDrive)

        # Copy marker data
        pathMarkerDataLocal = pathSessionLocal + 'MarkerData/OpenPose_1x736/'
        pathMarkerDataDrive = pathSessionDrive + 'MarkerData/OpenPose_1x736/'

        for camera_setup in camera_setups:
            pathMarkerCamLocal = pathMarkerDataLocal + camera_setup + '/'
            pathMarkerCamDrive = pathMarkerDataDrive + camera_setup + '/'

            for folder in os.listdir(pathMarkerCamLocal):
                # check if folder is a folder
                if not os.path.isdir(pathMarkerCamLocal + folder):
                    continue    
                pathFolderDrive = pathMarkerCamDrive + folder + '/'
                pathFolderLocal = pathMarkerCamLocal + folder + '/'
                os.makedirs(pathFolderDrive, exist_ok=True)
                for file in os.listdir(pathFolderLocal):
                    if file.endswith('.trc') or file.endswith('.yaml'):
                        pathFileDrive = pathFolderDrive + file
                        pathFileLocal = pathFolderLocal + file
                        shutil.copy2(pathFileLocal, pathFileDrive)

        




