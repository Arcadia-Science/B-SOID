import subprocess
import os
import time

# Descriptions of environment variables that need to be specified to run the CLI command
# WORKING_DIR_BSOID: Path to where you want outputs to go.
# PREFIX_BSOID: Prefix that you want appended to output files.
# SOFTWARE_BSOID: Pose estimation software; must be 'DeepLabCut','SLEAP', or 'OpenPose'.
# FTYPE_BSOID: File type; DeepLabCut: 'h5' or 'csv', SLEAP: 'h5', and OpenPose: 'json'.
# ROOT_PATH_BSOID: Path to the working directory containing sub-directories that have input .csv,.h5, or .json files.
# FRAMERATE_BSOID: Framerate for pose estimate files; in frames per second.
# VALUE_BSOID: Number of sub-directories that have input .csv,.h5, or .json files.
# DATA_DIR_BSOID: Path of sub-directories that have input .csv,.h5, or .json files; provided as a list and relative to working_dir, e.g. '/1_1,/1_2,/2_1,/2_2'.
# MIN_CLUSTER_BSOID: default='0.5',Minimum cluster size, based on minimum temporal bout and will represent a %. Impacts number of clusters.
# MAX_CLUSTER_BSOID: default='1', Maximum cluster size, will represent a % and impacts number of clusters.
# AUTOSAVE_BSOID: Whether or not you want to autosave clustering as you go. Should be 'Yes' or 'No'. Default is 'Yes'.
# POSE_LIST_BSOID: List of poses to include in analysis, for example, 'R_rear,L_rear'.
# MIN_TIME_BSOID: Minimum time for bout in ms. Default is 200.
# NUMBER_EXAMPLES_BSOID: Number of non-repeated examples for video snippets. Default is 5.
# PLAYBACK_SPEED_BSOID: Playback speed for video snippets. Default is 0.75X.
# FRACTION_BSOID: Training input fraction (do not change this value if you wish to generate the side-by-side video seen on B-SOiD GitHub page). Default is 1, minimum is 0.1, maximum is 1.

# The BSOiD pipeline can be run with a command like this (definitions of environmental variables followed by 'python run_streamlit_cli.py'):
# SOFTWARE_BSOID='DeepLabCut' FTYPE_BSOID='csv' ROOT_PATH_BSOID='/Users/Desktop/training/' FRAMERATE_BSOID=120 WORKING_DIR_BSOID='/Users/Desktop/training/output' PREFIX_BSOID='controltry' VALUE_BSOID=4.0 DATA_DIR_BSOID='/1_1,/1_2,/2_1,/2_2' AUTOSAVE_BSOID='Yes' POSE_LIST_BSOID='R_rear,L_rear' python run_streamlit_cli.py


def run_streamlit():
    signal_file = 'app_done.txt'
    if os.path.exists(signal_file):
        os.remove(signal_file)

    process = subprocess.Popen(["streamlit", "run", "bsoid_app.py"])

    while not os.path.exists(signal_file):
        time.sleep(1)

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()

    if os.path.exists(signal_file):
        os.remove(signal_file)

if __name__ == "__main__":
    run_streamlit()
