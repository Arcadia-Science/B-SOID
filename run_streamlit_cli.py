import sys
import subprocess
import os
import argparse

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

# The BSOiD pipeline can be run with a command like this (definitions of environmental variables followed by 'python run_streamlit_cli.py'):
# SOFTWARE_BSOID='DeepLabCut' FTYPE_BSOID='csv' ROOT_PATH_BSOID='/Users/Desktop/training/' FRAMERATE_BSOID=120 WORKING_DIR_BSOID='/Users/Desktop/training/output' PREFIX_BSOID='controltry' VALUE_BSOID=4.0 DATA_DIR_BSOID='/1_1,/1_2,/2_1,/2_2' AUTOSAVE_BSOID='Yes' POSE_LIST_BSOID='R_rear,L_rear' python run_streamlit_cli.py


# Define the command to run the Streamlit app
def main():
    cmd = [
        "streamlit",
        "run",
        "bsoid_app.py",
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
