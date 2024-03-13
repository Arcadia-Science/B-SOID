import sys
import subprocess
import os
import argparse

# Descriptions of environment variables that need to be specified to run the CLI command
# WORKING_DIR_BSOID: Path to where you want outputs to go.
# PREFIX_BSOID: Prefix that you want appended to output files.
# FTYPE_BSOID: File type; DeepLabCut: 'h5' or 'csv', SLEAP: 'h5', and OpenPose: 'json'.
# ROOT_PATH_BSOID: Path to the working directory containing sub-directories that have input .csv,.h5, or .json files.
# FRAMERATE_BSOID: Framerate for pose estimate files; in frames per second.
# DATA_DIR_BSOID: Path of sub-directories that have input .csv,.h5, or .json files; provided as a list and relative to working_dir, e.g. '/1_1,/1_2,/2_1,/2_2'.

# The BSOiD prediction step can be run with a command like this (definitions of environmental variables followed by 'python run_streamlit_cli_predict.py'):
# FTYPE_BSOID='csv' ROOT_PATH_BSOID='/Users/Desktop/training/' FRAMERATE_BSOID=120 WORKING_DIR_BSOID='/Users/Desktop/training/output' PREFIX_BSOID='controltry' DATA_DIR_BSOID='/1_1,/1_2,/2_1,/2_2' python run_streamlit_cli_predict.py


# Define the command to run the Streamlit app
def main():
    cmd = [
        "streamlit",
        "run",
        "bsoid_predict.py",
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
