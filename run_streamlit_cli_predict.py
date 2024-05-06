import subprocess
import os
import time

# Descriptions of environment variables that need to be specified to run the CLI command can be found in the README file(./README.md).

# The BSOiD prediction step can be run with a command like this (definitions of environmental variables followed by 'python run_streamlit_cli_predict.py'):
# FTYPE_BSOID='csv' ROOT_PATH_BSOID='/Users/Desktop/training/' FRAMERATE_BSOID=120 WORKING_DIR_BSOID='/Users/Desktop/training/output' PREFIX_BSOID='controltry' DATA_DIR_BSOID='/1_1,/1_2,/$

def run_streamlit():
    signal_file = 'app_done.txt'
    if os.path.exists(signal_file):
        os.remove(signal_file)

    process = subprocess.Popen(["streamlit", "run", "bsoid_predict.py"])

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
