import sys
import subprocess
import os
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Run a Streamlit app with a specified working directory.')

# Add an argument for the working directory
parser.add_argument('--working_dir', type=str, help='Path to where you want outputs to go')
parser.add_argument('--prefix', type=str, help='Prefix that you want appended to output files')
parser.add_argument('--software_choice', type=str, help="Pose estimation software; must be 'DeepLabCut','SLEAP', or 'OpenPose'")
parser.add_argument('--ftype', type=str, help="File type; DeepLabCut: 'h5' or 'csv', SLEAP: 'h5', and OpenPose: 'json'")
parser.add_argument('--root_path', type=str, help='Path to the working directory containing sub-directories that have input .csv,.h5, or .json files')
parser.add_argument('--framerate', type=float, help='Framerate for pose estimate files; in frames per second')
parser.add_argument('--value', type=float, help='Number of sub-directories that have input .csv,.h5, or .json files')
parser.add_argument('--data_directories', type=str, help="Path of sub-directories that have input .csv,.h5, or .json files; provided as a list and relative to working_di$
parser.add_argument('--min_cluster_range', type=str, help='Minimum cluster size, based on minimum temporal bout and will represent a %. Impacts number of clusters')
parser.add_argument('--max_cluster_range', type=str, help='Maximum cluster size, will represent a % and impacts number of clusters')
parser.add_argument('--autosave', type=str, help="Whether or not you want to autosave clustering as you go. Should be 'Yes' or 'No'")
parser.add_argument('--pose_list', type=str, help="List of poses to include in analysis, for example, 'R_rear,L_rear'")

# Parse the command line arguments
args = parser.parse_args()

#Set environment variables
os.environ['working_dir'] = args.working_dir
os.environ['prefix'] = args.prefix
os.environ['software_choice'] = args.software_choice
os.environ['ftype'] = args.ftype
os.environ['root_path'] = args.root_path
os.environ['framerate'] = str(args.framerate)
os.environ['value'] = str(args.value)
os.environ['data_directories'] = args.data_directories
os.environ['min_cluster_range'] = args.min_cluster_range
os.environ['max_cluster_range'] = args.max_cluster_range
os.environ['autosave'] = args.autosave
os.environ['pose_list'] = args.pose_list

# Define the command to run the Streamlit app
cmd = [
    "streamlit",
    "run",
    "bsoid_app.py",
]

# Run the Streamlit app
subprocess.run(cmd)
