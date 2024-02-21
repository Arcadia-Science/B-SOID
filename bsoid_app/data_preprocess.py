import os
from datetime import date
import sys
import h5py
import joblib
import randfacts
import streamlit as st
                    
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *
            
class Preprocess:
                    
    def __init__(self,WORKING_DIR,PREFIX,SOFTWARE_CHOICE,FTYPE,ROOT_PATH,FRAMERATE,DATA_DIRECTORIES,POSE_LIST,VALUE):   
        print('LOAD DATA and PREPROCESS')
        self.software = SOFTWARE_CHOICE
        self.FTYPE = FTYPE
        self.ROOT_PATH = ROOT_PATH
        self.FRAMERATE = FRAMERATE
        self.DATA_DIRECTORIES = DATA_DIRECTORIES.split(',')
        self.POSE_LIST = POSE_LIST
        self.WORKING_DIR = WORKING_DIR
        self.PREFIX = PREFIX
        self.pose_chosen = []
        self.input_filenames = []
        self.raw_input_data = []   
        self.processed_input_data = []
        self.sub_threshold = []
        try:
            os.listdir(self.ROOT_PATH)
            print(
                'You have selected {} as your root directory'.format(self.ROOT_PATH))
        except FileNotFoundError:
            st.error('No such root directory')
            sys.exit(1)
        no_dir = int(float(VALUE))
        print('You will be training on {} data file containing sub-directories.'.format(no_dir))
        print('You have selected {} as your _sub-directory(ies)_.'.format(self.DATA_DIRECTORIES))
        print('You have selected {} frames per second.'.format(self.FRAMERATE))
        try:
            os.listdir(self.WORKING_DIR)
            print('You have selected {} for B-SOiD working directory.'.format(self.WORKING_DIR))
        except FileNotFoundError:
            print('Error:Cannot access working directory, was there a typo or did you forget to create one?')
            sys.exit(1)
        today = date.today()
        d4 = today.strftime("%b-%d-%Y")
        if self.PREFIX:
            print('You have decided on {} as the PREFIX.'.format(self.PREFIX))
        else:
            st.error('Please enter a PREFIX.')
            sys.exit(1)
                
    def compile_data(self):
        if self.software == 'DeepLabCut' and self.FTYPE == 'csv':
            data_files = glob.glob(self.ROOT_PATH + self.DATA_DIRECTORIES[0] + '/*.csv')
            file0_df = pd.read_csv(data_files[0], low_memory=False)
            file0_array = np.array(file0_df)
            for a in self.POSE_LIST:
                indices = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                self.pose_chosen.extend(indices)  # Add indices to pose_chosen list
            self.pose_chosen.sort()  # Sort the pose indices
            print("Selected poses to include:", self.POSE_LIST)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.DATA_DIRECTORIES):  # Loop through folders
                f = get_filenames(self.ROOT_PATH, fd)
                my_bar = st.progress(0)
                for j, filename in enumerate(f):
                    file_j_df = pd.read_csv(filename, low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df)
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(filename)
                    my_bar.progress(round((j + 1) / len(f) * 100))
            with open(os.path.join(self.WORKING_DIR, str.join('', (self.PREFIX, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.ROOT_PATH, self.DATA_DIRECTORIES, self.FRAMERATE, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.FTYPE,
                                               np.array(self.processed_input_data).shape))
        elif self.software == 'DeepLabCut' and self.FTYPE == 'h5':
            data_files = glob.glob(self.ROOT_PATH + self.DATA_DIRECTORIES[0] + '/*.h5')
            file0_df = pd.read_hdf(data_files[0], low_memory=False)
            for a in self.POSE_LIST:
                index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_VALUEs(1))) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            print("Selected poses to include:", self.POSE_LIST)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.DATA_DIRECTORIES):
                f = get_filenamesh5(self.ROOT_PATH, fd)
                my_bar = st.progress(0)
                for j, filename in enumerate(f):
                    file_j_df = pd.read_hdf(filename, low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df)   
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(filename)
                    my_bar.progress(round((j + 1) / len(f) * 100))
            with open(os.path.join(self.WORKING_DIR, str.join('', (self.PREFIX, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.ROOT_PATH, self.DATA_DIRECTORIES, self.FRAMERATE, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.FTYPE,
                                               np.array(self.processed_input_data).shape))
        elif self.software == 'SLEAP' and self.FTYPE == 'h5':
            data_files = glob.glob(self.ROOT_PATH + self.DATA_DIRECTORIES[0] + '/*.h5')
            file0_df = h5py.File(data_files[0], 'r')
            for a in POSE_LIST:
                index = [i for i, s in enumerate(np.array(file0_df['node_names'][:])) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            print("Selected poses to include:", self.POSE_LIST)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.DATA_DIRECTORIES):
                f = get_filenamesh5(self.ROOT_PATH, fd)
                my_bar = st.progress(0)
                for j, filename in enumerate(f):
                    file_j_df = h5py.File(filename, 'r')
                    file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df['tracks'][:][0])
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(filename)
                    my_bar.progress(round((j + 1) / len(f) * 100))
            with open(os.path.join(self.WORKING_DIR, str.join('', (self.PREFIX, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.ROOT_PATH, self.DATA_DIRECTORIES, self.FRAMERATE, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.FTYPE,
                                               np.array(self.processed_input_data).shape))
        elif self.software == 'OpenPose' and self.FTYPE == 'json':
            data_files = glob.glob(self.ROOT_PATH + self.DATA_DIRECTORIES[0] + '/*.json')
            file0_df = read_json_single(data_files[0])
            file0_array = np.array(file0_df)  
            for a in POSE_LIST:
                index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            print("Selected poses to include:", self.POSE_LIST)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.DATA_DIRECTORIES):
                f = get_filenamesjson(self.ROOT_PATH, fd)
                json2csv_multi(f)
                filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
                file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                        low_memory=False)
                file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                self.raw_input_data.append(file_j_df)
                self.sub_threshold.append(p_sub_threshold)
                self.processed_input_data.append(file_j_processed)
                self.input_filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
            with open(os.path.join(self.WORKING_DIR, str.join('', (self.PREFIX, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.ROOT_PATH, self.DATA_DIRECTORIES, self.FRAMERATE, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.FTYPE,
                                               np.array(self.processed_input_data).shape))
