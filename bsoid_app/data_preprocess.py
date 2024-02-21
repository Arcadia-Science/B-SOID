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
        self.ftype = FTYPE
        self.root_path = ROOT_PATH
        self.framerate = FRAMERATE
        self.data_directories = DATA_DIRECTORIES.split(',')
        self.pose_list = POSE_LIST
        self.working_dir = WORKING_DIR
        self.prefix = PREFIX
        self.pose_chosen = []
        self.input_filenames = []
        self.raw_input_data = []   
        self.processed_input_data = []
        self.sub_threshold = []
        try:
            os.listdir(self.root_path)
            print(
                'You have selected {} as your root directory'.format(self.root_path))
        except FileNotFoundError:
            st.error('No such root directory')
            sys.exit(1)
        no_dir = int(float(VALUE))
        print('You will be training on {} data file containing sub-directories.'.format(no_dir))
        print('You have selected {} as your _sub-directory(ies)_.'.format(self.data_directories))
        print('You have selected {} frames per second.'.format(self.framerate))
        try:
            os.listdir(self.working_dir)
            print('You have selected {} for B-SOiD working directory.'.format(self.working_dir))
        except FileNotFoundError:
            print('Error:Cannot access working directory, was there a typo or did you forget to create one?')
            sys.exit(1)
        today = date.today()
        d4 = today.strftime("%b-%d-%Y")
        if self.prefix:
            print('You have decided on {} as the PREFIX.'.format(self.prefix))
        else:
            st.error('Please enter a PREFIX.')
            sys.exit(1)
                
    def compile_data(self):
        if self.software == 'DeepLabCut' and self.ftype == 'csv':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.csv')
            file0_df = pd.read_csv(data_files[0], low_memory=False)
            file0_array = np.array(file0_df)
            for a in self.pose_list:
                indices = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                self.pose_chosen.extend(indices)  # Add indices to pose_chosen list
            self.pose_chosen.sort()  # Sort the pose indices
            print("Selected poses to include:", self.pose_list)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.data_directories):  # Loop through folders
                f = get_filenames(self.root_path, fd)
                my_bar = st.progress(0)
                for j, filename in enumerate(f):
                    file_j_df = pd.read_csv(filename, low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df)
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(filename)
                    my_bar.progress(round((j + 1) / len(f) * 100))
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.ftype,
                                               np.array(self.processed_input_data).shape))
        elif self.software == 'DeepLabCut' and self.ftype == 'h5':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.h5')
            file0_df = pd.read_hdf(data_files[0], low_memory=False)
            for a in self.pose_list:
                index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_VALUEs(1))) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            print("Selected poses to include:", self.pose_list)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.data_directories):
                f = get_filenamesh5(self.root_path, fd)
                my_bar = st.progress(0)
                for j, filename in enumerate(f):
                    file_j_df = pd.read_hdf(filename, low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df)   
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(filename)
                    my_bar.progress(round((j + 1) / len(f) * 100))
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.ftype,
                                               np.array(self.processed_input_data).shape))
        elif self.software == 'SLEAP' and self.ftype == 'h5':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.h5')
            file0_df = h5py.File(data_files[0], 'r')
            for a in POSE_LIST:
                index = [i for i, s in enumerate(np.array(file0_df['node_names'][:])) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            print("Selected poses to include:", self.pose_list)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.data_directories):
                f = get_filenamesh5(self.root_path, fd)
                my_bar = st.progress(0)
                for j, filename in enumerate(f):
                    file_j_df = h5py.File(filename, 'r')
                    file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df['tracks'][:][0])
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(filename)
                    my_bar.progress(round((j + 1) / len(f) * 100))
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.ftype,
                                               np.array(self.processed_input_data).shape))
        elif self.software == 'OpenPose' and self.ftype == 'json':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.json')
            file0_df = read_json_single(data_files[0])
            file0_array = np.array(file0_df)  
            for a in POSE_LIST:
                index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            print("Selected poses to include:", self.pose_list)
            print('PREPROCESSING...')
            for i, fd in enumerate(self.data_directories):
                f = get_filenamesjson(self.root_path, fd)
                json2csv_multi(f)
                filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
                file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                        low_memory=False)
                file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                self.raw_input_data.append(file_j_df)
                self.sub_threshold.append(p_sub_threshold)
                self.processed_input_data.append(file_j_processed)
                self.input_filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                joblib.dump(
                    [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                     self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                )
            print('Processed a total of {} .{} files, and compiled into a '
                    '{} data list.'.format(len(self.processed_input_data), self.ftype,
                                               np.array(self.processed_input_data).shape))
