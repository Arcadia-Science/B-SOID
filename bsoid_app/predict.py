import os
from datetime import date

import h5py
import joblib
import streamlit as st

from bsoid_app.bsoid_utilities import statistics
from bsoid_app.bsoid_utilities.bsoid_classification import *
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *


class Prediction:

    def __init__(self, ROOT_PATH, DATA_DIRECTORIES, input_filenames, processed_input_data, WORKING_DIR, PREFIX,
                 FRAMERATE, pose_chosen, predictions, FTYPE, clf):
        print('PREDICT OLD/NEW FILES USING A MODEL')
        print('This could take some time for large datasets.')
        self.options = ['Labels tagged onto pose files', 'Group durations (in frames)', 'Transition matrix']
        self.root_path = ROOT_PATH
        self.data_directories = DATA_DIRECTORIES
        self.input_filenames = input_filenames
        self.processed_input_data = processed_input_data
        self.working_dir = WORKING_DIR
        self.prefix = PREFIX
        self.framerate = FRAMERATE
        self.pose_chosen = pose_chosen
        self.predictions = predictions
        self.clf = clf
        self.use_train = []
        self.new_root_path = []
        self.new_directories = []
        self.filetype = FTYPE
        self.new_prefix = []
        self.new_framerate = []
        self.new_data = []
        self.new_features = []
        self.nonfs_predictions = []
        self.folders = []
        self.folder = []
        self.filenames = []
        self.new_predictions = []
        self.all_df = []

    def setup(self):
        self.new_root_path = self.root_path
        self.filetype = [s for i, s in enumerate(['csv', 'h5', 'json'])
                         if s in self.input_filenames[0].partition('.')[-1]][0]
        self.new_directories = self.data_directories
        self.new_framerate = self.framerate
        self.new_prefix = self.prefix
        print('You have selected **{}** as your _sub-directory(ies)_.'.format(self.new_directories))
        print('Average video frame-rate for xxx.{} pose estimate files.'.format(self.filetype))
        print('You have selected **{} frames per second**.'.format(self.new_framerate))

    def predict(self):
        print('These files will be saved in {}/_your_data_folder_x_/BSOID'.format(self.new_root_path))
        if self.filetype == 'csv':
            for i, fd in enumerate(self.new_directories):
                f = get_filenames(self.new_root_path, fd)
                for j, filename in enumerate(f):
                    file_j_df = pd.read_csv(filename, low_memory=False)
                    file_j_processed, _ = adp_filt(file_j_df, self.pose_chosen)
                    self.all_df.append(file_j_df)
                    self.new_data.append(file_j_processed)
                    self.filenames.append(filename)
                    self.folder.append(fd)
                self.folders.append(fd)
        elif self.filetype == 'h5':
            try:
                for i, fd in enumerate(self.new_directories):
                    f = get_filenamesh5(self.new_root_path, fd)
                    for j, filename in enumerate(f):
                        file_j_df = pd.read_hdf(filename, low_memory=False)
                        file_j_processed, _ = adp_filt_h5(file_j_df, self.pose_chosen)
                        self.all_df.append(file_j_df)
                        self.new_data.append(file_j_processed)
                        self.filenames.append(filename)
                        self.folder.append(fd)
                    self.folders.append(fd)
            except:
                print('Detecting SLEAP .h5 files...')
                for i, fd in enumerate(self.new_directories):
                    f = get_filenamesh5(self.new_root_path, fd)
                    for j, filename in enumerate(f):
                        file_j_df = h5py.File(filename, 'r')
                        file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                        df = no_filt_sleap_h5(file_j_df, self.pose_chosen)
                        self.all_df.append(df)
                        self.new_data.append(file_j_processed)
                        self.filenames.append(filename)
                        self.folder.append(fd)
                    self.folders.append(fd)
        elif self.filetype == 'json':
            for i, fd in enumerate(self.new_directories):
                f = get_filenamesjson(self.root_path, fd)
                json2csv_multi(f)
                filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
                file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                        low_memory=False)
                file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                self.all_df.append(file_j_df)
                self.new_data.append(file_j_processed)
                self.filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
                self.folder.append(fd)
                self.folders.append(fd)
        print('Extracting features and predicting labels... ')
        labels_fs = []
        bar = st.progress(0)
        for i in range(0, len(self.new_data)):
            feats_new = bsoid_extract([self.new_data[i]], self.new_framerate)
            labels = bsoid_predict(feats_new, self.clf)
            for m in range(0, len(labels)):
                labels[m] = labels[m][::-1]
            labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
            for n, l in enumerate(labels):
                labels_pad[n][0:len(l)] = l
                labels_pad[n] = labels_pad[n][::-1]
                if n > 0:
                    labels_pad[n][0:n] = labels_pad[n - 1][0:n]
            labels_fs.append(labels_pad.astype(int))
            bar.progress(round((i + 1) / len(self.new_data) * 100))
        print('Frameshift arrangement of predicted labels...')
        for k in range(0, len(labels_fs)):
            labels_fs2 = []
            for l in range(math.floor(self.new_framerate / 10)):
                labels_fs2.append(labels_fs[k][l])
            self.new_predictions.append(np.array(labels_fs2).flatten('F'))
        print('Done frameshift-predicting a total of **{}** files.'.format(len(self.new_data)))
        for i in range(0, len(self.new_predictions)):
            filename_i = os.path.basename(self.filenames[i]).rpartition('.')[0]
            fs_labels_pad = np.pad(self.new_predictions[i], (0, len(self.all_df[i]) -
                                                             len(self.new_predictions[i])), 'edge')
            df2 = pd.DataFrame(fs_labels_pad, columns={'B-SOiD labels'})
            frames = [df2, self.all_df[i]]
            xyfs_df = pd.concat(frames, axis=1)
            runlen_df, dur_stats, tm_array, tm_df, tm_norm = statistics.main(self.new_predictions[i],
                                                                             len(np.unique(self.predictions)))
            try:
                os.mkdir(str.join('', (self.new_root_path, self.folder[i], '/BSOID')))
            except FileExistsError:
                pass
            if any('Labels tagged onto pose files' in o for o in self.options):
                xyfs_df.to_csv(os.path.join(
                    str.join('', (self.new_root_path, self.folder[i], '/BSOID')),
                    str.join('', (self.new_prefix, 'labels_pose_', str(self.new_framerate),
                                  'Hz', filename_i, '.csv'))),
                    index=True, chunksize=10000, encoding='utf-8')
                print('Saved Labels .csv in {}'.format(
                    str.join('', (self.new_root_path, self.folder[i], '/BSOID'))))
            if any('Group durations (in frames)' in o for o in self.options):
                runlen_df.to_csv(os.path.join(
                    str.join('', (self.new_root_path, self.folder[i], '/BSOID')),
                    str.join('', (self.new_prefix, 'bout_lengths_', str(self.new_framerate),
                                  'Hz', filename_i, '.csv'))),
                    index=True, chunksize=10000, encoding='utf-8')
                print('Saved Group durations .csv in {}'.format(
                    str.join('', (self.new_root_path, self.folder[i], '/BSOID'))))
            if any('Transition matrix' in o for o in self.options):
                tm_df.to_csv(os.path.join(
                    str.join('', (self.new_root_path, self.folder[i], '/BSOID')),
                    str.join('', (self.new_prefix, 'transitions_mat_',
                                  str(self.new_framerate), 'Hz', filename_i, '.csv'))),
                    index=True, chunksize=10000, encoding='utf-8')
                print('Saved transition matrix .csv in {}'.format(
                    str.join('', (self.new_root_path, self.folder[i], '/BSOID'))))
        with open(os.path.join(self.working_dir, str.join('', (self.new_prefix, '_predictions.sav'))), 'wb') as f:
            joblib.dump([self.folders, self.folder, self.filenames, self.new_data, self.new_predictions], f)
        print('**_CHECK POINT_**: Done predicting old/new files. Move on to '
                        '__Load up analysis app (please close current browser when new browser pops up)__.')

    def main(self):
        self.setup()
        self.predict()
