import base64
import ffmpeg
import h5py
import streamlit as st

from bsoid_app.bsoid_utilities.bsoid_classification import *
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *
from bsoid_app.bsoid_utilities.videoprocessing import *


@st.cache(allow_output_mutation=True)
def selected_file(d_file):
    return d_file


@st.cache(allow_output_mutation=True)
def selected_vid(vid_file):
    return vid_file


class Creator:

    def __init__(self, ROOT_PATH, DATA_DIRECTORIES, processed_input_data, pose_chosen, file_directory, PREFIX, FRAMERATE, MIN_TIME, FTYPE, NUMBER_EXAMPLES, PLAYBACK_SPEED, clf, input_filenames):
        print('GENERATE VIDEOS SNIPPETS FOR INTERPRETATION')
        self.root_path = ROOT_PATH
        self.data_directories = DATA_DIRECTORIES
        self.processed_input_data = processed_input_data
        self.pose_chosen = pose_chosen
        self.prefix = PREFIX
        self.min_time = MIN_TIME
        self.framerate = FRAMERATE
        self.playback_speed = PLAYBACK_SPEED
        self.clf = clf
        self.input_filenames = input_filenames
        self.file_directory = file_directory
        self.d_file = []
        self.vid_dir = []
        self.vid_file = []
        self.frame_dir = []
        self.filetype = FTYPE
        self.width = []
        self.height = []
        self.bit_rate = []
        self.num_frames = []
        self.avg_frame_rate = []
        self.shortvid_dir = []
        self.min_frames = []
        self.number_examples = NUMBER_EXAMPLES
        self.out_fps = []
        self.file_j_processed = []
 
    def setup(self):
        try:
            full_directory_path = os.path.join(self.root_path, self.file_directory.lstrip('/'))  # Strip leading '/' if present
            os.listdir(full_directory_path)
            print('You have selected {} as your csv/h5/json data sub-directory.'.format(self.file_directory))
        except FileNotFoundError:
            st.error('No such directory')
            return
        print('If your input was openpose **JSON(s)**, the app has converted into a SINGLE CSV for each folder. '
              'Hence, the following will autodetect CSV as your filetype.')
        if self.filetype in ['csv', 'h5']:
            print(full_directory_path)
            d_file = [os.path.join(full_directory_path, file) for file in os.listdir(full_directory_path) if file.endswith('.csv') or file.endswith('.h5')]
            self.d_file = selected_file(d_file)
            for file_path in self.d_file:
                csvname = os.path.basename(file_path).rpartition('.')[0]
                print(csvname)
        elif self.filetype == 'json':
            d_files = get_filenamesjson(self.root_path, self.file_directory)
            fname = d_files[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
            if not os.path.isfile(str.join('', (d_files[0].rpartition('/')[0], '/', fname, '.csv'))):
                json2csv_multi(d_files)
            directory_path = os.path.join(self.root_path, self.file_directory)
            d_file = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]
            self.d_file = selected_file(d_file)
        self.vid_dir = (str.join('', (self.root_path, self.file_directory)))
        try:
            os.listdir(self.vid_dir)
            print('You have selected {} as your video directory.'.format(self.vid_dir))
        except FileNotFoundError:
            print('No such directory')
        vid_file = [file for file in os.listdir(self.vid_dir) if file.endswith('.mp4') or file.endswith('.avi')]
        self.vid_file = selected_vid(vid_file)
        for vid_file_path in self.vid_file:
            full_vid_path = os.path.join(self.vid_dir, vid_file_path)
            try:
                probe = ffmpeg.probe(full_vid_path)
            except ffmpeg.Error as e:
                print(f'Error probing video file {vid_file_path}: {e}')
        if self.filetype == 'csv' or self.filetype == 'h5':
            print('You have selected {} matching {}.'.format(self.vid_file, self.d_file))
            for file_path in self.d_file:
                csvname = os.path.basename(file_path).rpartition('.')[0]
                print(csvname)
        else:
            print('You have selected {} matching {} json directory.'.format(self.vid_file, self.file_directory))
            csvname = os.path.basename(self.file_directory)
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/pngs')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/pngs', '/', csvname)))
        except FileExistsError:
            pass
        self.frame_dir = str.join('', (self.root_path, self.file_directory, '/pngs', '/', csvname))
        print('Created {} as your video frames directory.'.format(self.frame_dir, self.vid_file))
        for vid_file_path in self.vid_file:
            full_vid_path = os.path.join(self.vid_dir, vid_file_path)
            try:
                probe = ffmpeg.probe(full_vid_path)
                video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_info:
                    print(f"Processing video info for {vid_file_path}")
                    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                    self.width = int(video_info['width'])
                    self.height = int(video_info['height'])
                    self.num_frames = int(video_info['nb_frames'])
                    self.bit_rate = int(video_info['bit_rate'])
                    self.avg_frame_rate = round(
                        int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
            except ffmpeg.Error as e:
                print(f'Error probing video file {vid_file_path}: {e}')        
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/mp4s')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/mp4s', '/', csvname)))
        except FileExistsError:
            pass
        self.shortvid_dir = str.join('', (self.root_path, self.file_directory, '/mp4s', '/', csvname))
        print('Created {} as your behavioral snippets directory.'.format(self.shortvid_dir, self.vid_file))
        self.min_frames = round(float(self.min_time) * 0.001 * float(self.framerate))  
        print('Entered {} ms as minimum duration per bout, '
              'which is equivalent to {} frames.'.format(self.min_time, self.min_frames))
        print('Your will obtain a maximum of {} non-repeated output examples per group.'.format(self.number_examples))
        self.out_fps = int(float(self.playback_speed) * float(self.framerate))
        print('Playback at {} x speed (rounded to {} FPS).'.format(self.playback_speed, self.out_fps))
        
    def frame_extraction(self):
        print('Extracting frames from videos...')
        for vid_file_path in self.vid_file:
            full_vid_path = os.path.join(self.vid_dir, vid_file_path)
            try:
                probe = ffmpeg.probe(full_vid_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream is None:
                    print(f"No video stream found in {vid_file_path}")
                    continue
                avg_frame_rate = eval(video_stream['avg_frame_rate'])
                num_frames = int(video_stream['nb_frames'])
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                bit_rate = int(video_stream['bit_rate'])
                frames_dir = os.path.join(self.frame_dir, os.path.splitext(vid_file_path)[0] + '_frames')
                os.makedirs(frames_dir, exist_ok=True)
                (ffmpeg
                 .input(full_vid_path)
                 .filter('fps', fps=avg_frame_rate)
                 .output(os.path.join(frames_dir, 'frame%05d.png'), video_bitrate=bit_rate,
                         s=f"{int(width * 0.5)}x{int(height * 0.5)}", sws_flags='bilinear', start_number=0)
                 .run(capture_stdout=True, capture_stderr=True))
                print(f'Done extracting {num_frames} frames from video {vid_file_path} into {frames_dir}')
            except ffmpeg.Error as e:
                print(f'Error extracting frames from {vid_file_path}:')
                print('ffmpeg stdout:', e.stdout.decode('utf8'))
                print('ffmpeg stderr:', e.stderr.decode('utf8'))
        print('Finished extracting frames from all videos.')
    
    def create_videos(self):
        self.frame_extraction()
        self.file_j_processed = []
        for file_path in self.d_file:
            full_file_path = os.path.join(self.root_path, self.file_directory, file_path)
            if self.filetype == 'csv' or self.filetype == 'json':
                file_j_df = pd.read_csv(full_file_path, low_memory=False)
            elif self.filetype == 'h5':
                try:
                    file_j_df = pd.read_hdf(full_file_path, low_memory=False)
                except:
                    st.info('Detecting a SLEAP .h5 file...')
                    file_j_df = h5py.File(full_file_path, 'r')
            if self.filetype in ['csv', 'json']:
                file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
            elif self.filetype == 'h5':
                file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen) if not isinstance(file_j_df, h5py.File) \
                    else adp_filt_sleap_h5(file_j_df, self.pose_chosen)
            self.file_j_processed.append(file_j_processed)
        labels_fs = []
        fs_labels = []
        print('Predicting labels... ')
        for i in range(0, len(self.file_j_processed)):
            feats_new = bsoid_extract([self.file_j_processed[i]], self.framerate)
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
        print('Frameshifted arrangement of labels... ')
        for vid_file in self.vid_file:
            vid_name_without_extension = os.path.splitext(vid_file)[0]
            new_frame_dir = os.path.join(self.frame_dir, f"{vid_name_without_extension}_frames")
            self.frame_dir = new_frame_dir
            os.makedirs(self.frame_dir, exist_ok=True)
        print(self.frame_dir)
        print(self.min_frames)
        for k in range(0, len(labels_fs)):
            labels_fs2 = []
            for l in range(math.floor(self.framerate / 10)):
                labels_fs2.append(labels_fs[k][l])
                fs_labels.append(np.array(labels_fs2).flatten('F'))
            print('Done frameshift-predicting **{}**.'.format(self.d_file))
            print("Final fs_labels content:", fs_labels)
            if fs_labels:
                create_labeled_vid(fs_labels[0], int(self.min_frames), int(self.number_examples), int(self.out_fps), self.frame_dir, self.shortvid_dir)
                print('Done generating video snippets. Move on to Predict files using a model.')
            else:
                print("No labels available for creating video snippets.")
    
    def show_snippets(self):
        video_bytes = []
        grp_names = []
        files = []
        for file in os.listdir(self.shortvid_dir):
            files.append(file)
        sort_nicely(files)
        print('Creating gifs from mp4s...')
        for file in files:
            if file.endswith('0.mp4'):
                try:
                    example_vid_file = open(os.path.join(
                        str.join('', (self.shortvid_dir, '/', file.partition('.')[0], '.gif'))), 'rb')
                except FileNotFoundError:
                    convert2gif(str.join('', (self.shortvid_dir, '/', file)), TargetFormat.GIF)
                    example_vid_file = open(os.path.join(
                        str.join('', (self.shortvid_dir, '/', file.partition('.')[0], '.gif'))), 'rb')
                contents = example_vid_file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                video_bytes.append(data_url)
                grp_names.append('{}'.format(file.partition('.')[0]))
        col = [None] * 3
        col[0], col[1], col[2] = st.beta_columns([1, 1, 1])
        for i in range(0, len(video_bytes) + 3, 3):
            try:
                col[0].markdown(
                    f'<div class="container">'
                    f'<img src="data:image/gif;base64,{video_bytes[i]}" alt="" width="300" height="300">'
                    f'<div class="bottom-left">{grp_names[i]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                col[1].markdown(
                    f'<div class="container">'
                    f'<img src="data:image/gif;base64,{video_bytes[i + 1]}" alt="" width="300" height="300">'
                    f'<div class="bottom-left">{grp_names[i + 1]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                col[2].markdown(
                    f'<div class="container">'
                    f'<img src="data:image/gif;base64,{video_bytes[i + 2]}" alt="" width="300" height="300">'
                    f'<div class="bottom-left">{grp_names[i + 2]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except IndexError:
                pass

    def main(self):
        self.setup()
        self.create_videos()
        self.show_snippets()
