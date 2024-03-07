import os

from streamlit import caching

from bsoid_app import data_preprocess, extract_features, clustering, machine_learner, \
    export_training, video_creator, predict
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_css import local_css
from bsoid_app.bsoid_utilities.load_workspace import *

def get_env_variable(var_name, default_value=None):
    # Retrieve an environment variable. Ensure it is not empty if no default value is provided.
    value = os.environ.get(var_name, default_value)
    if default_value is None and not value:
        raise ValueError(f"Environment variable '{var_name}' is required and cannot be empty.")
    return value

try:
    WORKING_DIR = get_env_variable('WORKING_DIR_BSOID')
    PREFIX = get_env_variable('PREFIX_BSOID')
    FRAMERATE = float(get_env_variable('FRAMERATE_BSOID'))
    SOFTWARE_CHOICE = get_env_variable('SOFTWARE_BSOID')
    FTYPE = get_env_variable('FTYPE_BSOID')
    ROOT_PATH = get_env_variable('ROOT_PATH_BSOID')
    VALUE = get_env_variable('VALUE_BSOID')
    DATA_DIRECTORIES = get_env_variable('DATA_DIR_BSOID')
    MIN_CLUSTER_RANGE = float(get_env_variable('MIN_CLUSTER_BSOID', '0.5'))
    MAX_CLUSTER_RANGE = float(get_env_variable('MAX_CLUSTER_BSOID', '1'))
    AUTOSAVE = get_env_variable('AUTOSAVE_BSOID', 'Yes')
    POSE_LIST = get_env_variable('POSE_LIST_BSOID')
    MIN_TIME = get_env_variable('MIN_TIME','200')
    NUMBER_EXAMPLES = get_env_variable('NUMBER_EXAMPLES', '5')
    PLAYBACK_SPEED = get_env_variable('PLAYBACK_SPEED', '0.75')
    FRACTION = float(get_env_variable('FRACTION', '1'))

except ValueError as e:
    print(e)
    exit(1)  # Exit if any required variable is missing or if a conversion to float fails

st.set_page_config(page_title='B-SOiD v2.0', page_icon="🐁",
                   layout='wide', initial_sidebar_state='auto')
local_css("bsoid_app/bsoid_utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>B-SOID</span></span> " \
        "   <span class='h2'>--version 2.0 🐁</span> </div>"
st.markdown(title, unsafe_allow_html=True)
st.text('')


processor = data_preprocess.Preprocess(WORKING_DIR, PREFIX, SOFTWARE_CHOICE, FTYPE, ROOT_PATH, FRAMERATE, DATA_DIRECTORIES, POSE_LIST, VALUE)
processor.compile_data()

[_, _, FRAMERATE, _, _, _, processed_input_data, _] = load_data(WORKING_DIR, PREFIX)
extractor = extract_features.Extract(WORKING_DIR, PREFIX, processed_input_data, FRAMERATE, FRACTION)
extractor.main()

[_, sampled_embeddings] = load_embeddings(WORKING_DIR, PREFIX)
clusterer = clustering.Cluster(WORKING_DIR, PREFIX, sampled_embeddings, AUTOSAVE, MIN_CLUSTER_RANGE, MAX_CLUSTER_RANGE)
clusterer.main()

[sampled_features, _] = load_embeddings(WORKING_DIR, PREFIX)
[_, assignments, assign_prob, soft_assignments] = load_clusters(WORKING_DIR, PREFIX)
exporter = export_training.Export(WORKING_DIR, PREFIX, sampled_features,
                                  assignments, assign_prob, soft_assignments)
exporter.save_csv()

[features, _] = load_feats(WORKING_DIR, PREFIX)
[sampled_features, _] = load_embeddings(WORKING_DIR, PREFIX)
[_, assignments, _, _] = load_clusters(WORKING_DIR, PREFIX)
learning_protocol = machine_learner.Protocol(WORKING_DIR, PREFIX, features, sampled_features, assignments)
learning_protocol.main()

[ROOT_PATH, DATA_DIRECTORIES, FRAMERATE, pose_chosen, input_filenames, _, processed_input_data, _] \
    = load_data(WORKING_DIR, PREFIX)
[_, _, _, clf, _, _] = load_classifier(WORKING_DIR, PREFIX)
creator = video_creator.Creator(ROOT_PATH, DATA_DIRECTORIES, processed_input_data, pose_chosen,
                                WORKING_DIR, PREFIX, FRAMERATE, MIN_TIME, NUMBER_EXAMPLES, PLAYBACK_SPEED, clf, input_filenames)
creator.main()

[ROOT_PATH, DATA_DIRECTORIES, FRAMERATE, pose_chosen, input_filenames, _, processed_input_data, _] \
    = load_data(WORKING_DIR, PREFIX)
[_, _, _, clf, _, predictions] = load_classifier(WORKING_DIR, PREFIX)
predictor = predict.Prediction(ROOT_PATH, DATA_DIRECTORIES, input_filenames, processed_input_data, WORKING_DIR,
                               PREFIX, FRAMERATE, pose_chosen, predictions, clf)
predictor.main()

streamlit_run('./bsoid_app/bsoid_analysis')
