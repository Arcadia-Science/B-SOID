import os

from streamlit import caching

from bsoid_app import data_preprocess, extract_features, clustering, machine_learner, \
    export_training, video_creator, predict
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_css import local_css
from bsoid_app.bsoid_utilities.load_workspace import *

def get_env_variable(var_name, default_value=None):
    #Retrieve an environment variable. Ensure it is not empty if no default value is provided.
    value = os.environ.get(var_name, default_value)
    if default_value is None and not value:
        raise ValueError(f"Environment variable '{var_name}' is required and cannot be empty.")
    return value

try:
    working_dir = get_env_variable('WORKING_DIR_BSOID')
    prefix = get_env_variable('PREFIX_BSOID')
    framerate = float(get_env_variable('FRAMERATE_BSOID'))
    software_choice = get_env_variable('SOFTWARE_BSOID')
    ftype = get_env_variable('FTYPE_BSOID')
    root_path = get_env_variable('ROOT_PATH_BSOID')
    value = get_env_variable('VALUE_BSOID')
    data_directories = get_env_variable('DATA_DIR_BSOID')
    min_cluster_range = float(get_env_variable('MIN_CLUSTER_BSOID', '0.5'))
    max_cluster_range = float(get_env_variable('MAX_CLUSTER_BSOID', '1'))
    autosave = get_env_variable('AUTOSAVE_BSOID', 'Yes')
    pose_list = get_env_variable('POSE_LIST_BSOID')
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


processor = data_preprocess.preprocess(working_dir,prefix,software_choice,ftype,root_path,framerate,data_directories,pose_list,value)
processor.compile_data()

[_, _, framerate, _, _, _, processed_input_data, _] = load_data(working_dir, prefix)
extractor = extract_features.extract(working_dir, prefix, processed_input_data, framerate)
extractor.main()

[_, sampled_embeddings] = load_embeddings(working_dir, prefix)
clusterer = clustering.cluster(working_dir, prefix, sampled_embeddings,autosave, min_cluster_range,max_cluster_range)
clusterer.main()

[sampled_features, _] = load_embeddings(working_dir, prefix)
[_, assignments, assign_prob, soft_assignments] = load_clusters(working_dir, prefix)
exporter = export_training.export(working_dir, prefix, sampled_features,
                                  assignments, assign_prob, soft_assignments)
exporter.save_csv()

[features, _] = load_feats(working_dir, prefix)
[sampled_features, _] = load_embeddings(working_dir, prefix)
[_, assignments, _, _] = load_clusters(working_dir, prefix)
learning_protocol = machine_learner.protocol(working_dir, prefix, features, sampled_features, assignments)
learning_protocol.main()

[root_path, data_directories, framerate, pose_chosen, input_filenames, _, processed_input_data, _] \
    = load_data(working_dir, prefix)
[_, _, _, clf, _, _] = load_classifier(working_dir, prefix)
creator = video_creator.creator(root_path, data_directories, processed_input_data, pose_chosen,
                                working_dir, prefix, framerate, clf, input_filenames)
creator.main()

[root_path, data_directories, framerate, pose_chosen, input_filenames, _, processed_input_data, _] \
    = load_data(working_dir, prefix)
[_, _, _, clf, _, predictions] = load_classifier(working_dir, prefix)
predictor = predict.prediction(root_path, data_directories, input_filenames, processed_input_data, working_dir,
                               prefix, framerate, pose_chosen, predictions, clf)
predictor.main()

streamlit_run('./bsoid_app/bsoid_analysis')
