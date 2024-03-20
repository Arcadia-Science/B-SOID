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
    MIN_TIME = int(get_env_variable('MIN_TIME_BSOID','200'))
    NUMBER_EXAMPLES = int(get_env_variable('NUMBER_EXAMPLES_BSOID', '5'))
    PLAYBACK_SPEED = float(get_env_variable('PLAYBACK_SPEED_BSOID', '0.75'))
    FRACTION = float(get_env_variable('FRACTION_BSOID', '1'))

except ValueError as e:
    print(e)
    exit(1)  # Exit if any required variable is missing or if a conversion to float fails

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

for directory in DATA_DIRECTORIES:
    full_directory_path = os.path.join(ROOT_PATH, directory)
    # Create a new instance of Creator for the current directory
    creator = video_creator.Creator(
        ROOT_PATH, [full_directory_path], processed_input_data, pose_chosen,
        full_directory_path, PREFIX, FRAMERATE, MIN_TIME, NUMBER_EXAMPLES, 
        PLAYBACK_SPEED, clf, input_filenames,FTYPE)
    
    creator.main()
