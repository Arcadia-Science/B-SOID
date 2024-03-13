import os

from streamlit import caching

from bsoid_app import predict
from bsoid_app.bsoid_utilities import visuals
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
    FTYPE = get_env_variable('FTYPE_BSOID')
    ROOT_PATH = get_env_variable('ROOT_PATH_BSOID')
    DATA_DIRECTORIES = get_env_variable('DATA_DIR_BSOID')

except ValueError as e:
    print(e)
    exit(1)  # Exit if any required variable is missing or if a conversion to float fails

[ROOT_PATH, DATA_DIRECTORIES, FRAMERATE, pose_chosen, input_filenames, _, processed_input_data, _] \
    = load_data(WORKING_DIR, PREFIX)
[_, _, _, clf, _, predictions] = load_classifier(WORKING_DIR, PREFIX)

predictor = predict.Prediction(ROOT_PATH, DATA_DIRECTORIES, input_filenames, processed_input_data, WORKING_DIR,
                               PREFIX, FRAMERATE, pose_chosen, predictions, FTYPE, clf)
predictor.main()
