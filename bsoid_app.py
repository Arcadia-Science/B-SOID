from streamlit import caching

from bsoid_app import data_preprocess, extract_features, clustering, machine_learner, \
    export_training, video_creator, predict
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_css import local_css
from bsoid_app.bsoid_utilities.load_workspace import *


def streamlit_run(pyfile):
    os.system("streamlit run {}.py".format(pyfile))
working_dir = os.environ.get('working_dir', '')
prefix = os.environ.get('prefix', '')

st.set_page_config(page_title='B-SOiD v2.0', page_icon="🐁",
                   layout='wide', initial_sidebar_state='auto')
local_css("bsoid_app/bsoid_utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>B-SOID</span></span> " \
        "   <span class='h2'>--version 2.0 🐁</span> </div>"
st.markdown(title, unsafe_allow_html=True)
st.text('')
processor = data_preprocess.preprocess()
processor.compile_data()
working_dir, prefix = query_workspace()
[_, _, framerate, _, _, _, processed_input_data, _] = load_data(working_dir, prefix)
extractor = extract_features.extract(working_dir, prefix, processed_input_data, framerate)
extractor.main()
[_, sampled_embeddings] = load_embeddings(working_dir, prefix)
clusterer = clustering.cluster(working_dir, prefix, sampled_embeddings)
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

