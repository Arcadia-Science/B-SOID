### Important note:
#### This version of B-SOiD has been modified to enable running through the CLI. It has only been tested using CSV files from DeepLabCut with corresponding .mp4 videos. We have tried to enable the other file types used in the original code, but cannot guarantee they will work smoothly. The usage section has been modified to reflect the CLI usage. We have not included functionality for launching bsoid_analysis.py (the final step in the original pipeline).
<br>

![B-SOiD flowchart](demo/appv2_files/bsoid_version2.png)
[![DOI](https://zenodo.org/badge/196603884.svg)](https://zenodo.org/badge/latestdoi/196603884)

![](demo/appv2_files/bsoid_mouse_openfield1.gif)
![](demo/appv2_files/bsoid_mouse_openfield2.gif)
![](demo/appv2_files/bsoid_exercise.gif)

### Why B-SOiD ("B-side")?
[DeepLabCut](https://github.com/AlexEMG/DeepLabCut) <sup>1,2,3</sup>,
[SLEAP](https://github.com/murthylab/sleap) <sup>4</sup>, and
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) <sup>5</sup>
have revolutionized the way behavioral scientists analyze data.
These algorithm utilizes recent advances in computer vision and deep learning to automatically estimate 3D-poses.
Interpreting the positions of an animal can be useful in studying behavior;
however, it does not encompass the whole dynamic range of naturalistic behaviors.

B-SOiD identifies behaviors using a unique pipeline where unsupervised learning meets supervised classification.
The unsupervised behavioral segmentation relies on non-linear dimensionality reduction <sup>6,7,9,10</sup>,
whereas the supervised classification is standard scikit-learn <sup>8</sup>.

Behavioral segmentation of open field in DeepLabCut, or B-SOiD ("B-side"), as the name suggested,
 was first designed as a pipeline using pose estimation file from DeepLabCut as input. Now, it has extended to handle
DeepLabCut (.h5, .csv), SLEAP (.h5), and OpenPose (.json) files.

### Installation

#### Step 1: Install Conda/Mamba

This repository uses conda to manage the development software environment.

You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/). After installing conda and [mamba](https://mamba.readthedocs.io/en/latest/), proceed with the next steps.

#### Step 2: Clone B-SOID repository

Git clone the web URL (example below) or download ZIP.

Change your current working directory to the location where you want the cloned directory to be made.
```bash
git clone https://github.com/Arcadia-Science/B-SOID.git
```

### Usage
#### Step 1: Setup, open an anaconda/python3 instance and install dependencies with the requirements file
```
cd /path/to/B-SOID/
```

For MacOS users (note: you need a Mac with an Intel chip or use the Rosetta emulator on an Apple silicon based Macs):
```bash
conda env create -n bsoid_v2 -f requirements.yaml (macOS)
```

or for Windows users:
```bash
conda env create -n bsoid_v2 -f requirements_win.yaml (windows)
```

or for Linux users (this was tested with Ubuntu version 22.2):

Linux `ffmpeg` distribution on Conda doesn't work (it has missing libraries). Because of this, you need to install `ffmpeg` separately from the Conda environment setup. First, let's do that:

```bash
sudo apt update && sudo apt upgrade
sudo apt install ffmpeg # At this time, the most recent ffmpeg version is 4.2.2
```

Then let's create the Conda environment. Sadly Conda/Mamba is not able to resolve the dependencies when the channel_priority setting is set to `strict`. So, we need to change that:

```bash
# Adjust the channel_priority setting
conda config --set channel_priority flexible

# Create the conda environment
conda env create -n bsoid_v2 -f requirements_linux.yaml

# Optionally update the channel_priority setting to be strict
conda config --set channel_priority strict
```

Once your setup is complete, activate the Conda environment:
```bash
conda activate bsoid_v2
```

You should now see (bsoid_v2) $yourusername@yourmachine ~ %

#### Step 2: Run the pipeline through the CLI!
##### Note: If you are running on an AWS instance and would like to see outputs from the web application, you will need to add an inbound rule to your security group with Type: “Custom TCP Rule”, Port Range:8501, and Source: MyIP. Seeing the web application can be useful for debugging since errors sometimes only appear there. Port 8501 is the custom port used by Streamlit. However, users should be careful to make sure that this is the right port when they run the app; for example, if you incorrectly shutdown so the process is running the background and you run the streamlit app again, it will select the next port.
##### Environment variables that need to be specified to run the CLI command:

`WORKING_DIR_BSOID`: Path to where you want outputs to go.<br>
`PREFIX_BSOID`: Prefix that you want appended to output files.<br>
`SOFTWARE_BSOID`: Pose estimation software; must be 'DeepLabCut','SLEAP', or 'OpenPose'.<br>
`FTYPE_BSOID`: File type; DeepLabCut: 'h5' or 'csv', SLEAP: 'h5', and OpenPose: 'json'.<br>
`ROOT_PATH_BSOID`: Path to the working directory containing sub-directories that have input .csv,.h5, or .json files.<br>
`FRAMERATE_BSOID`: Framerate for pose estimate files; in frames per second.<br>
`VALUE_BSOID`: Number of sub-directories that have input .csv,.h5, or .json files.<br>
`DATA_DIR_BSOID`: Path of sub-directories that have input .csv,.h5, or .json files; provided as a list and relative to working_dir, e.g. '/1_1,/1_2,/2_1,/2_2'.<br>
`MIN_CLUSTER_BSOID`: default='0.5',Minimum cluster size, based on minimum temporal bout and will represent a %. Impacts number of clusters.<br>
`MAX_CLUSTER_BSOID`: default='1', Maximum cluster size, will represent a % and impacts number of clusters.<br>
`AUTOSAVE_BSOID`: Whether or not you want to autosave clustering as you go. Should be 'Yes' or 'No'. Default is 'Yes'.<br>
`POSE_LIST_BSOID`: List of poses to include in analysis, for example, 'R_rear,L_rear'.<br>
`MIN_TIME_BSOID`: Minimum time for bout in ms. Default is 200.<br>
`NUMBER_EXAMPLES_BSOID`: Number of non-repeated examples for video snippets. Default is 5. Decreasing this number will speed up runtime because fewer example gifs will be generated per cluster.<br>
`PLAYBACK_SPEED_BSOID`: Playback speed for video snippets. Default is 0.75X.<br>
`FRACTION_BSOID`: Training input fraction (do not change this value if you wish to generate the side-by-side video seen on B-SOiD GitHub page). Default is 1, minimum is 0.1, maximum is 1.<br>

##### Creating a new model:
```
SOFTWARE_BSOID='DeepLabCut' FTYPE_BSOID='csv' ROOT_PATH_BSOID='/Users/Desktop/training/' FRAMERATE_BSOID=120 WORKING_DIR_BSOID='/Users/Desktop/training/output' PREFIX_BSOID='controltry' NUMBER_EXAMPLES_BSOID=1 VALUE_BSOID=4.0 DATA_DIR_BSOID='/1_1,/1_2,/2_1,/2_2' AUTOSAVE_BSOID='Yes' POSE_LIST_BSOID='R_rear,L_rear' python run_streamlit_cli.py
```
<br>

##### Predicting files using a model:

Your environmental variables should be set to match the directories that contain the trained model. Only `WORKING_DIR_BSOID`, `PREFIX_BSOID`, `FTYPE_BSOID`, `ROOT_PATH_BSOID`, `FRAMERATE_BSOID`, and `DATA_DIR_BSOID` need to be defined for prediction. Outputs from prediction will be within those directories in a folder named BSOID.

```
FTYPE_BSOID='csv' ROOT_PATH_BSOID='/Users/Desktop/training/' FRAMERATE_BSOID=120 WORKING_DIR_BSOID='/Users/Desktop/training/output' PREFIX_BSOID='controltry' DATA_DIR_BSOID='/1_1,/1_2,/2_1,/2_2' python run_streamlit_cli_predict.py
```
<br>

#### Resources
We have provided our 6 body part [DeepLabCut model](yttri-bottomup_dlc-model/dlc-models/).
We also included two example 5 minute clips
([labeled_clip1](yttri-bottomup_dlc-model/examples/raw_clip1DLC_resnet50_OpenFieldHighResApr8shuffle1_1030000_labeled.mp4),
[labeled_clip2](yttri-bottomup_dlc-model/examples/raw_clip2DLC_resnet50_OpenFieldHighResApr8shuffle1_1030000_labeled.mp4))
as proxy for how well we trained our model.
The raw video
([raw_clip1](yttri-bottomup_dlc-model/examples/raw_clip1.mp4),
[raw_clip2](yttri-bottomup_dlc-model/examples/raw_clip2.mp4))
and the corresponding [h5/pickle/csv](yttri-bottomup_dlc-model/examples/) files are included as well.



#### Archives
* [matlab](docs/matlab_tutorial.md)
* [python-tsne](docs/python3_tutorial.md)
* [python-umap](docs/bsoid_umap_tutorial.md)
* [bsoid app version 1](docs/bsoid_app_init.md)

### Contributing

Pull requests are welcome. For recommended changes that you would like to see, open an issue.
Join our [slack group](https://join.slack.com/t/b-soid/shared_invite/zt-dksalgqu-Eix8ZVYYFVVFULUhMJfvlw)
for more instantaneous feedback.

There are many exciting avenues to explore based on this work.
Please do not hesitate to contact us for collaborations.

### License

This software package provided without warranty of any kind and is licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
If you use our algorithm and/or model/data, please cite us! Preprint/peer-review will be announced in the following section.

### News
September 2019: First B-SOiD preprint on [bioRxiv](https://www.biorxiv.org/content/10.1101/770271v1)

March 2020: Updated version of our preprint on [bioRxiv](https://www.biorxiv.org/content/10.1101/770271v2)

#### References
1. [Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018 Sep;21(9):1281-1289. doi: 10.1038/s41593-018-0209-y. Epub 2018 Aug 20. PubMed PMID: 30127430.](https://www.nature.com/articles/s41593-018-0209-y)

2. [Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc. 2019 Jul;14(7):2152-2176. doi: 10.1038/s41596-019-0176-0. Epub 2019 Jun 21. PubMed PMID: 31227823.](https://doi.org/10.1038/s41596-019-0176-0)

3. [Insafutdinov E., Pishchulin L., Andres B., Andriluka M., Schiele B. (2016) DeeperCut: A Deeper, Stronger, and Faster Multi-person Pose Estimation Model. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9910. Springer, Cham](http://arxiv.org/abs/1605.03170)

4. [Pereira, Talmo D., Nathaniel Tabris, Junyu Li, Shruthi Ravindranath, Eleni S. Papadoyannis, Z. Yan Wang, David M. Turner, et al. 2020. “SLEAP: Multi-Animal Pose Tracking.” bioRxiv.](https://doi.org/10.1101/2020.08.31.276246)

5. [Cao Z, Hidalgo Martinez G, Simon T, Wei SE, Sheikh YA. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. IEEE Trans Pattern Anal Mach Intell. 2019 Jul 17. Epub ahead of print. PMID: 31331883.](https://doi.org/10.1109/TPAMI.2019.2929257).

6. [McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.](http://arxiv.org/abs/1802.03426)

7. [McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. The Journal of Open Source Software, 2(11), 205.](https://doi.org/10.21105/joss.00205)

8. [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.](http://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)

9. [L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research 15(Oct):3221-3245, 2014.](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

10. [Chen M. EM Algorithm for Gaussian Mixture Model (EM GMM). MATLAB Central File Exchange. Retrieved July 15, 2019.](https://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model-em-gmm)
