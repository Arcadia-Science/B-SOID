import os

import hdbscan
import joblib
import numpy as np
import streamlit as st
import randfacts

from bsoid_app.config import *
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_workspace import load_clusters
from streamlit import caching

class Cluster:

    def __init__(self, WORKING_DIR, PREFIX, sampled_embeddings, AUTOSAVE, MIN_CLUSTER_RANGE, MAX_CLUSTER_RANGE):
        print('IDENTIFY AND TWEAK NUMBER OF CLUSTERS.')
        self.WORKING_DIR = WORKING_DIR
        self.PREFIX = PREFIX
        self.sampled_embeddings = sampled_embeddings
        self.cluster_range = []
        self.AUTOSAVE = AUTOSAVE
        self.MIN_CLUSTER_RANGE = MIN_CLUSTER_RANGE
        self.MAX_CLUSTER_RANGE = MAX_CLUSTER_RANGE
        self.cluster_range = [MIN_CLUSTER_RANGE, MAX_CLUSTER_RANGE]
        print(f'Your cluster range is set to {self.cluster_range}')
        self.assignments = []
        self.assign_prob = []
        self.soft_assignments = []

    def hierarchy(self):
        max_num_clusters = -np.infty
        num_clusters = []
        self.min_cluster_size = np.linspace(self.cluster_range[0], self.cluster_range[1], 25)

        for min_c in self.min_cluster_size:
            learned_hierarchy = hdbscan.HDBSCAN(
                prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * self.sampled_embeddings.shape[0])),
                **HDBSCAN_PARAMS).fit(self.sampled_embeddings)
            num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
            if num_clusters[-1] > max_num_clusters:
                max_num_clusters = num_clusters[-1]
                retained_hierarchy = learned_hierarchy
        self.assignments = retained_hierarchy.labels_
        self.assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
        self.soft_assignments = np.argmax(self.assign_prob, axis=1)
        print('Done assigning labels for {} instances ({} minutes) '
                'in {} D space'.format(self.assignments.shape,
                                           round(self.assignments.shape[0] / 600),
                                           self.sampled_embeddings.shape[1]))

    def show_classes(self):
        print('Showing {}% data that were confidently assigned.'
                 ''.format(round(self.assignments[self.assignments >= 0].shape[0] /
                                 self.assignments.shape[0] * 100)))
        fig1, plt1 = visuals.plot_classes(self.sampled_embeddings[self.assignments >= 0],
                                          self.assignments[self.assignments >= 0])
        plt1.suptitle('HDBSCAN assignment')
        col1, col2 = st.beta_columns([2, 2])
        col1.pyplot(fig1)
        fig1.savefig('hdbscan_assignment.png', dpi=300, bbox_inches='tight')


    def save(self):
        if self.AUTOSAVE == 'Yes':
            print('You have chosen to AUTOSAVE clustering as you go (overwrites previously saved clustering).')
            with open(os.path.join(self.WORKING_DIR, str.join('', (self.PREFIX, '_clusters.sav'))), 'wb') as f:
                joblib.dump([self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments], f)

    def main(self):
        try:
            caching.clear_cache()
            [self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments] = \
                load_clusters(self.WORKING_DIR, self.PREFIX)
            print(
                'Done assigning labels for {} instances in {} D space. Move on to create '
                'a model.'.format(self.assignments.shape, self.sampled_embeddings.shape[1]))
            print('Your last saved run range was {}% to {}%'.format(self.min_cluster_size[0],
                                                                                  self.min_cluster_size[-1]))
            caching.clear_cache()
            self.hierarchy()
            self.save()
            self.show_classes()
        except (AttributeError, FileNotFoundError) as e:
            self.hierarchy()
            self.save()
            self.show_classes()
