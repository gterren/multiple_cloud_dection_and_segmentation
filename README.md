# Detection of Clouds in Multiple Wind Velocity Fields

This repository includes the files necessary for the implementation of the cloud dection algorithm explained in article https://arxiv.org/pdf/2105.03535.pdf.

## Mixture Models

The mixture models with different distribution that were implemented in the article are in the file detection_clustering_utils.py.

## Cloud Segmentation

The cloud segmentation algorithms in the repositoty https://github.com/gterren/cloud_segmentation are implemented for real time applications in this file cloud_segmentation_utils.py.

## Velocity Vectors

Implementations of different optical flow algorithms are in file lib_motion_vector.py. These include: Pyramidal Lucas-Kanade, Weithed Lucas-Kanade, Pyramidal Weighted Lucas-Kanade, Cross-Correlation, Normalized Cross-Correlation, Honk-Schunk and more...

## Dataset

A sample dataset is publicaly available in DRYAD repository: https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m
