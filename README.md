# Vision and language representations in multimodal AI models and human social brain regions during natural movie viewing
 Code for Small et al. 2025 "Vision and language representations in multimodal AI models and human social brain regions during natural movie viewing." published in the second proceedings of UniReps, a workshop at NeurIPS. [Link to OpenReview](https://openreview.net/forum?id=pS1UjuYuJu#discussion)

The fMRI data for this paper will be released at a later date (the data is a subset of a dataset whose collection is on-going). This repository contains the conda environment and code necessary for running the encoding analyses in individual subjects and second level analyses, including plotting figures for the paper. 

## Conda Environment / Prerequisites
```conda env create -f environment.yml```

## Running the code
You can plot the figures using pre-computed analyses using the code in ```code/plot_second_project_2024.py``` with ```load=True``` for all plotting functions. 