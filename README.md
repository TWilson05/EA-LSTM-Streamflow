# Glacier-Aware EA-LSTM Streamflow Prediction

## Setup Instructions:
This pipeline requires a significant amount of data and preprocessing. The download links and preprocessing instructions are outlined in `notebooks/01_data_preprocessing.ipynb`. This notebook should be run in full as a first step in order to download the required data and process it into a lighter weight dataset. Note that the ERA5 data can take a significant amount of time to download.

## High Performance Compute Setup (UBC ARC Sockeye)
To train the EA-LSTM model, this project utilized UBC ARC Sockeye. The following steps can be followed to set up this project on Sockeye:

1. **Extract project to a zip file**
   Run the following script in your terminal in your root directory: `zip -r project.zip . \
    -x "data/raw/*" \
    -x "**/__pycache__/*" \
    -x "*.egg-info/*" \
    -x "*.pyc"`