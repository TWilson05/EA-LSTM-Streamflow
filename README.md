# Glacier-Aware EA-LSTM Streamflow Prediction

## Setup Instructions:
This pipeline requires a significant amount of data and preprocessing. The download links and preprocessing instructions are outlined in `notebooks/01_data_preprocessing.ipynb`. This notebook should be run in full as a first step in order to download the required data and process it into a lighter weight dataset. Note that the ERA5 data can take a significant amount of time to download.

## High Performance Compute Setup (UBC ARC Sockeye)
To train the EA-LSTM model, this project utilized UBC ARC Sockeye. The following steps can be followed to set up this project on Sockeye:

1. **Extract project to a zip file**
   Run the following script in your terminal in your root directory: `python bundle_project.py` to create a zip file for the project.
2. **Upload the zip file to Sockeye**
   Run the command `scp project_upload.zip <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/` replacing `<cwl>` with your UBC CWL and `<alloc-code>` with your Sockeye allocation code.
   Note that to connect to Sockeye you must be connected to a UBC secure network or connect to [UBC myVPN](https://it.ubc.ca/services/email-voice-internet/myvpn/setup-documents)
3. **Connect and extract the module**
   SSH into Sockeye `ssh <cwl>@sockeye.arc.ubc.ca` and navigate to `cd /scratch/<alloc-code>`.
   Then unzip the folder and navigate to it by running `unzip project_upload.zip -d ealstm_project` and `cd ealstm_project`.
4. **Setup environment**
   