# Glacier-Aware EA-LSTM Streamflow Prediction
This repository implements an Entity-Aware LSTM (EA-LSTM) to predict daily streamflow in Western Canadian basins. It explicitly models static basin attributes—specifically glacier coverage, mean elevation, and basin area.

## Local Setup & Preprocessing
Before training on the cluster, data must be downloaded and preprocessed locally.

1. **Install Dependencies**
   Ensure you have Python 3.10+ installed, then run:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
2. **Run Preprocessing**
   Run `notebooks/01_data_preprocessing.ipynb` in full.
   * Note: This notebook downloads ERA5 reanalysis data which can take a significant amount of time depending on the server queues.
   * Outcome: This generates the lightweight CSVs in `data/processed/` required for training.

## High Performance Compute Setup (UBC ARC Sockeye)
To train the EA-LSTM model, this project utilized UBC ARC Sockeye. The following steps can be followed to set up this project on Sockeye:

1. **Create a Secrets File**
   Create a file named `secrets.env` in the project root containing your email and Sockeye allocation code.
   ```
   # secrets.env
   EMAIL="<your email>"
   ACCOUNT="<alloc-code>-gpu"
   ```
   Note: The -gpu suffix is required for running in a GPU environment.
2. **Bundle the Project**
   Run the bundle script, `python bundle_project.py`, in your terminal from the project root. This creates a zip file excluding the raw data.
3. **Upload to Sockeye**
   Run `scp project_upload.zip <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/` replacing `<cwl>` with your UBC CWL and `<alloc-code>` with your Sockeye allocation code.
   Note that to connect to Sockeye you must be connected to a UBC secure network or connect to [UBC myVPN](https://it.ubc.ca/services/email-voice-internet/myvpn/setup-documents)
4. **Connect and Extract**
   SSH into Sockeye and unzip the project.
   ```
   ssh <cwl>@sockeye.arc.ubc.ca
   cd /scratch/<alloc-code>
   unzip project_upload.zip -d ealstm_project
   cd ealstm_project
   ```
5. **Setup Environment (One-time)**
   This script loads the required Python modules, creates a virtual environment, and installs dependencies.
   ```
   chmod +x hpc/setup_env.sh
   ./hpc/setup_env.sh
   ```
6. **Submit the Job**
   The submit script automatically handles directory setup, secrets injection, and SLURM submission.
   ```
   chmod +x hpc/submit.sh
   ./hpc/submit.sh
   ```

## Monitoring and Results
* **Check Status:** Run `squeue -u <cwl>` to see your job in the queue.
* **View Logs:** Once running, track progress live: `tail -f logs/train_*.out`
* **Retrieve Results:** After training, download the predictions to your local machine:
  ```scp <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/ealstm_project/data/processed/test_set_predictions.csv ./data/processed/```
   