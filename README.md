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

## File Tree
````
EA-LSTM-Streamflow/
├── data/
│   ├── output/
│   │   ├── area/
│   │   ├── baseline/
│   │   ├── phase_split/
│   │   └── topographic/
│   ├── processed/
│   │   ├── climate/
│   │   │   ├── daily_fraction_below_zero.csv
│   │   │   ├── daily_precipitation.csv
│   │   │   ├── daily_rainfall.csv
│   │   │   ├── daily_snowfall.csv
│   │   │   ├── daily_temp_max.csv
│   │   │   └── daily_temp_min.csv
│   │   ├── combined_streamflow.csv
│   │   ├── glacier_volume_change.csv
│   │   └── static_attributes.csv
│   └── raw/
│       ├── dem_data/
│       ├── drainage_areas/
│       ├── era5/
│       │   ├── precipitation/
│       │   └── temperature/
│       ├── mass_balance/
│       │   └── ts_monthly_const_area_lstm.csv
│       ├── RGI-western-canada/
│       ├── spatial_bounds.csv
│       └── station_metadata.csv
├── hpc/
│   ├── job.sh
│   ├── setup_env.sh
│   └── submit.sh
├── models/
│   ├── area/
│   ├── baseline/
│   ├── phase-split/
│   └── topographic/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── EXP_* (experimental analysis notebooks)
├── src/
│   ├── __init__.py
│   ├── climate.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── inference.py
│   ├── models.py
│   ├── processing.py
│   ├── spatial_utils.py
│   └── training.py
├── .gitignore
├── bundle_project.py
├── postprocessing_requirements.txt
├── README.md
├── requirements.txt
├── run_training.py
├── secrets.env
└── setup.py
````
**Files of particular note:**
* `data/processed/combined_streamflow.csv`: this is the ground-truth streamflow data in units of $m^3/s$. Note that there will be some gaps in this data.
* `data/processed/glacier_volume_change.csv`: this is the monthly changes in mass balance from the mass balance model aggregated for each station. Units are in millions of cubic meters of water (MCM).
* `data/processed/static_attributes.csv`: this is the values of static variables for each station. Area is in units of $km^2$, elevation is in $m$, and slope is unitless.
* `data/processed/climate/`: this folder contains CSV files of the dynamic variables, structured in the same way as `combined_streamflow.csv`. Temperature has units of degrees celcius while precipitation variables are in units of millimeters averaged over the basin.
* `data/output` contains daily predictions for each model, structured in the same manner as `combined_streamflow` except using units of millimeters over the basin area.
* `src/config.py` is used to consistantly reference common files and directories. Include this in your import statment when developing code or performing analysis. `from src.config import ___`.

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
   *Troubleshooting*: If you recieve an error, try running this first and then trying again:
   ```
   sed -i 's/\r$//' setup_env.sh
   sed -i 's/\r$//' submit.sh
   sed -i 's/\r$//' job.sh
   ```
6. **Submit the Job**
   The submit script automatically handles directory setup, secrets injection, and SLURM submission.
   ```
   chmod +x hpc/submit.sh
   ./hpc/submit.sh
   ```

### Monitoring and Results
* **Check Status:** Run `squeue -u <cwl>` to see your job in the queue.
* **View Logs:** Once running, track progress live: `tail -f logs/train_*.out`
* **Retrieve Results:**
   Once the job status has changed to `COMPLETED`, you can download the trained model and predictions to your local machine.
   The following code will download the necessary files produced.
   ```bash
   # download test set predictions
   scp <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/ealstm_project/data/output/test_set_predictions.csv ./data/output/
   ```
   ```
   # download saved model
   scp -r <cwl>@sockeye.arc.ubc.ca:/scratch/<alloc-code>/ealstm_project/models/ ./models/
   ```