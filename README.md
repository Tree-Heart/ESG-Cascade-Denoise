# ESG-Cascade-Denoise

[Read the preprint: A Cascade Spatial-Temporal Denoising Strategy](https://doi.org/10.5281/zenodo.23852433)

This repository contains the analysis code and results for the preprint:
"A Cascade Spatial-Temporal Denoising Strategy for Cardiac Artifact Removal in Spinal Cord Electrophysiology: A Pilot Study"

## 1. Requirements
Ensure you have Python 3.10+ installed. Install the required packages:

pip install -r requirements.txt

## 2. Data Availability
The raw data used in this study is publicly available on OpenNeuro (Dataset ds004388). 
Please download the required data and place them in the data/ directory.

For the Tibial task (run-02):
Download data for sub-001, sub-012, and sub-018.
The expected directory structure is:
data/sub-001/sub-001_task-tibial_run-02_eeg.set
data/sub-012/sub-012_task-tibial_run-02_eeg.set
data/sub-018/sub-018_task-tibial_run-02_eeg.set

For the Median task (run-03):
Download data for sub-001, sub-009, and sub-029.
The expected directory structure is:
data/sub-001-median/sub-001_task-median_run-03_eeg.set
data/sub-009-median/sub-009_task-median_run-03_eeg.set
data/sub-029-median/sub-029_task-median_run-03_eeg.set

## 3. Usage

Step 1: Preprocessing
Run the preprocessing scripts to downsample and filter the raw data:
python scripts/1_preprocess.py
python scripts/1_preprocess_median.py

Step 2: Denoising & RI Calculation
Run the batch denoising scripts to calculate the Residual Index (RI):
python scripts/2_batch_denoise.py
python scripts/2_batch_denoise_median.py

Step 3: Order-Reversal Ablation Control
Run the ablation script to verify the sequence robustness (single-subject median control):
python scripts/3_ablation_reverse.py

Step 4: Statistical Analysis & Plotting
Generate Figure 1, Figure 2, and Figure 3 (Cross-task validation):
python scripts/3_ultimate_plot.py
python scripts/4_cross_task_plot.py

Step 5: SEP Retention Analysis
Extract all-channel SEP amplitudes and generate Figure 4 and Figure 5:
python scripts/6_all_channel_sep.py
python scripts/7_sep_retention_plot.py

## 4. Results
The output CSV files and the final figures will be saved in the results/ directory. Key outputs include:
- Fig1_Ultimate_Comparison.png (Tibial task RI comparison)
- Fig3_Cross_Task_Validation.png (Cross-task RI comparison)
- Fig4_EEG_Retention.png (EEG SEP retention control)
- Fig5_ESG_Retention.png (ESG SEP retention trade-off)
- all_channel_sep_retention.csv (Detailed SEP retention data)
- sep_retention_stats.csv (Statistical summary of SEP retention)
