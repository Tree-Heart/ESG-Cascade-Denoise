# ESG-Cascade-Denoise

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23851089.svg)](https://doi.org/10.5281/zenodo.23851089)

This repository contains the analysis code and results for the preprint:
"A Cascade Spatial-Temporal Denoising Strategy for Cardiac Artifact Removal in Spinal Cord Electrophysiology"

## 1. Requirements
Ensure you have Python 3.10+ installed. Install the required packages:
pip install -r requirements.txt

## 2. Data Availability
The raw data used in this study is publicly available on OpenNeuro (Dataset ds004388). 
Please download the task-tibial_run-02 data for sub-001, sub-012, and sub-018 and place them in the data/ directory.

The expected directory structure is:
data/sub-001/sub-001_task-tibial_run-02_eeg.set
data/sub-012/sub-012_task-tibial_run-02_eeg.set
data/sub-018/sub-018_task-tibial_run-02_eeg.set

## 3. Usage
Run the complete pipeline (Preprocessing -> Denoising -> Statistics & Plotting):
python scripts/1_preprocess.py
python scripts/2_batch_denoise.py
python scripts/3_ultimate_plot.py

## 4. Results
The output CSV files and the final comparison boxplot (Fig3_Ultimate_Comparison.png) will be saved in the results/ directory.
