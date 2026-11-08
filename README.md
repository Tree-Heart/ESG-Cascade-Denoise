# ESG-Cascade-Denoise

[Read the v3.0 preprint: Re-evaluating Cardiac Artifact Removal in Spinal Electrophysiology]

This repository contains the complete analysis code and results for the manuscript:
**"Re-evaluating Cardiac Artifact Removal in Spinal Electrophysiology: A Clean Ground-Truth Evaluation Framework Reveals the Fundamental Deadlock of Linear Denoising"**

This study systematically evaluates four linear denoising strategies (SSP, ATS, Cascade, and Slew-Rate) across 228 ESG channels, demonstrating a fundamental Pareto deadlock between artifact suppression (RI) and neural morphology preservation (CC_GT, SSIM) due to the spatio-temporal collinearity of ECG and SEPs.

## 1. Requirements
Ensure you have Python 3.10+ installed. Install the required packages:

pip install -r requirements.txt

## 2. Data Availability
The raw data used in this study is publicly available on OpenNeuro (Dataset ds004388). Please download the required data and place them in the data/ directory.

For the Tibial task (run-02): Download data for sub-001, sub-012, and sub-018.
For the Median task (run-03): Download data for sub-001, sub-009, and sub-029.

Expected directory structure:

data/
├── sub-001/
│   └── sub-001_task-tibial_run-02_eeg.set
├── sub-012/
│   └── sub-012_task-tibial_run-02_eeg.set
├── sub-018/
│   └── sub-018_task-tibial_run-02_eeg.set
├── sub-001-median/
│   └── sub-001_task-median_run-03_eeg.set
├── sub-009-median/
│   └── sub-009_task-median_run-03_eeg.set
└── sub-029-median/
    └── sub-029_task-median_run-03_eeg.set

## 3. Usage

### Part A: Preprocessing & Baseline (v2.1)
Run the preprocessing and basic RI calculation scripts:

python scripts/v2.1_RI_analysis/1_preprocess.py
python scripts/v2.1_RI_analysis/1_preprocess_median.py
python scripts/v2.1_RI_analysis/2_batch_denoise.py
python scripts/v2.1_RI_analysis/2_batch_denoise_median.py
python scripts/v2.1_RI_analysis/3_ablation_reverse.py

### Part B: v3.0 Multi-Metric Evaluation & Pareto Frontier
Run the new scripts to reproduce the core v3.0 results (Pseudo-GT construction, 7-metric evaluation, and the Pareto deadlock):

python scripts/v3.0_multimetric/all_subjects_multi_metric_eval.py
python scripts/v3.0_multimetric/plot_metrics_boxplot.py
python scripts/v3.0_multimetric/pareto_frontier.py
python scripts/v3.0_multimetric/grid_0_to_1.py

Note: Ensure BASE_DIR in the v3.0 scripts points to your local data path.

## 4. Key Results and Outputs
The output CSV files and figures will be saved in the results/ directory.

Core v3.0 Outputs (n=228 ESG channels):
- v3.0_All_228_ESG_Channels_Metrics.csv: The master data table containing all 7 metrics for all methods across 228 channels.
- v3.0_All_228_Channels_Boxplot_Final.png: Figure 1 (Main text). Boxplots showing the distributions of RI, CC_GT, CC_WB, RE, SD, KR2, and SSIM.
- Pareto_Frontier_Final.png: The Pareto curve showing the Gated Cascade strategy cannot break the SSP and ATS bounds.
- Tradeoff_Line_0_to_1.png: The dual-Y axis plot demonstrating the continuous Pareto deadlock (SEP retention vs. RI).
- Grid_0_to_1_Detailed.csv: The raw data driving the continuous trade-off analysis.

Supplementary / v2.1 Outputs:
- all_channel_sep_retention.csv: Detailed SEP amplitude retention data per channel.
- sep_retention_stats.csv: Statistical summary of SEP retention (EEG vs. ESG).
- Fig3_Cross_Task_Validation.png: Cross-task RI comparison (Tibial vs. Median).
