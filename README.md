# ESG-Cascade-Denoise

[Read the v3.0 preprint: Re-evaluating Cardiac Artifact Removal in Spinal Electrophysiology]

This repository contains the complete analysis code and results for the manuscript:
**"The Inescapable Pareto Deadlock: An Exhaustive Evaluation of Additive and Multiplicative Linear Operations for Cardiac Artifact Removal in Spinal Electrophysiology" (v3.1)**

This study serves as the definitive exhaustive evaluation of the linear space in ESG denoising. By exploring all 60 additive (subtraction) cascades and 150 hybrid (multiplicative/divisive) cascades under Oracle calibration, we demonstrate that all linear arithmetic operations are fundamentally trapped in an unbreakable Pareto deadlock due to the spatio-temporal collinearity of ECG and SEPs. 

Key findings include:
1. Pure additive operations max out at artifact suppression (`CC_contaminated` = 0.326) but catastrophically destroy morphology (`SSIM_clean` = 0.18).
2. Hybrid operations artificially inflate the Pearson correlation (`CC_contaminated` = 0.42) by exploiting localized phase alignments, but further destroy signal morphology (`SSIM_clean` plummets to 0.13) due to the mathematical singularity of the division operator.

## 1. Requirements
Ensure you have Python 3.10+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

## 2. Data Availability
The raw data used in this study is publicly available on OpenNeuro (Dataset ds004388). Please download the required data and place them in the `data/` directory.

For this v3.1 exhaustive evaluation, we primarily focus on the **sub-001-median** session to perform the 210-cascade permutation search due to computational constraints. 

Expected directory structure:
```text
data/
└── sub-001-median/
    └── sub-001_clean.fif
```
*(Note: You can easily extend the scripts to other subjects by modifying the `fif_path` variable).*

## 3. Usage

### Part A: v3.1 Exhaustive Additive Cascades (60 Combinations)
Run the Oracle calibration script to search all 60 subtraction cascades and evaluate dual-objective metrics (`CC_contaminated` and `SSIM_clean`).

```bash
python scripts_v3.1/7_all-60_cascade_oracle_math_new.py
```

### Part B: v3.1 Hybrid Operations (150 Combinations)
Run the division and multiplication tests. These scripts expand the operator pool to include non-additive transformations, proving the "fidelity illusion".

```bash
python scripts_v3.1/8_division_test.py
python scripts_v3.1/9_multiplicative_test.py
python scripts_v3.1/10_hybrid_laws_cascade.py
```

### Part C: Final Pareto Frontier Plotting
Generate the ultimate Pareto frontier scatter plot and the waveform collapse comparison plots.

```bash
python scripts_v3.1/11_plot_final_deadlock.py
```

## 4. Key Results and Outputs
The output CSV files and figures are saved in the `results_v3.1/` directory.

### Main Text Figures:
- **`v3.1_Final_Deadlock_Frontier.png`**: Figure 1 in the manuscript. The inescapable Pareto frontier across all 210 linear cascades, showing the "morphological wall" (SSIM = 0.5) that no linear method can breach.
- **`v3.1_Mul_Waveform_Check_MultiGrid.png`**: Figure 2. Waveform-level evidence showing how multiplicative operators fail to remove artifacts or distort the envelope.
- **`v3.1_Waveform_Check_MultiGrid.png`**: Figure 3. Waveform-level evidence showing the abnormal localized amplification (waveform explosion) caused by the division operator's singularity.

### Raw Data Tables:
- **`v3.1_Oracle_Exhaustive_60.csv`**: The complete metric data for the 60 pure subtraction cascades.
- **`v3.1_Hybrid_Laws_Exhaustive.csv`**: The complete metric data for the 150 hybrid (subtraction + division + multiplication) cascades.

---
### Legacy (v3.0 Baseline)
The `legacy/` folder contains the v3.0 baseline code and results (228-channel multi-metric evaluation), which demonstrated the initial Pareto deadlock among four representative linear methods. 
