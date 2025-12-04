# electron_collision_prediction
# Dielectron Invariant Mass M Prediction (CERN)

This project implements a **regression pipeline** to predict the **invariant mass `M`** of electron pairs in high-energy collisions.

## Dataset

- Contains ~100k dielectron collision events. Taken from Kaggle: Cern Electron Collision Data. 
- Main columns:
  - `E1`, `E2`: Total energy of each electron (GeV)
  - `px1, py1, pz1, px2, py2, pz2`: Momentum components (GeV)
  - `pt1, pt2`: Transverse momentum (GeV)
  - `eta1, eta2`: Pseudorapidity
  - `phi1, phi2`: Phi angle (rad)
  - `Q1, Q2`: Electron charge
  - `M`: Invariant mass (GeV) — **target**

## Preprocessing

1. Remove rows with `NaN` in `M`.  
2. Add derived features:
   - `E_sum`, `pt_sum`, `pt_ratio`, `eta_diff`, `phi_diff`, `deltaR`  
   - Differences of momentum components (`dpx`, `dpy`, `dpz`)
3. Scale features using `StandardScaler`.  
4. Train/test split (80% / 20%).  

## Model

The following regression model was trained and evaluated:

| Model             | Metrics (Test set) |
|-----------------|------------------|
| Linear Regression | MAE = 4.9797, RMSE= 6.9269, R² = 0.9250   |
| RandomForest | MAE =  0.6350, RMSE = 1.4304 , R²= 0.9968  |
| MLP | MAE =  , RMSE =  , R² =  |

> Evaluation metrics and the plot `pred_vs_true.png` illustrate predicted vs. true M values.

## Results

- Plot comparing predictions and ground truth: `pred_vs_true.png`.  


