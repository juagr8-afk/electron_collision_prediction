import os
import zipfile
import glob
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


OUT_PROCESSED = "processed_data.csv"
EXTRACT_DIR = "extracted_data"
CHUNKSIZE = 100_000
TEST_SIZE = 0.2
RND = 42
N_JOBS = -1
TRAIN_ON_SAMPLE = False   
SAMPLE_N = 50_000
BATCH_SIZE = 256
EPOCHS = 30


# find column alias
def find_alias(colset, base):
    if base in colset:
        return base
    base_norm = re.sub(r'[_\-\s]', '', base.lower())
    for c in colset:
        if re.sub(r'[_\-\s]', '', c.lower()) == base_norm:
            return c
    pat = re.compile(re.sub(r'(\d+)', r'.*\\1', base), re.IGNORECASE)
    for c in colset:
        if pat.search(c):
            return c
    return None

# reconstructing px,py,pz from pt,phi,eta if px{idx},py{idx},pz{idx} are missing
def try_reconstruct(chunk, idx):
    cols = list(chunk.columns)
    pt_col = find_alias(cols, f"pt{idx}")
    phi_col = find_alias(cols, f"phi{idx}")
    eta_col = find_alias(cols, f"eta{idx}")
    created = []
    if pt_col and phi_col and eta_col:
        px_c = f"px{idx}"
        py_c = f"py{idx}"
        pz_c = f"pz{idx}"
        if px_c not in chunk.columns:
            chunk[px_c] = chunk[pt_col] * np.cos(chunk[phi_col])
            created.append(px_c)
        if py_c not in chunk.columns:
            chunk[py_c] = chunk[pt_col] * np.sin(chunk[phi_col])
            created.append(py_c)
        if pz_c not in chunk.columns:
            chunk[pz_c] = chunk[pt_col] * np.sinh(chunk[eta_col])
            created.append(pz_c)
    return chunk, created

# extracting csvs
def robust_process_zip(extract_dir=EXTRACT_DIR, out_csv=OUT_PROCESSED, chunksize=CHUNKSIZE):
    csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)

    if os.path.exists(out_csv):
        os.remove(out_csv)

    processed_any = False

    for csv_path in csv_files:
        reader = pd.read_csv(csv_path, chunksize=chunksize)
        for chunk in tqdm(reader, desc=f"chunks {os.path.basename(csv_path)}", leave=False):
            cols = list(chunk.columns)
            m_alias = find_alias(cols, "M")
            if m_alias is None:
                print(" No 'M' in this chunk, skip")
                continue

            # try reconstructing px/py/pz if missing
            for idx in [1,2]:
                chunk, created = try_reconstruct(chunk, idx)
                

            cols = list(chunk.columns)
            has_E = (find_alias(cols, "E1") is not None) and (find_alias(cols, "E2") is not None)
            has_px1 = all(find_alias(cols, f"{c}1") is not None for c in ["px","py","pz"])
            has_px2 = all(find_alias(cols, f"{c}2") is not None for c in ["px","py","pz"])

            def getcol(df, name, default=None):
                alias = find_alias(df.columns, name)
                if alias is not None:
                    return df[alias]
                else:
                    if default is None:
                        return pd.Series([np.nan]*len(df))
                    return pd.Series([default]*len(df))

            # backup dataframe
            dfc = pd.DataFrame()
            dfc["E1"] = getcol(chunk, "E1", np.nan)
            dfc["E2"] = getcol(chunk, "E2", np.nan)
            dfc["pt1"] = getcol(chunk, "pt1", np.nan)
            dfc["pt2"] = getcol(chunk, "pt2", np.nan)
            dfc["phi1"] = getcol(chunk, "phi1", np.nan)
            dfc["phi2"] = getcol(chunk, "phi2", np.nan)
            dfc["eta1"] = getcol(chunk, "eta1", np.nan)
            dfc["eta2"] = getcol(chunk, "eta2", np.nan)

            # px/py/pz backup
            for idx in [1,2]:
                for comp in ["px","py","pz"]:
                    cname = f"{comp}{idx}"
                    alias = find_alias(chunk.columns, cname)
                    if alias is not None:
                        dfc[cname] = chunk[alias]
                    else:
                        if comp == "px":
                            pt_alias = find_alias(chunk.columns, f"pt{idx}")
                            phi_alias = find_alias(chunk.columns, f"phi{idx}")
                            if pt_alias and phi_alias:
                                dfc[cname] = chunk[pt_alias] * np.cos(chunk[phi_alias])
                            else:
                                dfc[cname] = np.nan
                        elif comp == "py":
                            pt_alias = find_alias(chunk.columns, f"pt{idx}")
                            phi_alias = find_alias(chunk.columns, f"phi{idx}")
                            if pt_alias and phi_alias:
                                dfc[cname] = chunk[pt_alias] * np.sin(chunk[phi_alias])
                            else:
                                dfc[cname] = np.nan
                        else:  # pz
                            pt_alias = find_alias(chunk.columns, f"pt{idx}")
                            eta_alias = find_alias(chunk.columns, f"eta{idx}")
                            if pt_alias and eta_alias:
                                dfc[cname] = chunk[pt_alias] * np.sinh(chunk[eta_alias])
                            else:
                                dfc[cname] = np.nan

            # target
            dfc["M"] = chunk[m_alias]

            # derived features
            dfc["E_sum"] = dfc["E1"] + dfc["E2"]
            dfc["pt_sum"] = dfc["pt1"] + dfc["pt2"]
            dfc["pt_ratio"] = (dfc["pt1"] + 1e-9) / (dfc["pt2"] + 1e-9)
            dfc["eta_diff"] = dfc["eta1"] - dfc["eta2"]
            phi_diff = dfc["phi1"] - dfc["phi2"]
            phi_diff = (phi_diff + np.pi) % (2*np.pi) - np.pi
            dfc["phi_diff"] = phi_diff
            dfc["deltaR"] = np.sqrt(dfc["eta_diff"]**2 + dfc["phi_diff"]**2)

            dfc["px_sum"] = dfc["px1"] + dfc["px2"]
            dfc["py_sum"] = dfc["py1"] + dfc["py2"]
            dfc["pz_sum"] = dfc["pz1"] + dfc["pz2"]
            dfc["p_sum"] = np.sqrt(dfc["px_sum"]**2 + dfc["py_sum"]**2 + dfc["pz_sum"]**2)
            dfc["pt_system"] = np.sqrt(dfc["px_sum"]**2 + dfc["py_sum"]**2)

            q1 = getcol(chunk, "Q1", 0.0)
            q2 = getcol(chunk, "Q2", 0.0)
            dfc["Q_tot"] = pd.to_numeric(q1, errors="coerce") + pd.to_numeric(q2, errors="coerce")

            p1_dot_p2 = dfc["px1"]*dfc["px2"] + dfc["py1"]*dfc["py2"] + dfc["pz1"]*dfc["pz2"]
            p1_norm = np.sqrt(dfc["px1"]**2 + dfc["py1"]**2 + dfc["pz1"]**2)
            p2_norm = np.sqrt(dfc["px2"]**2 + dfc["py2"]**2 + dfc["pz2"]**2)
            denom = p1_norm * p2_norm
            cos_theta = np.where(denom == 0, 0.0, p1_dot_p2 / denom)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            dfc["opening_angle"] = np.arccos(cos_theta)

            with np.errstate(divide='ignore', invalid='ignore'):
                dfc["eta_system"] = 0.5 * np.log(np.where(dfc["p_sum"] - dfc["pz_sum"] == 0, 1.0, (dfc["p_sum"] + dfc["pz_sum"]) / (dfc["p_sum"] - dfc["pz_sum"])))
            dfc["eta_system"] = dfc["eta_system"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # replace +/-inf with NaN
            dfc = dfc.replace([np.inf, -np.inf], np.nan)

            # delete rows without target M
            dfc = dfc[~dfc["M"].isna()]
            if dfc.shape[0] == 0:
                continue

            # final columns
            final_cols = ["E1","E2","px1","py1","pz1","px2","py2","pz2","pt1","pt2","eta1","eta2","phi1","phi2",
                          "E_sum","pt_sum","pt_ratio","eta_diff","phi_diff","deltaR","px_sum","py_sum","pz_sum",
                          "p_sum","pt_system","eta_system","opening_angle","Q_tot","M"]
            cols_exist = [c for c in final_cols if c in dfc.columns]
            out_chunk = dfc[cols_exist]
            header = not os.path.exists(out_csv)
            out_chunk.to_csv(out_csv, mode="a", header=header, index=False)
            processed_any = True


# functions for models 
def evaluate_regression(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"mae":mae, "rmse":rmse, "r2":r2}

def plot_pred_vs_true(y_true, y_pred, fname="pred_vs_true.png"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.25, s=6)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel("M true (GeV)")
    plt.ylabel("M pred (GeV)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

# train processed_data.csv 
def train_from_processed(out_csv=OUT_PROCESSED, train_on_sample=TRAIN_ON_SAMPLE, sample_n=SAMPLE_N):

    # samples
    if train_on_sample:
        df_all = pd.read_csv(out_csv)
        df = df_all.sample(n=sample_n, random_state=RND) if len(df_all) > sample_n else df_all
        del df_all
    else:      
        df = pd.read_csv(out_csv)

    print("df raw shape:", df.shape)



    # replace +/-inf with NaN y delete rows where  target M is NaN 
    df = df.replace([np.inf, -np.inf], np.nan)
    if "M" not in df.columns:
        raise KeyError("No 'M' in processed_data.csv")

    # Imputation with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "M"]  
   

    imputer = SimpleImputer(strategy="mean")
    df_num = df[numeric_cols]
    df_num_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=numeric_cols, index=df.index)
    df[numeric_cols] = df_num_imputed

    # prepare X,y
    X = df.drop(columns=["M"]).astype(np.float32)
    y = df["M"].astype(np.float32).values
    feature_names = X.columns.tolist()
    print("Features used:", feature_names)
    print("X shape:", X.shape, " y shape:", y.shape)

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=TEST_SIZE, random_state=RND)
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(imputer, "imputer.joblib")

    results = {}

    # 1) Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    results["linear"] = evaluate_regression(y_test, lr_pred, label="LinearRegression")
    joblib.dump(lr, "linear_regressor.joblib")
    plot_pred_vs_true(y_test, lr_pred, fname="lr_pred_vs_true.png")

    # 2) RandomForest
    print("\nTraining RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=200, random_state=RND, n_jobs=N_JOBS)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    results["rf"] = evaluate_regression(y_test, rf_pred, label="RandomForest")
    joblib.dump(rf, "rf_regressor_no_mcalc.joblib")
    plot_pred_vs_true(y_test, rf_pred, fname="rf_pred_vs_true.png")

    # 3) MLP (Keras) 
    print("\nTraining MLP (Keras)...")
    lp = build_mlp(X_train_s.shape[1])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    mc = callbacks.ModelCheckpoint("mlp_model.h5", save_best_only=True, monitor="val_loss")
    mlp.fit(X_train_s, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, mc], verbose=2)
    mlp = models.load_model("mlp_model.h5")
    mlp_pred = mlp.predict(X_test_s).ravel()
    results["mlp"] = evaluate_regression(y_test, mlp_pred, label="MLP")
    plot_pred_vs_true(y_test, mlp_pred, fname="mlp_pred_vs_true.png")
 
    
    # save comparative predictions
    out_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred_lr": lr_pred,
        "y_pred_rf": rf_pred,
        "y_pred_mlp": mlp_pred
    })
    out_df.to_csv("predictions.csv", index=False)
    joblib.dump(results, "regression_results.joblib")
    

# Loading data
print("Upload zip file")
uploaded = files.upload()
if len(uploaded) == 0:
    raise RuntimeError("No files uploaded")
zip_name = list(uploaded.keys())[0]


# extract zip in file EXTRACT_DIR
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(zip_name, "r") as z:
    z.extractall(EXTRACT_DIR)


robust_process_zip(EXTRACT_DIR, OUT_PROCESSED, CHUNKSIZE)

train_from_processed(OUT_PROCESSED, train_on_sample=TRAIN_ON_SAMPLE, sample_n=SAMPLE_N)
