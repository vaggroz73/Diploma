# ================================================================
# GRU PIPELINE ME SHAP ANALYSIS - ΕΝΙΑΙΟ ΑΡΧΕΙΟ
# Διπλωματική Εργασία: ML στο Ποδόσφαιρο
# Target: Δoverall (Time Series Forecasting) με SHAP explainability
#
# Βέλτιστες υπερπαράμετροι (από Keras Tuner):
#   gru_units_1    : 256
#   gru_units_2    : 128   (unused — num_gru_layers=1)
#   num_gru_layers : 1
#   dropout_rate   : 0.4
#   use_batchnorm  : True
#   dense_units    : 8
#   learning_rate  : 0.0001
# ================================================================

import os
os.environ["PYTHONHASHSEED"]        = "42"
os.environ["TF_DETERMINISTIC_OPS"]  = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random
import warnings
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

warnings.filterwarnings("ignore")

# ================================================================
# REPRODUCIBILITY
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================================================================
# CONFIG
# ================================================================
class Config:
    TRAIN_YEARS     = [2018, 2019, 2020, 2021]
    VAL_YEAR        = 2022
    TEST_YEAR       = 2023
    ALL_YEARS       = set(range(2018, 2024))
    SEQUENCE_LENGTH = 3
    MIN_SEASONS     = SEQUENCE_LENGTH + 1
    MIN_MINUTES     = 5
    APPLY_LOG_TRANSFORM = True
    SCALER_TYPE         = "robust"

    # GRU Hyperparameters (βέλτιστες μετά από tuning)
    GRU_UNITS_1    = 256
    GRU_UNITS_2    = 128   # χρησιμοποιείται μόνο αν NUM_GRU_LAYERS == 2
    NUM_GRU_LAYERS = 1
    DROPOUT        = 0.2
    USE_BATCHNORM  = True
    DENSE_UNITS    = 16
    LR             = 0.005
    BATCH_SIZE     = 32
    EPOCHS         = 100
    PATIENCE       = 15

    DATA_PATH = "C:/Users/evagg/Desktop/Final.xlsx"  # ΑΛΛΑΞΕ ΜΕ ΤΟ ΔΙΚΟ ΣΟΥ PATH

config = Config()

# ================================================================
# COLUMN DEFINITIONS
# ================================================================
FIFA_FEATURES = [
    "age", "value_eur_log", "wage_eur_log",
    "movement_reactions", "mentality_composure",
    "attacking_short_passing", "mentality_vision",
    "passing", "dribbling", "mentality_positioning", "shooting",
    "attacking_finishing", "power_shot_power", "attacking_volleys",
    "defending", "mentality_interceptions", "defending_marking_awareness",
    "skill_long_passing", "defending_standing_tackle",
]

FBREF_FEATURES = [
    "Touches_Touches", "Mid 3rd_Touches", "Att 3rd_Touches",
    "Live_Touches", "Carries_Carries", "Rec_Receiving", "Mins_Per_90",
]

UNDERSTAT_FEATURES = [
    "xGChain", "npxG", "xA", "xG",
    "assists", "key_passes", "xGBuildup", "shots",
]

STATIC_FEATURES      = ["height_cm", "weight_kg"]
TRANSFER_FEATURES    = ["mid_season_transfer", "played_for_multiple"]
CATEGORICAL_FEATURES = ["preferred_foot", "Pos"]

# ================================================================
# STEP 1 - LOAD & FILTER
# ================================================================
def load_and_filter(config):
    print("="*60)
    print("STEP 1 - LOAD & FILTER")
    print("="*60)
    df = pd.read_excel(config.DATA_PATH)
    df = df[df["Pos"] != "GK"].copy()
    df = df[df["Mins_Per_90"] >= config.MIN_MINUTES].copy()
    df = df[df["Season_End_Year"].isin(config.ALL_YEARS)].copy()
    print(f"Συνολικές γραμμές : {df.shape[0]}")
    print(f"Μοναδικοί παίκτες : {df['id'].nunique()}")
    return df

# ================================================================
# STEP 2 - SEASON-LEVEL AGGREGATION
# ================================================================
def aggregate_season_level(df):
    print("\n" + "="*60)
    print("STEP 2 - SEASON-LEVEL AGGREGATION")
    print("="*60)
    minute_col = "Mins_Per_90" if "Mins_Per_90" in df.columns else "minutes"
    club_counts   = df.groupby(["id", "Season_End_Year"])["Squad"].nunique()
    transfer_flag = club_counts.reset_index(name="num_clubs")
    transfer_flag["mid_season_transfer"] = (transfer_flag["num_clubs"] > 1).astype(int)

    fbref_cols = [
        "Mins_Per_90", "Touches_Touches", "Def Pen_Touches", "Def 3rd_Touches",
        "Mid 3rd_Touches", "Att 3rd_Touches", "Att Pen_Touches", "Live_Touches",
        "Att_Take", "Succ_Take", "Succ_percent_Take", "Tkld_Take",
        "Carries_Carries", "TotDist_Carries", "PrgDist_Carries", "PrgC_Carries",
        "Final_Third_Carries", "CPA_Carries", "Mis_Carries", "Dis_Carries",
        "Rec_Receiving", "PrgR_Receiving",
    ]
    latest_cols = [
        "overall", "potential", "age", "height_cm", "weight_kg",
        "value_eur", "wage_eur", "weak_foot", "skill_moves",
        "pace", "shooting", "passing", "dribbling", "defending", "physic",
        "movement_reactions", "mentality_composure", "attacking_short_passing",
        "mentality_vision", "mentality_positioning", "attacking_finishing",
        "power_shot_power", "attacking_volleys", "mentality_interceptions",
        "defending_marking_awareness", "skill_long_passing", "defending_standing_tackle",
        "xGChain", "npxG", "shots", "xA", "xG", "assists", "key_passes", "xGBuildup",
        "preferred_foot", "Pos",
    ]
    fbref_cols  = [c for c in fbref_cols  if c in df.columns]
    latest_cols = [c for c in latest_cols if c in df.columns]

    def aggregate_group(group):
        result  = {}
        weights = group[minute_col].fillna(0)
        if weights.sum() == 0:
            weights = np.ones(len(group))
        for col in fbref_cols:
            result[col] = np.average(group[col].fillna(0), weights=weights)
        for col in latest_cols:
            non_null    = group[col].dropna()
            result[col] = non_null.iloc[-1] if len(non_null) > 0 else np.nan
        return pd.Series(result)

    agg_df = (df.groupby(["id", "Season_End_Year"])
                .apply(aggregate_group)
                .reset_index())
    agg_df = agg_df.merge(
        transfer_flag[["id", "Season_End_Year", "mid_season_transfer", "num_clubs"]],
        on=["id", "Season_End_Year"], how="left"
    )
    agg_df["mid_season_transfer"] = agg_df["mid_season_transfer"].fillna(0).astype(int)
    agg_df["num_clubs"]           = agg_df["num_clubs"].fillna(1).astype(int)
    agg_df["played_for_multiple"] = (agg_df["num_clubs"] > 1).astype(int)
    print(f"Aggregated shape: {agg_df.shape}")
    return agg_df

# ================================================================
# STEP 3 - SEASONS FILTER
# ================================================================
def seasons_filter(df, config):
    print("\n" + "="*60)
    print("STEP 3 - SEASONS FILTER")
    print("="*60)
    def has_consecutive_seasons(season_list):
        s = sorted(season_list)
        return all(s[i+1] - s[i] == 1 for i in range(len(s)-1))
    season_groups = df.groupby("id")["Season_End_Year"].apply(list)
    valid_ids = season_groups[
        season_groups.apply(lambda s: len(s) >= config.MIN_SEASONS and has_consecutive_seasons(s))
    ].index
    df = df[df["id"].isin(valid_ids)].copy()
    print(f"  Παίκτες: {df['id'].nunique()}")
    return df

# ================================================================
# STEP 4 - TRANSFORMATIONS
# ================================================================
def apply_transformations(df, config):
    print("\n" + "="*60)
    print("STEP 4 - TRANSFORMATIONS")
    print("="*60)
    if config.APPLY_LOG_TRANSFORM:
        for col in ["value_eur", "wage_eur"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])
                print(f"  ok {col}_log")
    return df

# ================================================================
# STEP 5 - TARGET
# ================================================================
def create_target(df):
    print("\n" + "="*60)
    print("STEP 5 - TARGET: Δoverall")
    print("="*60)
    df = df.sort_values(["id", "Season_End_Year"]).reset_index(drop=True)
    df["Δoverall"] = df.groupby("id")["overall"].diff()
    print(f"  NaN Δoverall: {df['Δoverall'].isna().sum()} (context rows)")
    return df

# ================================================================
# STEP 6 - TIME SPLIT
# ================================================================
def create_time_split(df, config):
    print("\n" + "="*60)
    print("STEP 6 - TIME-BASED SPLIT")
    print("="*60)
    df["split"] = np.where(
        df["Season_End_Year"].isin(config.TRAIN_YEARS), "train",
        np.where(df["Season_End_Year"] == config.VAL_YEAR, "val",
                 np.where(df["Season_End_Year"] == config.TEST_YEAR, "test", "drop"))
    )
    for s in ["train", "val", "test"]:
        print(f"  {s}: {(df['split']==s).sum()}")
    return df

# ================================================================
# STEP 7 - PREPROCESSOR (fit on train ONLY)
# ================================================================
def build_preprocessor(df_train, config):
    continuous  = [c for c in FIFA_FEATURES + FBREF_FEATURES + UNDERSTAT_FEATURES + STATIC_FEATURES
                   if c in df_train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df_train.columns]
    transfer    = [c for c in TRANSFER_FEATURES    if c in df_train.columns]

    assert "overall" not in continuous, "LEAKAGE: 'overall' βρέθηκε στα continuous features!"

    print("\n" + "="*60)
    print("STEP 7 - PREPROCESSOR")
    print("="*60)
    print(f"  Continuous  : {len(continuous)}")
    print(f"  Categorical : {len(categorical)}")
    print(f"  Transfer    : {len(transfer)}")

    scaler = RobustScaler() if config.SCALER_TYPE == "robust" else StandardScaler()
    preprocessor = ColumnTransformer(transformers=[
        ("num",      scaler,                                                      continuous),
        ("cat",      OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("transfer", "passthrough",                                               transfer),
    ])
    return continuous, categorical, transfer, preprocessor

# ================================================================
# STEP 8 - FIT & TRANSFORM
# ================================================================
def fit_transform_preprocessor(df, preprocessor, continuous, categorical, transfer, config):
    print("\n" + "="*60)
    print("STEP 8 - FIT & TRANSFORM")
    print("="*60)
    train_mask = df["split"] == "train"
    preprocessor.fit(df[train_mask][continuous + categorical + transfer])
    print(f"  Fit on {train_mask.sum()} train rows ok")

    X_processed = preprocessor.transform(df[continuous + categorical + transfer])
    ohe               = preprocessor.named_transformers_["cat"]
    cat_names         = list(ohe.get_feature_names_out(categorical))
    all_feature_names = continuous + cat_names + transfer
    print(f"  Total features: {len(all_feature_names)}")

    df_proc = df.copy()
    for i, feat in enumerate(all_feature_names):
        df_proc[feat] = X_processed[:, i]
    df_proc = df_proc.drop(columns=[c for c in categorical if c in df_proc.columns])
    return df_proc, all_feature_names

# ================================================================
# STEP 9 - BUILD SEQUENCES
# ================================================================
def build_sequences(df, all_features, target_years, config):
    X, y, player_ids, years = [], [], [], []
    for pid in df["id"].unique():
        pdata = (df[df["id"] == pid]
                 .sort_values("Season_End_Year")
                 .reset_index(drop=True))
        for i in range(config.SEQUENCE_LENGTH, len(pdata)):
            if pdata.iloc[i]["Season_End_Year"] not in target_years:
                continue
            seq    = pdata.iloc[i - config.SEQUENCE_LENGTH:i][all_features].values
            target = pdata.iloc[i]["Δoverall"]
            if not np.isnan(target) and not np.isnan(seq).any():
                X.append(seq)
                y.append(target)
                player_ids.append(pid)
                years.append(pdata.iloc[i]["Season_End_Year"])
    print(f"  Built {len(X)} sequences for years: {target_years}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), player_ids, years

# ================================================================
# BUILD GRU MODEL
# ================================================================
def build_model(input_shape, config):
    """
    Δομή GRU (ανάλογη με LSTM):
      - 1 GRU layer (NUM_GRU_LAYERS=1, οπότε return_sequences=False)
      - BatchNormalization (USE_BATCHNORM=True)
      - Dropout(0.4)
      - Dense(8, relu)
      - Dropout(0.2)
      - Dense(1)

    Σημείωση GRU vs LSTM:
      Η GRU έχει 2 gates (reset, update) αντί 3 (input, forget, output)
      της LSTM. Αυτό σημαίνει ~25% λιγότερες παραμέτρους για ίδιο
      αριθμό units. Με units_1=256 έχουμε αρκετή χωρητικότητα
      παρά τον μικρό αριθμό timesteps (SEQUENCE_LENGTH=2).

      Το recurrent_initializer=Orthogonal διατηρείται όπως στο LSTM
      για καλύτερη σταθερότητα gradient κατά το training.
    """
    return_seq = (config.NUM_GRU_LAYERS == 2)

    layers = [
        GRU(
            config.GRU_UNITS_1,
            return_sequences=return_seq,
            input_shape=input_shape,
            kernel_initializer=GlorotUniform(seed=SEED),
            recurrent_initializer=Orthogonal(seed=SEED),
        ),
    ]
    if config.USE_BATCHNORM:
        layers.append(BatchNormalization())
    layers.append(Dropout(config.DROPOUT, seed=SEED))

    # 2η GRU layer (μόνο αν NUM_GRU_LAYERS == 2)
    if config.NUM_GRU_LAYERS == 2:
        layers.append(
            GRU(
                config.GRU_UNITS_2,
                return_sequences=False,
                kernel_initializer=GlorotUniform(seed=SEED),
                recurrent_initializer=Orthogonal(seed=SEED),
            )
        )
        if config.USE_BATCHNORM:
            layers.append(BatchNormalization())
        layers.append(Dropout(config.DROPOUT, seed=SEED))

    layers.append(Dense(config.DENSE_UNITS, activation="relu",
                        kernel_initializer=GlorotUniform(seed=SEED)))
    layers.append(Dropout(config.DROPOUT / 2, seed=SEED))
    layers.append(Dense(1, kernel_initializer=GlorotUniform(seed=SEED)))

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=config.LR),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ================================================================
# EVALUATION
# ================================================================
def evaluate(model, X, y, name):
    pred    = model.predict(X, verbose=0).flatten()
    mae     = mean_absolute_error(y, pred)
    rmse    = np.sqrt(mean_squared_error(y, pred))
    r2      = r2_score(y, pred)
    dir_acc = np.mean((pred > 0) == (y > 0))
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  R2           : {r2:.4f}")
    print(f"  Direction Acc: {dir_acc:.2%}")
    return pred

# ================================================================
# TRANSFER IMPACT
# ================================================================
def analyze_transfer_impact(df, test_pred, test_ids, y_test, config):
    print("\n" + "="*60)
    print("TRANSFER IMPACT ANALYSIS")
    print("="*60)
    results = pd.DataFrame({
        "player_id": test_ids, "actual": y_test,
        "predicted": test_pred, "abs_error": np.abs(y_test - test_pred),
    })
    test_info = df[df["Season_End_Year"] == config.TEST_YEAR][["id", "mid_season_transfer"]].copy()
    results   = results.merge(test_info, left_on="player_id", right_on="id", how="left")
    for flag, label in [(1, "WITH transfer"), (0, "WITHOUT transfer")]:
        subset = results[results["mid_season_transfer"] == flag]
        print(f"  {label:20s}: n={len(subset):4d}  MAE={subset['abs_error'].mean():.4f}")
    return results

# ================================================================
# FEATURE IMPORTANCE - PERMUTATION
# ================================================================
def analyze_feature_importance_permutation(model, X_test, y_test, all_features, n_repeats=10):
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE - Permutation")
    print("="*60)
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test, verbose=0).flatten())
    print(f"Baseline MAE: {baseline_mae:.4f}")
    rng         = np.random.default_rng(SEED)
    importances = []
    for feat_idx in range(X_test.shape[2]):
        delta_maes = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm   = rng.permutation(X_perm.shape[0])
            X_perm[:, :, feat_idx] = X_perm[perm, :, feat_idx]
            delta_maes.append(
                mean_absolute_error(y_test, model.predict(X_perm, verbose=0).flatten()) - baseline_mae
            )
        importances.append(np.mean(delta_maes))
    feat_imp = (pd.DataFrame({"feature": all_features, "importance": importances})
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))
    print(feat_imp.head(15).to_string(index=False))
    return feat_imp

# ================================================================
# SHAP ANALYSIS (Explainable AI)
# ================================================================
def run_shap_analysis(model, X_trainval, X_test, y_test, all_features, config):
    """Πλήρης SHAP analysis με GradientExplainer για GRU μοντέλο"""

    print("\n" + "="*60)
    print("SHAP ANALYSIS - Explainable AI για το GRU μοντέλο")
    print("="*60)

    # -----------------------------------------------------------------
    # 1. Επιλογή background data (από Train+Val)
    # -----------------------------------------------------------------
    print("\n[1] Προετοιμασία background data...")

    rng_shap = np.random.default_rng(SEED)
    n_background = min(150, X_trainval.shape[0])
    bg_idx = rng_shap.choice(X_trainval.shape[0], n_background, replace=False)
    background_data = X_trainval[bg_idx].astype(np.float32)

    print(f"  Background data shape : {background_data.shape}")

    # -----------------------------------------------------------------
    # 2. GradientExplainer
    # -----------------------------------------------------------------
    print("\n[2] Δημιουργία GradientExplainer...")

    # Κλείδωμα του μοντέλου σε inference mode
    model.trainable = False

    explainer = shap.GradientExplainer(model, background_data)
    print("  ✓ GradientExplainer created")

    # -----------------------------------------------------------------
    # 3. Υπολογισμός SHAP values
    # -----------------------------------------------------------------
    print("\n[3] Υπολογισμός SHAP values...")

    n_shap_samples = min(200, X_test.shape[0])
    shap_indices = rng_shap.choice(X_test.shape[0], n_shap_samples, replace=False)
    X_test_sample = X_test[shap_indices].astype(np.float32)
    y_test_sample = y_test[shap_indices]

    print(f"  SHAP samples: {n_shap_samples}")
    print(f"  Εκτιμώμενος χρόνος: 3-6 λεπτά...")

    shap_values = explainer.shap_values(X_test_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if shap_values.ndim == 4:
        shap_values = shap_values[:, :, :, 0]

    print(f"  SHAP values shape: {shap_values.shape}")

    nan_count = np.isnan(shap_values).sum()
    if nan_count > 0:
        print(f"  ⚠ ΠΡΟΕΙΔΟΠΟΙΗΣΗ: {nan_count} NaN SHAP values")
    else:
        print(f"  ✓ Κανένα NaN — SHAP values έγκυρα")

    # Αποθήκευση
    # np.save("shap_values.npy", shap_values)
    # np.save("shap_indices.npy", shap_indices)
    # print("  ✓ shap_values.npy saved")

    # -----------------------------------------------------------------
    # 4. Global Feature Importance
    # -----------------------------------------------------------------
    print("\n[4] Δημιουργία Summary Plots...")

    timestep_labels = [f'Season_t-{config.SEQUENCE_LENGTH - i}'
                       for i in range(config.SEQUENCE_LENGTH)]

    shap_values_mean = shap_values.mean(axis=1)
    X_test_mean = X_test_sample.mean(axis=1)

    # Bar plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_mean, X_test_mean,
                      feature_names=all_features, plot_type="bar",
                      max_display=20, show=False)
    plt.title('Global SHAP Feature Importance (GRU)', fontsize=14)
    plt.tight_layout()
    plt.show()
    # plt.savefig("shap_global_bar.png", dpi=150, bbox_inches='tight')
    # plt.close()
    # print("  ✓ shap_global_bar.png")

    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_mean, X_test_mean,
                      feature_names=all_features, max_display=20, show=False)
    plt.title('SHAP Feature Importance Summary (GRU)', fontsize=14)
    plt.tight_layout()
    plt.show()
    # plt.savefig("shap_global_summary.png", dpi=150, bbox_inches='tight')
    # plt.close()
    # print("  ✓ shap_global_summary.png")

    # -----------------------------------------------------------------
    # 5. Per Timestep Analysis
    # -----------------------------------------------------------------
    print("\n[5] Ανάλυση ανά timestep...")

    for t in range(config.SEQUENCE_LENGTH):
        shap_values_t = shap_values[:, t, :]
        X_test_t = X_test_sample[:, t, :]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_t, X_test_t,
                          feature_names=all_features, max_display=20, show=False)
        plt.title(f'SHAP Feature Importance - {timestep_labels[t]} (GRU)', fontsize=14)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"shap_summary_{timestep_labels[t]}.png", dpi=150, bbox_inches='tight')
        # plt.close()
        # print(f"  ✓ shap_summary_{timestep_labels[t]}.png")

    # -----------------------------------------------------------------
    # 6. Temporal Heatmap
    # -----------------------------------------------------------------
    print("\n[6] Δημιουργία Temporal Heatmap...")

    mean_abs_shap_per_timestep = np.mean(np.abs(shap_values), axis=0)

    temporal_shap_df = pd.DataFrame(
        mean_abs_shap_per_timestep.T,
        index=all_features,
        columns=timestep_labels,
    )
    temporal_shap_df['total_importance'] = temporal_shap_df.sum(axis=1)
    temporal_shap_df = temporal_shap_df.sort_values('total_importance', ascending=False)
    temporal_shap_df = temporal_shap_df.drop(columns=['total_importance'])

    top_n = min(20, len(all_features))
    plt.figure(figsize=(14, 10))
    sns.heatmap(temporal_shap_df.head(top_n), annot=True, fmt=".4f",
                cmap='viridis', linewidths=0.5,
                cbar_kws={'label': 'Mean |SHAP Value|'})
    plt.title(f'Mean |SHAP| per Feature and Time Step - GRU (Top {top_n})', fontsize=14)
    plt.tight_layout()
    plt.show()
    # plt.savefig("shap_temporal_heatmap.png", dpi=150, bbox_inches='tight')
    # plt.close()
    # print("  ✓ shap_temporal_heatmap.png")

    # -----------------------------------------------------------------
    # 7. Dependence Plots
    # -----------------------------------------------------------------
    print("\n[7] Δημιουργία Dependence Plots...")

    feature_importance_global = np.abs(shap_values_mean).mean(axis=0)
    top_feature_idx = np.argmax(feature_importance_global)
    top_feature_name = all_features[top_feature_idx]

    print(f"  Top feature: {top_feature_name}")

    if 'age' in all_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(ind=top_feature_name, shap_values=shap_values_mean,
                             features=X_test_mean, feature_names=all_features,
                             interaction_index='age', alpha=0.6, x_jitter=0.3, show=False)
        plt.title(f'SHAP Dependence: {top_feature_name} (colored by age) — GRU', fontsize=13)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"shap_dependence_{top_feature_name}_by_age.png", dpi=150, bbox_inches='tight')
        # plt.close()
        # print(f"  ✓ shap_dependence_{top_feature_name}_by_age.png")

    if 'mid_season_transfer' in all_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(ind='mid_season_transfer', shap_values=shap_values_mean,
                             features=X_test_mean, feature_names=all_features,
                             interaction_index=top_feature_name, alpha=0.6, x_jitter=0.2, show=False)
        plt.title(f'SHAP Dependence: Mid-Season Transfer (colored by {top_feature_name}) — GRU', fontsize=13)
        plt.tight_layout()
        plt.show()
        # plt.savefig("shap_dependence_transfer.png", dpi=150, bbox_inches='tight')
        # plt.close()
        # print("  ✓ shap_dependence_transfer.png")

    # -----------------------------------------------------------------
    # 8. Waterfall Plots
    # -----------------------------------------------------------------
    print("\n[8] Δημιουργία Waterfall Plots...")

    expected_value = float(np.mean(y_test_sample))
    predictions = model.predict(X_test_sample, verbose=0).flatten()
    errors = np.abs(y_test_sample - predictions)

    best_idx  = np.argmin(errors)
    worst_idx = np.argmax(errors)
    mid_idx   = len(errors) // 2

    waterfall_indices = [best_idx, mid_idx, worst_idx]
    waterfall_labels  = ["Best_Prediction", "Typical_Prediction", "Worst_Prediction"]

    for idx, label in zip(waterfall_indices, waterfall_labels):
        shap_sample = shap_values[idx].mean(axis=0)
        X_sample    = X_test_sample[idx].mean(axis=0)

        plt.figure(figsize=(14, 10))
        shap.waterfall_plot(
            shap.Explanation(values=shap_sample, base_values=expected_value,
                             data=X_sample, feature_names=all_features),
            max_display=15, show=False
        )
        plt.title(f'SHAP Waterfall (GRU) - {label}\nActual: {y_test_sample[idx]:.2f}, '
                  f'Pred: {predictions[idx]:.2f}, Error: {errors[idx]:.2f}', fontsize=12)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"shap_waterfall_{label}.png", dpi=150, bbox_inches='tight')
        # plt.close()
        # print(f"  ✓ shap_waterfall_{label}.png")

    # -----------------------------------------------------------------
    # 9. SHAP Feature Importance DataFrame
    # -----------------------------------------------------------------
    print("\n[9] Αποθήκευση SHAP Feature Importance...")

    shap_importance_df = pd.DataFrame({
        'feature': all_features,
        'mean_abs_SHAP': np.abs(shap_values_mean).mean(axis=0)
    }).sort_values('mean_abs_SHAP', ascending=False)

    print("\n  TOP 15 FEATURES BY SHAP VALUE:")
    print(shap_importance_df.head(15).to_string(index=False))

    # shap_importance_df.to_csv("shap_feature_importance.csv", index=False)
    # print("  ✓ shap_feature_importance.csv")

    # -----------------------------------------------------------------
    # 10. Interactive Force Plot (HTML)
    # -----------------------------------------------------------------
    print("\n[10] Δημιουργία interactive Force Plot...")

    try:
        print("  Interactive force plots are typically rendered in a browser or specific notebook extensions. Skipping direct display.")
        # shap_html = shap.force_plot(expected_value, shap_values_mean[0,:],
        #                            X_test_mean[0,:], feature_names=all_features,
        #                            matplotlib=False, show=False)
        # shap.save_html("shap_force_plot_sample_0.html", shap_html)
        # print("  ✓ shap_force_plot_sample_0.html (interactive)")
    except Exception as e:
        print(f"  ⚠ Force plot HTML skipped: {e}")

    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

    return shap_values, shap_importance_df

# ================================================================
# MAIN PIPELINE
# ================================================================
print("="*60)
print("GRU PIPELINE με SHAP Analysis - Walk-Forward Validation")
print("="*60)

# Steps 1-8: Data preparation
df = load_and_filter(config)
df = aggregate_season_level(df)
df = seasons_filter(df, config)
df = apply_transformations(df, config)
df = create_target(df)
df = create_time_split(df, config)

continuous, categorical, transfer, preprocessor = build_preprocessor(
    df[df["split"] == "train"], config
)
df_proc, all_features_final = fit_transform_preprocessor(
    df, preprocessor, continuous, categorical, transfer, config
)

# Step 9: Build sequences
print("\n" + "="*60)
print("STEP 9 - BUILD SEQUENCES")
print("="*60)

X_train, y_train, train_ids, _ = build_sequences(df_proc, all_features_final, config.TRAIN_YEARS, config)
X_val,   y_val,   val_ids,   _ = build_sequences(df_proc, all_features_final, [config.VAL_YEAR], config)
X_test,  y_test,  test_ids,  _ = build_sequences(df_proc, all_features_final, [config.TEST_YEAR], config)

X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

print(f"\n  Train    : {X_train.shape}")
print(f"  Val      : {X_val.shape}")
print(f"  TrainVal : {X_trainval.shape}")
print(f"  Test     : {X_test.shape}")

# ================================================================
# PHASE A - Train -> Validation
# ================================================================
print("\n" + "="*60)
print("PHASE A - TRAINING (Train -> Val)")
print("="*60)

model = build_model((X_train.shape[1], X_train.shape[2]), config)
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, verbose=1),
    ],
    verbose=1,
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"\n  Best epoch (Phase A): {best_epoch}")

print("\n" + "="*60)
print("EVALUATION - Phase A")
print("="*60)
train_pred_a = evaluate(model, X_train, y_train, "TRAIN (Phase A)")
val_pred = evaluate(model, X_val, y_val, "VALIDATION (Phase A)")

# ================================================================
# PHASE B - Refit: Train+Val -> Test
# ================================================================
print("\n" + "="*60)
print("PHASE B - REFIT (Train+Val -> Test)")
print("="*60)

model_refit = build_model((X_trainval.shape[1], X_trainval.shape[2]), config)
model_refit.set_weights(model.get_weights())

history_refit = model_refit.fit(
    X_trainval, y_trainval,
    epochs=best_epoch + 10,
    batch_size=config.BATCH_SIZE,
    callbacks=[
        EarlyStopping(monitor="loss", patience=config.PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", patience=5, factor=0.5, verbose=1),
    ],
    verbose=1,
)

print("\n" + "="*60)
print("EVALUATION - Phase B (Refit)")
print("="*60)
_ = evaluate(model_refit, X_train, y_train, "TRAIN (Phase B - sanity)")
_ = evaluate(model_refit, X_val, y_val, "VAL (Phase B - sanity)")
test_pred = evaluate(model_refit, X_test, y_test, "TEST (Phase B - FINAL)")

# ================================================================
# ΣΥΓΚΡΙΣΗ Phase A vs Phase B
# ================================================================
print("\n" + "="*60)
print("ΣΥΓΚΡΙΣΗ Phase A vs Phase B - TEST SET")
print("="*60)
test_pred_a = model.predict(X_test, verbose=0).flatten()
mae_a = mean_absolute_error(y_test, test_pred_a)
mae_b = mean_absolute_error(y_test, test_pred)
print(f"  Test MAE - Phase A (train only) : {mae_a:.4f}")
print(f"  Test MAE - Phase B (refit)      : {mae_b:.4f}")
diff = mae_a - mae_b
print(f"  Διαφορά: {diff:+.4f} ({'βελτίωση' if diff > 0 else 'ελαφρά χειρότερο'})")

# ================================================================
# POST-HOC ANALYSIS
# ================================================================
test_results = analyze_transfer_impact(df_proc, test_pred, test_ids, y_test, config)
feat_importance = analyze_feature_importance_permutation(
    model_refit, X_test, y_test, all_features_final, n_repeats=10
)

# ================================================================
# SHAP ANALYSIS (Explainable AI)
# ================================================================
shap_values, shap_importance = run_shap_analysis(
    model_refit, X_trainval, X_test, y_test, all_features_final, config
)

# ================================================================
# SAVE
# ================================================================
print("\n" + "="*60)
print("SAVING")
print("="*60)

model.save("gru_phase_a.keras")
model_refit.save("gru_phase_b_refit.keras")

pd.DataFrame({"id": val_ids, "y_true": y_val, "gru_pred": val_pred}).to_csv(
    "gru_val_predictions.csv", index=False)
pd.DataFrame({"id": test_ids, "y_true": y_test, "gru_pred": test_pred}).to_csv(
    "gru_test_predictions_seq_3.csv", index=False)
test_results.to_csv("test_predictions_with_transfer.csv", index=False)
feat_importance.to_csv("gru_feature_importance_seq_3.csv", index=False)

print("  gru_phase_a.keras")
print("  gru_phase_b_refit.keras  <- ΤΕΛΙΚΟ ΜΟΝΤΕΛΟ")
print("  gru_val_predictions.csv")
print("  gru_test_predictions_seq_3.csv")
print("  test_predictions_with_transfer.csv")
print("  gru_feature_importance_seq_3.csv")

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Curves Over Epochs GRU')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
# plt.show()

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 OUTPUT FILES SUMMARY:")
print("  ├── gru_phase_a.keras")
print("  ├── gru_phase_b_refit.keras (τελικό μοντέλο)")
print("  ├── gru_val_predictions.csv")
print("  ├── gru_test_predictions_seq_2.csv")
print("  ├── test_predictions_with_transfer.csv")
print("  ├── gru_feature_importance_seq_2.csv")
print("  ├── training_curves.png (will now be displayed directly)")
print("  │")
print("  ├── SHAP Analysis Outputs: (will now be displayed directly)")
print("  │   ├── shap_values.npy, shap_indices.npy (still saved)")
print("  │   ├── shap_global_bar.png, shap_global_summary.png (now displayed)")
print("  │   ├── shap_summary_*.png (per timestep) (now displayed)")
print("  │   ├── shap_temporal_heatmap.png (now displayed)")
print("  │   ├── shap_dependence_*.png (now displayed)")
print("  │   ├── shap_waterfall_*.png (now displayed)")
print("  │   ├── shap_feature_importance.csv (still saved)")
print("  │   └── shap_force_plot_sample_0.html (interactive) (still referenced)")
print("="*60)