import os
import random

os.environ["PYTHONHASHSEED"]         = "42"
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Reproducibility — seeds
# ----------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
class Config:
    TRAIN_YEARS = [2018, 2019, 2020, 2021]
    VAL_YEAR    = 2022
    TEST_YEAR   = 2023
    ALL_YEARS   = set(range(2018, 2024))

    SEQUENCE_LENGTH = 2
    MIN_SEASONS     = SEQUENCE_LENGTH + 1   # = 3
    MIN_MINUTES     = 5

    APPLY_LOG_TRANSFORM = True
    SCALER_TYPE         = "robust"

    EPOCHS     = 100
    PATIENCE   = 15
    BATCH_SIZE = 32

    DATA_PATH = "C:/Users/evagg/Desktop/Final.xlsx"

    # Keras Tuner
    TUNER_ALGO           = "bayesian"   # "random" ή "bayesian"
    MAX_TRIALS           = 30
    EXECUTIONS_PER_TRIAL = 1
    TUNER_DIR            = "tuner_results"
    TUNER_PROJECT        = "lstm_delta_overall"

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
# ΣΗΜΑΝΤΙΚΟ: 'overall' ΔΕΝ είναι εδώ — αποφεύγουμε trivial leakage.

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
# STEP 1 — LOAD & FILTER
# ================================================================
def load_and_filter(config):
    print("="*60)
    print("STEP 1 — LOAD & FILTER")
    print("="*60)
    df = pd.read_excel(config.DATA_PATH)
    df = df[df["Pos"] != "GK"].copy()
    df = df[df["Mins_Per_90"] >= config.MIN_MINUTES].copy()
    df = df[df["Season_End_Year"].isin(config.ALL_YEARS)].copy()
    print(f"  Συνολικές γραμμές : {df.shape[0]}")
    print(f"  Μοναδικοί παίκτες : {df['id'].nunique()}")
    return df


# ================================================================
# STEP 2 — SEASON-LEVEL AGGREGATION
# ================================================================
def aggregate_season_level(df):
    print("\n" + "="*60)
    print("STEP 2 — SEASON-LEVEL AGGREGATION")
    print("="*60)

    minute_col = "Mins_Per_90" if "Mins_Per_90" in df.columns else "minutes"

    club_counts   = df.groupby(["id", "Season_End_Year"])["Squad"].nunique()
    transfer_flag = club_counts.reset_index(name="num_clubs")
    transfer_flag["mid_season_transfer"] = (transfer_flag["num_clubs"] > 1).astype(int)
    print(f"  Mid-season transfers: {transfer_flag['mid_season_transfer'].sum()}")

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

    dupes = agg_df.duplicated(subset=["id", "Season_End_Year"]).sum()
    print(f"  Duplicates (should be 0): {dupes}")
    print(f"  Aggregated shape        : {agg_df.shape}")
    return agg_df


# ================================================================
# STEP 3 — SEASONS FILTER
# ================================================================
def seasons_filter(df, config):
    print("\n" + "="*60)
    print("STEP 3 — SEASONS FILTER")
    print("="*60)

    def has_consecutive_seasons(season_list):
        s = sorted(season_list)
        return all(s[i+1] - s[i] == 1 for i in range(len(s)-1))

    season_groups = df.groupby("id")["Season_End_Year"].apply(list)
    valid_ids = season_groups[
        season_groups.apply(
            lambda s: len(s) >= config.MIN_SEASONS and has_consecutive_seasons(s)
        )
    ].index
    df = df[df["id"].isin(valid_ids)].copy()

    counts = df.groupby("id")["Season_End_Year"].count()
    print(f"  Παίκτες που περνούν το φίλτρο: {df['id'].nunique()}")
    for n, c in counts.value_counts().sort_index().items():
        print(f"    {n} σεζόν: {c} παίκτες")
    return df


# ================================================================
# STEP 4 — LOG TRANSFORM
# ================================================================
def apply_transformations(df, config):
    print("\n" + "="*60)
    print("STEP 4 — TRANSFORMATIONS")
    print("="*60)
    if config.APPLY_LOG_TRANSFORM:
        for col in ["value_eur", "wage_eur"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])
                print(f"  ok {col}_log")
    return df


# ================================================================
# STEP 5 — TARGET
# ================================================================
def create_target(df):
    """
    Δoverall = overall(t) - overall(t-1).
    Γραμμές NaN (πρώτη σεζόν κάθε παίκτη) κρατούνται ως context rows.
    """
    print("\n" + "="*60)
    print("STEP 5 — TARGET: Δoverall")
    print("="*60)
    df = df.sort_values(["id", "Season_End_Year"]).reset_index(drop=True)
    df["Δoverall"] = df.groupby("id")["overall"].diff()
    print(f"  NaN Δoverall (context rows): {df['Δoverall'].isna().sum()}")
    return df


# ================================================================
# STEP 6 — TIME SPLIT
# ================================================================
def create_time_split(df, config):
    print("\n" + "="*60)
    print("STEP 6 — TIME-BASED SPLIT")
    print("="*60)
    df["split"] = np.where(
        df["Season_End_Year"].isin(config.TRAIN_YEARS), "train",
        np.where(df["Season_End_Year"] == config.VAL_YEAR, "val",
                 np.where(df["Season_End_Year"] == config.TEST_YEAR, "test", "drop"))
    )
    for s in ["train", "val", "test"]:
        print(f"  {s.capitalize():6s}: {(df['split']==s).sum()} records")
    return df


# ================================================================
# STEP 7 — PREPROCESSOR (fit on TRAIN only — walk-forward policy)
# ================================================================
def build_preprocessor(df_train, config):
    """
    Ο preprocessor γίνεται fit ΜΟΝΟ στο TRAIN set.

    Walk-Forward Policy:
      Ο scaler παραμένει frozen σε ολόκληρο το pipeline.
      ΔΕΝ κάνει refit στο train+val κατά το Phase B.
      Αυτό εξασφαλίζει ότι τα statistics κανονικοποίησης
      δεν επηρεάζονται από val data → αποφυγή leakage.
    """
    continuous  = [c for c in FIFA_FEATURES + FBREF_FEATURES + UNDERSTAT_FEATURES + STATIC_FEATURES
                   if c in df_train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df_train.columns]
    transfer    = [c for c in TRANSFER_FEATURES    if c in df_train.columns]

    assert "overall" not in continuous, \
        "LEAKAGE: 'overall' βρέθηκε στα continuous features!"

    print("\n" + "="*60)
    print("STEP 7 — PREPROCESSOR (fit on TRAIN only)")
    print("="*60)
    print(f"  Continuous  (RobustScaler) : {len(continuous)}")
    print(f"  Categorical (OHE)          : {len(categorical)}")
    print(f"  Transfer    (passthrough)  : {len(transfer)}")

    scaler = RobustScaler() if config.SCALER_TYPE == "robust" else StandardScaler()
    preprocessor = ColumnTransformer(transformers=[
        ("num",      scaler,                                                      continuous),
        ("cat",      OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("transfer", "passthrough",                                               transfer),
    ])
    return continuous, categorical, transfer, preprocessor


# ================================================================
# STEP 8 — FIT & TRANSFORM
# ================================================================
def fit_transform_preprocessor(df, preprocessor, continuous, categorical, transfer, config):
    """
    Fit ΜΟΝΟ στα train rows, transform σε train + val + test.
    Αυτή η συνάρτηση καλείται μία φορά — δεν υπάρχει refit.
    """
    print("\n" + "="*60)
    print("STEP 8 — FIT & TRANSFORM")
    print("="*60)

    train_mask = df["split"] == "train"
    preprocessor.fit(df[train_mask][continuous + categorical + transfer])
    print(f"  Preprocessor fit on {train_mask.sum()} train rows ok")

    X_processed = preprocessor.transform(df[continuous + categorical + transfer])
    ohe               = preprocessor.named_transformers_["cat"]
    cat_names         = list(ohe.get_feature_names_out(categorical))
    all_feature_names = continuous + cat_names + transfer
    print(f"  Total features after encoding: {len(all_feature_names)}")

    df_proc = df.copy()
    for i, feat in enumerate(all_feature_names):
        df_proc[feat] = X_processed[:, i]
    df_proc = df_proc.drop(columns=[c for c in categorical if c in df_proc.columns])
    return df_proc, all_feature_names


# ================================================================
# STEP 9 — BUILD SEQUENCES
# ================================================================
def build_sequences(df, all_features, target_years, config):
    """
    Input  : features [t - SEQUENCE_LENGTH, ..., t-1]
    Target : Δoverall(t)
    """
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
# TRANSFER IMPACT ANALYSIS
# ================================================================
def analyze_transfer_impact(df, test_pred, test_ids, y_test, config):
    print("\n" + "="*60)
    print("TRANSFER IMPACT ANALYSIS")
    print("="*60)
    results = pd.DataFrame({
        "player_id": test_ids,
        "actual":    y_test,
        "predicted": test_pred,
        "abs_error": np.abs(y_test - test_pred),
    })
    test_info = df[df["Season_End_Year"] == config.TEST_YEAR][["id", "mid_season_transfer"]].copy()
    results   = results.merge(test_info, left_on="player_id", right_on="id", how="left")
    for flag, label in [(1, "WITH transfer"), (0, "WITHOUT transfer")]:
        subset = results[results["mid_season_transfer"] == flag]
        print(f"  {label:20s}: n={len(subset):4d}  MAE={subset['abs_error'].mean():.4f}")
    return results


# ================================================================
# FEATURE IMPORTANCE — Permutation
# ================================================================
def analyze_feature_importance_permutation(model, X_test, y_test, all_features, n_repeats=10):
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE — Permutation Importance")
    print("="*60)
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test, verbose=0).flatten())
    print(f"  Baseline MAE: {baseline_mae:.4f}")

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
    print("\n  Top 15 features (ΔMAE μετά από shuffle):")
    print(feat_imp.head(15).to_string(index=False))
    return feat_imp


# ================================================================
# BUILD TUNABLE MODEL
# ================================================================
def build_tunable_model(hp, input_shape):
    """
    Search space:
      lstm_units_1   : 32 / 64 / 128
      lstm_units_2   : 16 / 32 / 64
      num_lstm_layers: 1 ή 2
      dropout_rate   : 0.1 / 0.2 / 0.3 / 0.4
      use_batchnorm  : True / False
      dense_units    : 8 / 16 / 32
      learning_rate  : 1e-4 / 5e-4 / 1e-3 / 5e-3

    Reproducibility: fixed seed σε κάθε initializer.
    """
    lstm_units_1    = hp.Choice("lstm_units_1",    values=[32, 64, 128])
    lstm_units_2    = hp.Choice("lstm_units_2",    values=[16, 32, 64])
    num_lstm_layers = hp.Choice("num_lstm_layers", values=[1, 2])
    dropout_rate    = hp.Choice("dropout_rate",    values=[0.1, 0.2, 0.3, 0.4])
    use_batchnorm   = hp.Boolean("use_batchnorm")
    dense_units     = hp.Choice("dense_units",     values=[8, 16, 32])
    lr              = hp.Choice("learning_rate",   values=[1e-4, 5e-4, 1e-3, 5e-3])

    model = Sequential()

    # 1η LSTM layer
    return_seq = (num_lstm_layers == 2)
    model.add(LSTM(
        lstm_units_1,
        return_sequences=return_seq,
        input_shape=input_shape,
        kernel_initializer=GlorotUniform(seed=SEED),
        recurrent_initializer=Orthogonal(seed=SEED),
    ))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate, seed=SEED))

    # 2η LSTM layer (προαιρετική)
    if num_lstm_layers == 2:
        model.add(LSTM(
            lstm_units_2,
            return_sequences=False,
            kernel_initializer=GlorotUniform(seed=SEED),
            recurrent_initializer=Orthogonal(seed=SEED),
        ))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate, seed=SEED))

    # Dense head
    model.add(Dense(dense_units, activation="relu",
                    kernel_initializer=GlorotUniform(seed=SEED)))
    model.add(Dropout(dropout_rate / 2, seed=SEED))
    model.add(Dense(1, kernel_initializer=GlorotUniform(seed=SEED)))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ================================================================
# MAIN PIPELINE
# ================================================================
print("="*60)
print("LSTM PIPELINE — HYPERPARAMETER TUNING + WALK-FORWARD")
print("="*60)

# ── Steps 1–8: Data preparation ──────────────────────────────────
df = load_and_filter(config)
df = aggregate_season_level(df)
df = seasons_filter(df, config)
df = apply_transformations(df, config)
df = create_target(df)
df = create_time_split(df, config)

# Preprocessor fit ΜΟΝΟ στο train — παραμένει frozen για όλο το pipeline
continuous, categorical, transfer, preprocessor = build_preprocessor(
    df[df["split"] == "train"], config
)
df_proc, all_features_final = fit_transform_preprocessor(
    df, preprocessor, continuous, categorical, transfer, config
)

# ── Step 9: Build sequences ───────────────────────────────────────
print("\n" + "="*60)
print("STEP 9 — BUILD SEQUENCES")
print("="*60)

X_train, y_train, train_ids, _ = build_sequences(df_proc, all_features_final, config.TRAIN_YEARS,  config)
X_val,   y_val,   val_ids,   _ = build_sequences(df_proc, all_features_final, [config.VAL_YEAR],   config)
X_test,  y_test,  test_ids,  _ = build_sequences(df_proc, all_features_final, [config.TEST_YEAR],  config)

# Train+Val για το Phase B — ίδιος preprocessor, απλά συνένωση
X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

print(f"\n  Train     : {X_train.shape}")
print(f"  Val       : {X_val.shape}")
print(f"  Train+Val : {X_trainval.shape}  ← για Phase B (refit)")
print(f"  Test      : {X_test.shape}      ← άγνωστα δεδομένα, αξιολόγηση μόνο στο τέλος")

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


# ================================================================
# PHASE A — STEP 10: HYPERPARAMETER TUNING (Train → Val)
# ================================================================
print("\n" + "="*60)
print("PHASE A — STEP 10: HYPERPARAMETER TUNING")
print(f"  Αλγόριθμος : {config.TUNER_ALGO}")
print(f"  Max trials : {config.MAX_TRIALS}")
print("="*60)

if config.TUNER_ALGO == "bayesian":
    tuner = kt.BayesianOptimization(
        lambda hp: build_tunable_model(hp, INPUT_SHAPE),
        objective="val_mae",
        max_trials=config.MAX_TRIALS,
        executions_per_trial=config.EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIR,
        project_name=config.TUNER_PROJECT,
        seed=SEED,
        overwrite=True,
    )
else:
    tuner = kt.RandomSearch(
        lambda hp: build_tunable_model(hp, INPUT_SHAPE),
        objective="val_mae",
        max_trials=config.MAX_TRIALS,
        executions_per_trial=config.EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIR,
        project_name=config.TUNER_PROJECT,
        seed=SEED,
        overwrite=True,
    )

tuner.search_space_summary()

# Seed reset callback — κρίσιμο για Keras Tuner reproducibility.
# Κάθε trial δημιουργεί νέο μοντέλο, οπότε το global seed
# επαναφέρεται. Χωρίς αυτό, κάθε trial παίρνει διαφορετικά βάρη.
class ResetSeedCallback(tf.keras.callbacks.Callback):
    def on_trial_begin(self, trial, logs=None):
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

tuning_callbacks = [
    EarlyStopping(
        monitor="val_mae",
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=0,
    ),
    ResetSeedCallback(),
]

print("\nΈναρξη αναζήτησης...")
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=tuning_callbacks,
    verbose=0,
)

# Αποτελέσματα tuning
print("\n" + "="*60)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ TUNING")
print("="*60)
tuner.results_summary(num_trials=5)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nΒέλτιστες υπερπαράμετροι:")
print(f"  lstm_units_1    : {best_hps.get('lstm_units_1')}")
print(f"  lstm_units_2    : {best_hps.get('lstm_units_2')}")
print(f"  num_lstm_layers : {best_hps.get('num_lstm_layers')}")
print(f"  dropout_rate    : {best_hps.get('dropout_rate')}")
print(f"  use_batchnorm   : {best_hps.get('use_batchnorm')}")
print(f"  dense_units     : {best_hps.get('dense_units')}")
print(f"  learning_rate   : {best_hps.get('learning_rate')}")

# ── Val predictions για ensemble ─────────────────────────────────
# Χρησιμοποιούμε το best tuning model (train-only), ΟΧΙ το final.
# Το final model έχει δει το val κατά την εκπαίδευση (train+val),
# οπότε οι val predictions του θα ήταν in-sample → leakage στο
# ensemble weight search.
print("\n" + "="*60)
print("VAL PREDICTIONS (για ensemble weight search)")
print("="*60)
tuning_model = tuner.get_best_models(num_models=1)[0]
tuning_model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
val_pred_for_ensemble = tuning_model.predict(X_val, verbose=0).flatten()

pd.DataFrame({
    "id":        val_ids,
    "y_true":    y_val,
    "lstm_pred": val_pred_for_ensemble,
}).to_csv("lstm_val_predictions.csv", index=False)
print(f"  n = {len(val_pred_for_ensemble)} παίκτες")
print("  ok lstm_val_predictions.csv")


# ================================================================
# PHASE B — STEP 11: FINAL MODEL REFIT (Train+Val → Test)
#
# Walk-Forward Validation:
#   - Preprocessor: FROZEN (fit on train, δεν αλλάζει)
#   - Sequences   : X_trainval / y_trainval (ήδη έτοιμα από Step 9)
#   - Μοντέλο     : νέο instance με best_hps, εκπαίδευση σε train+val
#   - Test         : X_test / y_test (ήδη scaled με train-only scaler)
#
# Δεν χρειάζεται rebuild sequences ή refit preprocessor.
# Τα X_test που φτιάχτηκαν στο Step 9 είναι ήδη σωστά.
# ================================================================
print("\n" + "="*60)
print("PHASE B — STEP 11: FINAL MODEL REFIT (Train+Val)")
print("Walk-Forward: preprocessor frozen, μόνο το μοντέλο refit-άρεται")
print("="*60)

# Reset seeds πριν τη δημιουργία του final model
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

final_model = build_tunable_model(best_hps, INPUT_SHAPE)
final_model.summary()

# EarlyStopping δεν έχει νόημα χωρίς val set.
# ReduceLROnPlateau στο train loss ως μόνο callback.
final_callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        patience=7,
        factor=0.5,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\nΕκπαίδευση final model στο Train+Val...")
final_history = final_model.fit(
    X_trainval, y_trainval,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=final_callbacks,
    verbose=1,
)


# ================================================================
# STEP 12 — EVALUATION
# ================================================================
print("\n" + "="*60)
print("STEP 12 — EVALUATION")
print("="*60)

# Train+Val: sanity check — επιβεβαιώνει ότι το μοντέλο έμαθε
_         = evaluate(final_model, X_trainval, y_trainval, "TRAIN+VAL (sanity check)")

# Test: η κύρια αξιολόγηση — εντελώς unseen data
test_pred = evaluate(final_model, X_test, y_test, "TEST (2023) — ΚΥΡΙΑ ΑΞΙΟΛΟΓΗΣΗ")

# Transfer impact
test_results = analyze_transfer_impact(df_proc, test_pred, test_ids, y_test, config)

# Feature importance (στο final model + test set)
feat_importance = analyze_feature_importance_permutation(
    final_model, X_test, y_test, all_features_final, n_repeats=10
)


# ================================================================
# ΣΥΝΟΨΗ TUNING TRIALS (για αναφορά στη διπλωματική)
# ================================================================
print("\n" + "="*60)
print("ΣΥΝΟΨΗ TUNING TRIALS")
print("="*60)
all_trials = tuner.oracle.get_best_trials(num_trials=config.MAX_TRIALS)
trials_records = []
for trial in all_trials:
    hp_vals = trial.hyperparameters.values
    score   = trial.score
    trials_records.append({**hp_vals, "val_mae": score})

trials_df = pd.DataFrame(trials_records).sort_values("val_mae")
print(trials_df.head(10).to_string(index=False))


# ================================================================
# SAVE
# ================================================================
print("\n" + "="*60)
print("SAVING")
print("="*60)

final_model.save("lstm_final_tuned.keras")
test_results.to_csv("test_predictions_with_transfer.csv", index=False)
feat_importance.to_csv("feature_importance.csv", index=False)
trials_df.to_csv("tuning_results.csv", index=False)

pd.DataFrame({
    "id":        test_ids,
    "y_true":    y_test,
    "lstm_pred": test_pred,
}).to_csv("lstm_test_predictions.csv", index=False)

print("  lstm_final_tuned.keras")
print("  lstm_val_predictions.csv        <- tuning model (train-only) — για ensemble")
print("  lstm_test_predictions.csv       <- final model (train+val)   — κύρια αξιολόγηση")
print("  test_predictions_with_transfer.csv")
print("  feature_importance.csv")
print("  tuning_results.csv")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)

# ================================================================
# ΔΙΟΡΘΩΜΕΝΑ STEPS 13–16 — SHAP ANALYSIS
# Αντικατάστησε τα αντίστοιχα steps στο LSTM_SHAP.py
#
# ΑΛΛΑΓΕΣ:
#   1. DeepExplainer → GradientExplainer (σταθερό με BatchNorm/Dropout)
#   2. np.random.choice → np.random.default_rng(SEED) (reproducibility)
#   3. dependence_plot color: integer → string (σωστή παράμετρος)
# ================================================================

# ================================================================
# STEP 13 — SHAP ANALYSIS (GradientExplainer — σωστό για LSTM)
# ================================================================
print("\n" + "="*60)
print("STEP 13 — SHAP ANALYSIS (GradientExplainer for LSTM)")
print("Explaining predictions for the final model on the test set")
print("="*60)

# ------------------------------------------------------------------
# 13.1 Επιλογή background data — REPRODUCIBLE με fixed seed
# ------------------------------------------------------------------
rng_shap     = np.random.default_rng(SEED)   # ← FIX #2: αντί για np.random
n_background = min(150, X_train.shape[0])
bg_idx       = rng_shap.choice(X_train.shape[0], n_background, replace=False)
background_data = X_train[bg_idx].astype(np.float32)

print(f"  Background data shape : {background_data.shape}")
print(f"  Explainer             : GradientExplainer")
print(f"  Γιατί όχι DeepExplainer: ασταθές με BatchNorm + Dropout σε TF2")

# ------------------------------------------------------------------
# 13.2 GradientExplainer  ← FIX #1
# ------------------------------------------------------------------
# Το μοντέλο πρέπει να τρέχει σε inference mode (Dropout=off).
# Το GradientExplainer το χειρίζεται αυτόματα.
explainer = shap.GradientExplainer(final_model, background_data)

# ------------------------------------------------------------------
# 13.3 Υπολογισμός SHAP values — REPRODUCIBLE
# ------------------------------------------------------------------
n_shap_samples = min(400, X_test.shape[0])
shap_indices   = rng_shap.choice(X_test.shape[0], n_shap_samples, replace=False)  # ← FIX #2
X_test_sample  = X_test[shap_indices].astype(np.float32)

print(f"\n  Υπολογισμός SHAP για {n_shap_samples} test samples...")
print(f"  Εκτιμώμενος χρόνος: 2-5 λεπτά (CPU)...")

shap_values = explainer.shap_values(X_test_sample)

# GradientExplainer επιστρέφει list για regression (1 output)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Αν έχει 4 διαστάσεις: (n, seq, feat, 1) → (n, seq, feat)
if shap_values.ndim == 4:
    shap_values = shap_values[:, :, :, 0]

print(f"  SHAP values shape: {shap_values.shape}")
print(f"  (samples × timesteps × features)")

# Έλεγχος για NaN — αν υπάρχουν, κάτι πήγε στραβά
nan_count = np.isnan(shap_values).sum()
if nan_count > 0:
    print(f"  ⚠ ΠΡΟΕΙΔΟΠΟΙΗΣΗ: {nan_count} NaN SHAP values — έλεγξε το μοντέλο")
else:
    print(f"  ✓ Κανένα NaN — SHAP values έγκυρα")

# ------------------------------------------------------------------
# 13.4 Αποθήκευση
# ------------------------------------------------------------------
np.save("shap_values.npy", shap_values)
np.save("shap_indices.npy", shap_indices)   # ώστε να ξέρεις ποιοι παίκτες
print("  ✓ shap_values.npy")
print("  ✓ shap_indices.npy")


# ================================================================
# STEP 14 — SUMMARY PLOTS (ανά timestep)
# ================================================================
print("\n" + "="*60)
print("STEP 14 — SHAP SUMMARY PLOTS (per timestep)")
print("="*60)

timestep_labels = [f't-{config.SEQUENCE_LENGTH - i}' for i in range(config.SEQUENCE_LENGTH)]

for t in range(config.SEQUENCE_LENGTH):
    shap_values_t = shap_values[:, t, :]   # (n, n_features)
    X_test_t      = X_test_sample[:, t, :] # (n, n_features)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_t,
        X_test_t,
        feature_names=all_features_final,
        show=False,
        max_display=20,
    )
    plt.title(f'SHAP Feature Importance at {timestep_labels[t]}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{timestep_labels[t]}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ shap_summary_{timestep_labels[t]}.png")


# ================================================================
# STEP 15 — TEMPORAL HEATMAP
# ================================================================
print("\n" + "="*60)
print("STEP 15 — SHAP TEMPORAL HEATMAP")
print("="*60)

mean_abs_shap_per_timestep = np.mean(np.abs(shap_values), axis=0)  # (timesteps, features)

temporal_shap_df = pd.DataFrame(
    mean_abs_shap_per_timestep.T,
    index=all_features_final,
    columns=timestep_labels,
)
temporal_shap_df['total_importance'] = temporal_shap_df.sum(axis=1)
temporal_shap_df = temporal_shap_df.sort_values('total_importance', ascending=False)
temporal_shap_df = temporal_shap_df.drop(columns=['total_importance'])

top_n_features = min(20, len(all_features_final))
plt.figure(figsize=(14, 10))
sns.heatmap(
    temporal_shap_df.head(top_n_features),
    annot=True, fmt=".3f",
    cmap='viridis',
    linewidths=0.5,
    cbar_kws={'label': 'Mean |SHAP Value|', 'shrink': 0.8},
)
plt.title(f'Mean |SHAP| per Feature and Time Step (Top {top_n_features})', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("shap_temporal_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ shap_temporal_heatmap.png")

# Ποια features αλλάζουν σημασία μεταξύ time steps
print("\n  Features με μεγαλύτερη χρονική μεταβολή:")
shap_diff = mean_abs_shap_per_timestep[-1] - mean_abs_shap_per_timestep[0]
for idx in np.argsort(shap_diff)[-5:][::-1]:
    print(f"    ↑ {all_features_final[idx]:35s}: {shap_diff[idx]:+.4f}")
for idx in np.argsort(shap_diff)[:5]:
    print(f"    ↓ {all_features_final[idx]:35s}: {shap_diff[idx]:+.4f}")


# ================================================================
# STEP 16 — DEPENDENCE PLOTS  ← FIX #3: color ως string, όχι int
# ================================================================
print("\n" + "="*60)
print("STEP 16 — SHAP DEPENDENCE PLOTS")
print("="*60)

feature_to_analyze = 'npxG'

if feature_to_analyze in all_features_final:
    feature_idx = all_features_final.index(feature_to_analyze)
    feature_importance_per_timestep = mean_abs_shap_per_timestep[:, feature_idx]
    best_timestep = int(np.argmax(feature_importance_per_timestep))

    print(f"  Feature      : {feature_to_analyze}")
    print(f"  Best timestep: {timestep_labels[best_timestep]}")

    shap_values_best_t = shap_values[:, best_timestep, :]
    X_test_best_t      = X_test_sample[:, best_timestep, :]

    # Dependence plot με color = 'age'  ← FIX #3: string, όχι integer
    if 'age' in all_features_final:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            ind=feature_to_analyze,          # string — σωστό
            shap_values=shap_values_best_t,
            features=X_test_best_t,
            feature_names=all_features_final,
            interaction_index='age',         # ← FIX #3: string, όχι age_idx int
            alpha=0.6,
            x_jitter=0.3,
            show=False,
        )
        plt.title(f'SHAP Dependence: {feature_to_analyze} at {timestep_labels[best_timestep]}\n'
                  f'Colored by age', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{feature_to_analyze}_by_age.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ shap_dependence_{feature_to_analyze}_by_age.png")

    # Dependence plot για mid_season_transfer colored by npxG
    if 'mid_season_transfer' in all_features_final:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            ind='mid_season_transfer',        # string — σωστό
            shap_values=shap_values_best_t,
            features=X_test_best_t,
            feature_names=all_features_final,
            interaction_index=feature_to_analyze,  # ← FIX #3: string
            alpha=0.6,
            x_jitter=0.2,
            show=False,
        )
        plt.title(f'SHAP Dependence: Mid-Season Transfer at {timestep_labels[best_timestep]}\n'
                  f'Colored by {feature_to_analyze}', fontsize=13)
        plt.tight_layout()
        plt.savefig("shap_dependence_transfer_by_npxG.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ shap_dependence_transfer_by_npxG.png")
else:
    print(f"  ⚠ '{feature_to_analyze}' δεν βρέθηκε στα features")
    print(f"  Διαθέσιμα: {all_features_final[:10]}...")
