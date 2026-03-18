# ================================================================
# LSTM PIPELINE — ΜΕ HYPERPARAMETER TUNING (Keras Tuner)
# Διπλωματική Εργασία: ML στο Ποδόσφαιρο
# Target: Δoverall (Time Series Forecasting)
#
# ΣΤΡΑΤΗΓΙΚΗ TUNING:
#   1. Αναζήτηση υπερπαραμέτρων χρησιμοποιώντας train → val
#   2. Final model: εκπαίδευση στο train+val με τις βέλτιστες παραμέτρους
#   3. Αξιολόγηση στο test set (άγνωστα δεδομένα 2023)
# ================================================================

# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt

from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
class Config:
    TRAIN_YEARS    = [2018, 2019, 2020, 2021]
    VAL_YEAR       = 2022
    TEST_YEAR      = 2023
    ALL_YEARS      = set(range(2018, 2024))

    SEQUENCE_LENGTH = 2
    MIN_SEASONS     = SEQUENCE_LENGTH + 1
    MIN_MINUTES     = 5

    APPLY_LOG_TRANSFORM = True
    SCALER_TYPE         = "robust"

    # Σταθερές παράμετροι (δεν tunable)
    EPOCHS      = 100
    PATIENCE    = 15
    BATCH_SIZE  = 32      # ← μπορεί να γίνει tunable αν θες

    DATA_PATH = "C:/Users/evagg/Desktop/Final.xlsx"

    # ── Keras Tuner settings ──────────────────────────────────────
    # Αλγόριθμος: "random" (γρήγορο) ή "bayesian" (πιο αποδοτικό)
    TUNER_ALGO      = "bayesian"
    MAX_TRIALS      = 30       # πόσα διαφορετικά configs θα δοκιμαστούν
    EXECUTIONS_PER_TRIAL = 1   # 1 = ταχύτερο, 2-3 = πιο σταθερό αποτέλεσμα
    TUNER_DIR       = "tuner_results"
    TUNER_PROJECT   = "lstm_delta_overall"

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
# STEPS 1–9: Ίδια με πριν (load, aggregate, filter, transform,
#            target, split, preprocessor, sequences)
# ================================================================

def load_and_filter(config):
    print("="*60)
    print("STEP 1 — LOAD & FILTER")
    print("="*60)
    df = pd.read_excel(config.DATA_PATH)
    df = df[df["Pos"] != "GK"].copy()
    df = df[df["Mins_Per_90"] >= config.MIN_MINUTES].copy()
    df = df[df["Season_End_Year"].isin(config.ALL_YEARS)].copy()
    print(f"Συνολικές γραμμές : {df.shape[0]}")
    print(f"Μοναδικοί παίκτες : {df['id'].nunique()}")
    return df


def aggregate_season_level(df):
    print("\n" + "="*60)
    print("STEP 2 — SEASON-LEVEL AGGREGATION")
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
    return agg_df


def seasons_filter(df, config):
    print("\n" + "="*60)
    print("STEP 3 — SEASONS FILTER")
    print("="*60)

    def has_consecutive_seasons(season_list):
        s = sorted(season_list)
        return all(s[i+1] - s[i] == 1 for i in range(len(s)-1))

    season_groups = df.groupby("id")["Season_End_Year"].apply(list)
    valid_ids = season_groups[
        season_groups.apply(lambda s: len(s) >= config.MIN_SEASONS and has_consecutive_seasons(s))
    ].index
    df = df[df["id"].isin(valid_ids)].copy()
    print(f"  Παίκτες που περνούν το φίλτρο: {df['id'].nunique()}")
    return df


def apply_transformations(df, config):
    print("\n" + "="*60)
    print("STEP 4 — TRANSFORMATIONS")
    print("="*60)
    if config.APPLY_LOG_TRANSFORM:
        for col in ["value_eur", "wage_eur"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])
    return df


def create_target(df):
    print("\n" + "="*60)
    print("STEP 5 — TARGET: Δoverall")
    print("="*60)
    df = df.sort_values(["id", "Season_End_Year"]).reset_index(drop=True)
    df["Δoverall"] = df.groupby("id")["overall"].diff()
    return df


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


def build_preprocessor(df_train, config):
    continuous  = [c for c in FIFA_FEATURES + FBREF_FEATURES + UNDERSTAT_FEATURES + STATIC_FEATURES
                   if c in df_train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df_train.columns]
    transfer    = [c for c in TRANSFER_FEATURES    if c in df_train.columns]

    scaler = RobustScaler() if config.SCALER_TYPE == "robust" else StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",      scaler, continuous),
            ("cat",      OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("transfer", "passthrough", transfer),
        ]
    )
    return continuous, categorical, transfer, preprocessor


def fit_transform_preprocessor(df, preprocessor, continuous, categorical, transfer, config,
                                fit_mask=None):
    """
    fit_mask: boolean Series — ποιες γραμμές να χρησιμοποιηθούν για fit.
    Αν None, χρησιμοποιεί μόνο τις train γραμμές (default behavior).
    Αυτό μας επιτρέπει να κάνουμε refit στο train+val για το final model.
    """
    if fit_mask is None:
        fit_mask = df["split"] == "train"

    preprocessor.fit(df[fit_mask][continuous + categorical + transfer])

    X_processed = preprocessor.transform(df[continuous + categorical + transfer])
    ohe         = preprocessor.named_transformers_["cat"]
    cat_names   = list(ohe.get_feature_names_out(categorical))
    all_feature_names = continuous + cat_names + transfer

    df_proc = df.copy()
    for i, feat in enumerate(all_feature_names):
        df_proc[feat] = X_processed[:, i]
    df_proc = df_proc.drop(columns=[c for c in categorical if c in df_proc.columns])
    return df_proc, all_feature_names


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
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), player_ids, years


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
    print(f"  R²           : {r2:.4f}")
    print(f"  Direction Acc: {dir_acc:.2%}")
    return pred


def analyze_transfer_impact(df, test_pred, test_ids, y_test, config):
    results  = pd.DataFrame({
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


def analyze_feature_importance_permutation(model, X_test, y_test, all_features, n_repeats=10):
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test, verbose=0).flatten())
    rng          = np.random.default_rng(SEED)
    importances  = []
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
    print("\nTop 15 features (ΔMAE μετά από shuffle):")
    print(feat_imp.head(15).to_string(index=False))
    return feat_imp


# ================================================================
# HYPERPARAMETER TUNING — build_tunable_model
# ================================================================
def build_tunable_model(hp, input_shape):
    """
    Ορίζει το search space για τον Keras Tuner.

    Παράμετροι που ψάχνουμε:
    ─────────────────────────────────────────────────────────────
    lstm_units_1   : μέγεθος 1ης LSTM (32 / 64 / 128)
    lstm_units_2   : μέγεθος 2ης LSTM (16 / 32 / 64)
    num_lstm_layers: 1 ή 2 LSTM layers
    dropout_rate   : 0.1 / 0.2 / 0.3 / 0.4
    use_batchnorm  : True/False
    dense_units    : μέγεθος τελευταίου Dense layer (8 / 16 / 32)
    learning_rate  : 1e-4 / 5e-4 / 1e-3 / 5e-3
    ─────────────────────────────────────────────────────────────
    """
    lstm_units_1    = hp.Choice("lstm_units_1",    values=[32, 64, 128])
    lstm_units_2    = hp.Choice("lstm_units_2",    values=[16, 32, 64])
    num_lstm_layers = hp.Choice("num_lstm_layers", values=[1, 2])
    dropout_rate    = hp.Choice("dropout_rate",    values=[0.1, 0.2, 0.3, 0.4])
    use_batchnorm   = hp.Boolean("use_batchnorm")
    dense_units     = hp.Choice("dense_units",     values=[8, 16, 32])
    lr              = hp.Choice("learning_rate",   values=[1e-4, 5e-4, 1e-3, 5e-3])

    model = Sequential()

    # ── 1η LSTM layer ─────────────────────────────────────────────
    return_seq = (num_lstm_layers == 2)
    model.add(LSTM(lstm_units_1, return_sequences=return_seq, input_shape=input_shape))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # ── 2η LSTM layer (προαιρετική) ────────────────────────────────
    if num_lstm_layers == 2:
        model.add(LSTM(lstm_units_2, return_sequences=False))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # ── Dense head ────────────────────────────────────────────────
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate / 2))
    model.add(Dense(1))

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
print("LSTM PIPELINE — ΜΕ HYPERPARAMETER TUNING")
print("="*60)

# ── Steps 1–8: Προεπεξεργασία (ίδια με πριν) ─────────────────────
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

# ── Step 9: Build sequences ──────────────────────────────────────
print("\n" + "="*60)
print("STEP 9 — BUILD SEQUENCES")
print("="*60)

X_train, y_train, train_ids, _ = build_sequences(df_proc, all_features_final, config.TRAIN_YEARS,    config)
X_val,   y_val,   val_ids,   _ = build_sequences(df_proc, all_features_final, [config.VAL_YEAR],     config)
X_test,  y_test,  test_ids,  _ = build_sequences(df_proc, all_features_final, [config.TEST_YEAR],    config)

# train+val sequences — χρησιμοποιούνται για εκπαίδευση του final model
X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

print(f"  Train     : {X_train.shape}")
print(f"  Val       : {X_val.shape}")
print(f"  Train+Val : {X_trainval.shape}")
print(f"  Test      : {X_test.shape}")

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


# ================================================================
# STEP 10 — HYPERPARAMETER TUNING
# ================================================================
print("\n" + "="*60)
print("STEP 10 — HYPERPARAMETER TUNING")
print("="*60)
print(f"  Αλγόριθμος  : {config.TUNER_ALGO}")
print(f"  Max trials  : {config.MAX_TRIALS}")
print(f"  Search space: lstm_units_1/2, num_lstm_layers,")
print(f"                dropout_rate, use_batchnorm, dense_units, lr")

# Επιλογή αλγόριθμου
if config.TUNER_ALGO == "bayesian":
    tuner = kt.BayesianOptimization(
        lambda hp: build_tunable_model(hp, INPUT_SHAPE),
        objective="val_mae",          # ελαχιστοποιούμε MAE στο validation
        max_trials=config.MAX_TRIALS,
        executions_per_trial=config.EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIR,
        project_name=config.TUNER_PROJECT,
        seed=SEED,
        overwrite=True,               # False = συνέχιση από εκεί που σταμάτησε
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

# Callbacks κατά τη φάση αναζήτησης
tuning_callbacks = [
    EarlyStopping(
        monitor="val_mae",
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=0,           # 0 = σιωπηλό — αποφεύγουμε πάρα πολύ output
    ),
]

print("\nΈναρξη αναζήτησης...")
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=tuning_callbacks,
    verbose=0,              # 0 = ελάχιστο output, 1 = progress bar ανά trial
)

# ── Εμφάνιση αποτελεσμάτων tuning ────────────────────────────────
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

# ================================================================
# ΑΠΟΘΗΚΕΥΣΗ VAL PREDICTIONS ΓΙΑ ENSEMBLE
# ================================================================
# ΚΡΙΣΙΜΟ: Χρησιμοποιούμε το ΚΑΛΥΤΕΡΟ μοντέλο της φάσης tuning
# (εκπαιδευμένο μόνο στο train), ΟΧΙ το final model (train+val).
#
# Γιατί: Το ensemble χρειάζεται val predictions για να βρει το
# βέλτιστο βάρος w. Αν χρησιμοποιήσουμε το final model (που έχει
# δει το val κατά την εκπαίδευση), οι val προβλέψεις θα είναι
# in-sample → τεχνητά καλές → λανθασμένο βάρος w → leakage.
#
# Το tuner.get_best_models(1)[0] επιστρέφει το μοντέλο με τα
# restore_best_weights=True, δηλαδή τα βάρη στο best val_mae epoch.
# ================================================================
print("\n" + "="*60)
print("ΑΠΟΘΗΚΕΥΣΗ VAL PREDICTIONS (για ensemble)")
print("="*60)

tuning_model = tuner.get_best_models(num_models=1)[0]
# Ορισμός input shape (απαιτείται πριν την πρόβλεψη)
tuning_model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))

val_pred_for_ensemble = tuning_model.predict(X_val, verbose=0).flatten()

pd.DataFrame({
    "id":        val_ids,
    "y_true":    y_val,
    "lstm_pred": val_pred_for_ensemble,
}).to_csv("lstm_val_predictions.csv", index=False)

print(f"  Προβλέψεις val από tuning model (train-only): {len(val_pred_for_ensemble)} παίκτες")
print("✓ lstm_val_predictions.csv")


# ================================================================
# STEP 11 — FINAL MODEL: Εκπαίδευση στο TRAIN+VAL
# ================================================================
print("\n" + "="*60)
print("STEP 11 — FINAL MODEL (Train+Val)")
print("="*60)
print("Ο scaler γίνεται refit στο train+val για maximum πληροφορία.")

# ── Refit preprocessor στο train+val ─────────────────────────────
# Σημείωση: Αυτό είναι σωστό — στο final model δεν υπάρχει
# validation set, οπότε μπορούμε να χρησιμοποιήσουμε όλα τα
# διαθέσιμα labeled δεδομένα για τον scaler.
trainval_mask = df["split"].isin(["train", "val"])
_, categorical_refit, transfer_refit, preprocessor_final = build_preprocessor(
    df[trainval_mask], config
)
df_proc_final, _ = fit_transform_preprocessor(
    df, preprocessor_final, continuous, categorical, transfer, config,
    fit_mask=trainval_mask
)

# ── Rebuild sequences με νέο scaler ──────────────────────────────
X_trainval_final, y_trainval_final, _, _ = build_sequences(
    df_proc_final, all_features_final,
    config.TRAIN_YEARS + [config.VAL_YEAR], config
)
X_test_final, y_test_final, test_ids_final, _ = build_sequences(
    df_proc_final, all_features_final, [config.TEST_YEAR], config
)

print(f"  Train+Val (για εκπαίδευση): {X_trainval_final.shape}")
print(f"  Test (για αξιολόγηση)     : {X_test_final.shape}")

# ── Κατασκευή final model με βέλτιστες παραμέτρους ───────────────
final_model = build_tunable_model(best_hps, INPUT_SHAPE)
final_model.summary()

# EarlyStopping δεν έχει νόημα χωρίς val set.
# Χρησιμοποιούμε ReduceLROnPlateau στο train loss.
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
    X_trainval_final, y_trainval_final,
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

# Αξιολόγηση στο train+val (για reference — δεν αντικατοπτρίζει
# γενίκευση, απλώς επιβεβαιώνει ότι το model έμαθε σωστά)
trainval_pred = evaluate(final_model, X_trainval_final, y_trainval_final, "TRAIN+VAL (reference)")

# ✅ Κύρια αξιολόγηση: test set (άγνωστα δεδομένα 2023)
test_pred = evaluate(final_model, X_test_final, y_test_final, "TEST (2023) — ΚΥΡΙΑ ΑΞΙΟΛΟΓΗΣΗ")

# Transfer analysis
print("\n" + "="*60)
print("TRANSFER IMPACT ANALYSIS")
print("="*60)
test_results = analyze_transfer_impact(df_proc_final, test_pred, test_ids_final, y_test_final, config)

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE — Permutation Importance")
print("="*60)
feat_importance = analyze_feature_importance_permutation(
    final_model, X_test_final, y_test_final, all_features_final, n_repeats=10
)


# ================================================================
# ΒΑΘΜΟΛΟΓΙΑ ΟΛΩΝ ΤΩΝ TRIALS (για αναφορά στη διπλωματική)
# ================================================================
print("\n" + "="*60)
print("ΣΥΝΟΨΗ TUNING TRIALS")
print("="*60)

all_trials = tuner.oracle.get_best_trials(num_trials=config.MAX_TRIALS)
trials_records = []
for trial in all_trials:
    hp_vals = trial.hyperparameters.values
    score   = trial.score   # val_mae
    trials_records.append({**hp_vals, "val_mae": score})

trials_df = pd.DataFrame(trials_records).sort_values("val_mae")
print(trials_df.head(10).to_string(index=False))


# ================================================================
# SAVE
# ================================================================
print("\n" + "="*60)
print("SAVING")
print("="*60)

final_model.save("lstm_final_tuned.h5")
test_results.to_csv("test_predictions.csv", index=False)
feat_importance.to_csv("feature_importance.csv", index=False)
trials_df.to_csv("tuning_results.csv", index=False)

pd.DataFrame({
    "id":         test_ids_final,
    "y_true":     y_test_final,
    "lstm_pred":  test_pred,
}).to_csv("lstm_test_predictions.csv", index=False)

print("✓ lstm_final_tuned.h5")
print("✓ test_predictions.csv")
print("✓ feature_importance.csv")
print("✓ tuning_results.csv")
print("✓ lstm_val_predictions.csv   ← από tuning model (train-only) — χρησιμοποιείται για ensemble weight search")
print("✓ lstm_test_predictions.csv  ← από final model (train+val)   — χρησιμοποιείται για ensemble test evaluation")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)