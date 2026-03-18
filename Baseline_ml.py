# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ================================================================
#  CONFIG
# ================================================================
class Config:
    TRAIN_YEARS = [2018, 2019, 2020, 2021]
    VAL_YEAR    = 2022
    TEST_YEAR   = 2023
    ALL_YEARS   = set(range(2018, 2024))   # όλες οι διαθέσιμες σεζόν

    MIN_MINUTES = 5
    LAGS        = [1,2]

    
    MIN_SEASONS = max(LAGS) + 2  

    SCALER_TYPE = "robust"   
    DATA_PATH   = "C:/Users/evagg/Desktop/Final.xlsx"

config = Config()



# ================================================================
#  COLUMN DEFINITIONS
# ================================================================
LOG_COLS = ["value_eur", "wage_eur"]

BASE_FIFA_FEATURES = [
    "age", "value_eur_log", "wage_eur_log",
    "movement_reactions", "mentality_composure",
    "attacking_short_passing", "mentality_vision",
    "passing", "dribbling", "mentality_positioning", "shooting",
    "attacking_finishing", "power_shot_power", "attacking_volleys",
    "defending", "mentality_interceptions", "defending_marking_awareness",
    "skill_long_passing", "defending_standing_tackle",
]
BASE_FBREF_FEATURES = [
    "Touches_Touches", "Mid 3rd_Touches", "Att 3rd_Touches",
    "Live_Touches", "Carries_Carries", "Rec_Receiving", "Mins_Per_90",
]
BASE_UNDERSTAT_FEATURES = [
    "xGChain", "npxG", "xA", "xG",
    "assists", "key_passes", "xGBuildup", "shots",
]

STATIC_FEATURES      = ["height_cm", "weight_kg"]
TRANSFER_FEATURES    = ["mid_season_transfer", "played_for_multiple"]
CATEGORICAL_FEATURES = ["preferred_foot", "Pos"]


# ================================================================
#  STEP 1: LOAD & FILTER
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


# ================================================================
#  STEP 2: SEASON-LEVEL AGGREGATION
# ================================================================
def aggregate_season_level(df):
    print("\n" + "="*60)
    print("STEP 2 — SEASON-LEVEL AGGREGATION")
    print("="*60)

    minute_col = "Mins_Per_90" if "Mins_Per_90" in df.columns else "minutes"

    # Transfer detection
    club_counts   = df.groupby(["id", "Season_End_Year"])["Squad"].nunique()
    transfer_flag = club_counts.reset_index(name="num_clubs")
    transfer_flag["mid_season_transfer"] = (transfer_flag["num_clubs"] > 1).astype(int)
    print(f"Mid-season transfer cases: {transfer_flag['mid_season_transfer'].sum()}")

    fbref_cols = [
        "Mins_Per_90", "Touches_Touches", "Def Pen_Touches", "Def 3rd_Touches",
        "Mid 3rd_Touches", "Att 3rd_Touches", "Att Pen_Touches", "Live_Touches",
        "Att_Take", "Succ_Take", "Succ_percent_Take", "Tkld_Take",
        "Carries_Carries", "TotDist_Carries", "PrgDist_Carries", "PrgC_Carries",
        "Final_Third_Carries", "CPA_Carries", "Mis_Carries", "Dis_Carries",
        "Rec_Receiving", "PrgR_Receiving",
    ]
    # FIFA + Understat + Categorical: latest value per season
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

    print(f"  FBref (weighted avg): {len(fbref_cols)}")
    print(f"  Latest value cols   : {len(latest_cols)}")

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

    print("\nAggregating...")
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
    print(f"Duplicates (should be 0): {dupes}")
    print(f"Aggregated shape        : {agg_df.shape}")
    return agg_df


# ================================================================
#  STEP 3: SEASONS FILTER
# ================================================================
def seasons_filter(df, config):
    print("\n" + "="*60)
    print("STEP 3 — SEASONS FILTER")
    print("="*60)

    def has_consecutive_seasons(season_list):
        """Ελέγχει αν οι σεζόν είναι συνεχόμενες (χωρίς κενά)."""
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
    print(f"  Παίκτες που περνούν το φίλτρο : {df['id'].nunique()}")
    print(f"  Κατανομή αριθμού σεζόν:")
    for n, c in counts.value_counts().sort_index().items():
        print(f"    {n} σεζόν: {c} παίκτες")
        
    # Πόσοι παίκτες απορρίφθηκαν;
    total_players = len(season_groups)
    kept_players = len(valid_ids)
    print(f"  Συνολικοί παίκτες : {total_players}")
    print(f"  Παίκτες που κρατήθηκαν : {kept_players} ({kept_players/total_players*100:.1f}%)")
    print(f"  Απορρίφθηκαν : {total_players - kept_players}")
    return df


# ================================================================
#  STEP 4: LOG TRANSFORM
# ================================================================
def apply_log_transform(df):
    print("\n" + "="*60)
    print("STEP 4 — LOG TRANSFORM")
    print("="*60)
    transformed = []
    for col in LOG_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_log"] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)
        transformed.append(col)
    print(f"  Transformed: {transformed}")
    return df


# ================================================================
#  STEP 5: CREATE LAG FEATURES
# ================================================================
def create_lag_features(df, config):
    lags = config.LAGS
    df   = df.sort_values(["id", "Season_End_Year"]).copy()

    time_varying = BASE_FIFA_FEATURES + BASE_FBREF_FEATURES + BASE_UNDERSTAT_FEATURES
    time_varying = [c for c in time_varying if c in df.columns]
    time_varying = list(dict.fromkeys(time_varying))  # deduplicate

    lag_cols = []
    for lag in lags:
        for col in time_varying:
            name       = f"{col}_lag{lag}"
            df[name]   = df.groupby("id")[col].shift(lag)
            lag_cols.append(name)

    print(f"\n" + "="*60)
    print("STEP 5 — LAG FEATURES")
    print("="*60)
    print(f"  {len(lags)} lags x {len(time_varying)} base cols = {len(lag_cols)} lag features")
    return df, lag_cols


# ================================================================
#  STEP 6: CREATE TARGET
# ================================================================
def create_target(df):
    print("\n" + "="*60)
    print("STEP 6 — TARGET: Δoverall")
    print("="*60)
    df = df.sort_values(["id", "Season_End_Year"]).copy()
    df["Δoverall"] = df.groupby("id")["overall"].diff()

    # Αφαίρεση overall και potential από τα features
    cols_to_drop = [c for c in ["overall", "potential"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped from features: {cols_to_drop}")
    print(f"  Δoverall stats:\n{df['Δoverall'].describe().round(3)}")
    return df


# ================================================================
#  STEP 7: DROP NaN ROWS
# ================================================================
def drop_nan_rows(df, lag_cols):
    before = len(df)
    df     = df.dropna(subset=["Δoverall"] + lag_cols).copy()
    print(f"\n" + "="*60)
    print("STEP 7 — DROP NaN ROWS")
    print("="*60)
    print(f"  Dropped          : {before - len(df)} rows")
    print(f"  Remaining rows   : {len(df)}")
    print(f"  Unique players   : {df['id'].nunique()}")
    print(f"  Available seasons: {sorted(df['Season_End_Year'].unique())}")
    
    usable_seasons = sorted(df['Season_End_Year'].unique())
    print(f"  Usable seasons after dropping NaN: {usable_seasons}")
    print(f"  Για lag={config.LAGS}, η πρώτη usable σεζόν είναι {usable_seasons[0]}")
    return df


# ================================================================
#  STEP 8: TIME-BASED SPLIT
# ================================================================
def create_time_split(df, config):
    print("\n" + "="*60)
    print("STEP 8 — TIME-BASED SPLIT")
    print("="*60)

    df_train = df[df["Season_End_Year"].isin(config.TRAIN_YEARS)].copy()
    df_val   = df[df["Season_End_Year"] == config.VAL_YEAR].copy()
    df_test  = df[df["Season_End_Year"] == config.TEST_YEAR].copy()

    print(f"  Train : {len(df_train):>4} rows  ({df_train['id'].nunique()} players)")
    print(f"  Val   : {len(df_val):>4} rows  ({df_val['id'].nunique()} players)")
    print(f"  Test  : {len(df_test):>4} rows  ({df_test['id'].nunique()} players)")

    if len(df_test) == 0:
        raise ValueError(
            f"TEST SET IS EMPTY!\n"
            f"Available seasons: {sorted(df['Season_End_Year'].unique())}\n"
            f"TEST_YEAR={config.TEST_YEAR} not found."
        )
    return df_train, df_val, df_test


# ================================================================
#  STEP 9: BUILD PREPROCESSOR
# ================================================================
def build_preprocessor(df_train, lag_cols, config):
    continuous  = [c for c in lag_cols + STATIC_FEATURES  if c in df_train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES         if c in df_train.columns]
    transfer    = [c for c in TRANSFER_FEATURES             if c in df_train.columns]

    print(f"\n" + "="*60)
    print("STEP 9 — PREPROCESSOR")
    print("="*60)
    print(f"  Continuous  (RobustScaler) : {len(continuous)}")
    print(f"  Categorical (OHE)          : {len(categorical)}")
    print(f"  Transfer    (passthrough)  : {len(transfer)}")

    scaler = RobustScaler() if config.SCALER_TYPE == "robust" else StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",      scaler,
             continuous),
            ("cat",      OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
            ("transfer", "passthrough",
             transfer),
        ]
    )
    return continuous, categorical, transfer, preprocessor


# ================================================================
#  MAIN PIPELINE
# ================================================================
print("="*60)
print("BASELINE MODELS PIPELINE - ΧΩΡΙΣ CROSS VALIDATION")
print("="*60)

# ── Steps 1-3: Load → Aggregate → Filter ────────────────────────
df = load_and_filter(config)
df = aggregate_season_level(df)
df = seasons_filter(df, config)
df = df.sort_values(["id", "Season_End_Year"]).reset_index(drop=True)

# ── Step 4: Log transform ────────────────────────────────────────
df = apply_log_transform(df)

# ── Step 5: Lag features ──────────────────────────────────
df, lag_cols = create_lag_features(df, config)

# ── Step 6: Target ──────────────────────────────────────────────
df = create_target(df)

# ── Step 7: Drop NaN rows ───────────────────────────────────────
df = drop_nan_rows(df, lag_cols)

# ── Step 8: Split ───────────────────────────────────────────────
df_train, df_val, df_test = create_time_split(df, config)

# ── Step 9: Preprocessor ────────────────────────────────────────
(continuous_features,
 categorical_features,
 transfer_features,
 preprocessor) = build_preprocessor(df_train, lag_cols, config)

feature_cols = continuous_features + categorical_features + transfer_features
target       = "Δoverall"

# Sanity check
for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
    missing = [c for c in feature_cols if c not in split.columns]
    if missing:
        raise ValueError(f"Missing features in {name}: {missing}")

print(f"\nTotal features: {len(feature_cols)}")
print(f"Target        : {target}")

# ================================================================
#  STATISTICS FOR Δoverall
# ================================================================
# Στατιστικά ανά split (train/val/test)
print("\n--- BY DATA SPLIT ---")
print("\nTrain Set (2018-2021):")
train_stats = df_train['Δoverall'].describe()
print(f"  Mean     : {train_stats['mean']:.4f}")
print(f"  Std      : {train_stats['std']:.4f}")
print(f"  Min      : {train_stats['min']:.4f}")
print(f"  Max      : {train_stats['max']:.4f}")
print(f"  Count    : {train_stats['count']:.0f}")

print("\nValidation Set (2022):")
val_stats = df_val['Δoverall'].describe()
print(f"  Mean     : {val_stats['mean']:.4f}")
print(f"  Std      : {val_stats['std']:.4f}")
print(f"  Min      : {val_stats['min']:.4f}")
print(f"  Max      : {val_stats['max']:.4f}")
print(f"  Count    : {val_stats['count']:.0f}")

print("\nTest Set (2023):")
test_stats = df_test['Δoverall'].describe()
print(f"  Mean     : {test_stats['mean']:.4f}")
print(f"  Std      : {test_stats['std']:.4f}")
print(f"  Min      : {test_stats['min']:.4f}")
print(f"  Max      : {test_stats['max']:.4f}")
print(f"  Count    : {test_stats['count']:.0f}")



# ================================================================
#  STEPS 10-11: FIT MODELS ON TRAIN → EVALUATE ON VAL
# ================================================================
print("\n" + "="*60)
print("STEPS 10-11: FIT ON TRAIN → EVALUATE ON VAL")
print("="*60)

baseline_models = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(alpha=1.0, random_state=42),
    "RandomForest":     RandomForestRegressor(
                            n_estimators=100, random_state=42,
                            n_jobs=-1, max_depth=10, min_samples_split=10),
    "XGBoost":          XGBRegressor(
                            n_estimators=300, max_depth=5,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8,
                            objective="reg:squarederror",
                            random_state=42, n_jobs=-1),
}

X_train = df_train[feature_cols];  y_train = df_train[target]
X_val   = df_val[feature_cols];    y_val   = df_val[target]

results     = []
best_models = {}

for model_name, model in baseline_models.items():
    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    model),
    ])
    pipeline.fit(X_train, y_train)
    best_models[model_name] = pipeline

    y_pred_val = pipeline.predict(X_val)
    mae_val    = mean_absolute_error(y_val, y_pred_val)
    rmse_val   = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_val     = r2_score(y_val, y_pred_val)
    print(f"  Val → MAE:{mae_val:.4f}  RMSE:{rmse_val:.4f}  R²:{r2_val:.4f}")

    results.append({
        "model":    model_name,
        "val_mae":  mae_val,
        "val_rmse": rmse_val,
        "val_r2":   r2_val,
    })

results_df      = pd.DataFrame(results).sort_values("val_mae")
best_model_name = results_df.iloc[0]["model"]
print(f"\n{'='*60}\nMODELS RANKED BY VAL MAE\n{'='*60}")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_model_name}")


# ================================================================
#  STEP 12: REFIT BEST MODEL ON TRAIN+VAL → PREDICT TEST
# ================================================================
print(f"\n{'='*60}")
print("STEP 12: REFIT ON TRAIN+VAL → PREDICT TEST")
print(f"{'='*60}")

df_tv  = pd.concat([df_train, df_val], ignore_index=True)
X_tv   = df_tv[feature_cols];    y_tv   = df_tv[target]
X_test = df_test[feature_cols];  y_test = df_test[target]

# Rebuild preprocessor ώστε το RobustScaler να fitάρει στο train+val
(cont_final, cat_final, trans_final, preproc_final) = build_preprocessor(
    df_tv, lag_cols, config
)

best_model_final = Pipeline([
    ("preprocessor", preproc_final),
    ("regressor",    baseline_models[best_model_name]),
])
best_model_final.fit(X_tv, y_tv)
y_pred_test = best_model_final.predict(X_test)

mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test   = r2_score(y_test, y_pred_test)

print(f"\nFinal Test Results ({best_model_name}):")
print(f"  MAE : {mae_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  R²  : {r2_test:.4f}")


# ================================================================
#  TRANSFER IMPACT ANALYSIS
# ================================================================
test_results = pd.DataFrame({
    "player_id":           df_test["id"].values,
    "actual":              y_test.values,
    "predicted":           y_pred_test,
    "abs_error":           np.abs(y_test.values - y_pred_test),
    "mid_season_transfer": df_test["mid_season_transfer"].values,
})
n_transfer      = (test_results["mid_season_transfer"] == 1).sum()
n_no_transfer   = (test_results["mid_season_transfer"] == 0).sum()
transfer_mae    = test_results.loc[test_results["mid_season_transfer"]==1, "abs_error"].mean()
no_transfer_mae = test_results.loc[test_results["mid_season_transfer"]==0, "abs_error"].mean()

print(f"\n{'='*60}\nTRANSFER IMPACT ANALYSIS\n{'='*60}")
print(f"  With transfer    : n={n_transfer},    MAE={transfer_mae:.4f}")
print(f"  Without transfer : n={n_no_transfer}, MAE={no_transfer_mae:.4f}")
print(f"  Difference       : {transfer_mae - no_transfer_mae:.4f}")


# ================================================================
#  VISUALIZATION
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
errors = y_test.values - y_pred_test

# Actual vs Predicted
axes[0,0].scatter(y_test, y_pred_test, alpha=0.6, s=40)
axes[0,0].plot([y_test.min(), y_test.max()],
               [y_test.min(), y_test.max()], "r--", lw=2)
axes[0,0].set(xlabel="Πραγματικό Δoverall", ylabel="Προβλεπόμενο Δoverall",
              title=f"Test – {best_model_name}\nR²={r2_test:.3f}")
axes[0,0].grid(True, alpha=0.3)

# Error distribution
axes[0,1].hist(errors, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
axes[0,1].axvline(0, color="r", linestyle="--", lw=2)
axes[0,1].set(xlabel="Σφάλμα", ylabel="Συχνότητα",
              title=f"Κατανομή Σφαλμάτων\nMAE={mae_test:.3f}")
axes[0,1].grid(True, alpha=0.3)

# Model comparison
bars = axes[0,2].bar(range(len(results_df)), results_df["val_mae"],
                     tick_label=results_df["model"],
                     color=["skyblue", "lightgreen", "salmon", "orange"])
axes[0,2].set(ylabel="Validation MAE", title="Σύγκριση Μοντέλων")
axes[0,2].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, results_df["val_mae"]):
    axes[0,2].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom")

# Player development curves
for pid in df_test["id"].unique()[:5]:
    pdata = df[df["id"] == pid].sort_values("Season_End_Year")
    if "overall" in pdata.columns:
        axes[1,0].plot(pdata["Season_End_Year"], pdata["overall"],
                       marker="o", label=f"P{pid}", lw=2)
axes[1,0].set(xlabel="Έτος", ylabel="Overall Rating",
              title="Εξέλιξη Overall (Δείγμα)")
axes[1,0].legend(fontsize=7)
axes[1,0].grid(True, alpha=0.3)

# Feature importance (RF / XGBoost)
if best_model_name in ["RandomForest", "XGBoost"]:
    try:
        importances = best_model_final.named_steps["regressor"].feature_importances_
        ohe         = best_model_final.named_steps["preprocessor"].named_transformers_["cat"]
        ohe_names   = list(ohe.get_feature_names_out(cat_final))
        all_names   = (cont_final + ohe_names + trans_final)[:len(importances)]
        feat_imp    = (pd.DataFrame({"feature": all_names, "importance": importances})
                       .sort_values("importance", ascending=False)
                       .head(10))
        axes[1,1].barh(feat_imp["feature"], feat_imp["importance"], color="lightgreen")
        axes[1,1].set(xlabel="Importance", title=f"Top 10 Features ({best_model_name})")
        axes[1,1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Feature importance plot failed: {e}")
        axes[1,1].text(0.5, 0.5, "N/A", ha="center", va="center",
                       transform=axes[1,1].transAxes)

# Residuals
axes[1,2].scatter(y_pred_test, errors, alpha=0.6, s=40)
axes[1,2].axhline(0, color="r", linestyle="--", lw=2)
axes[1,2].set(xlabel="Προβλεπόμενες Τιμές", ylabel="Υπόλοιπα",
              title="Residuals Plot")
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ================================================================
#  SAVE RESULTS
# ================================================================
predictions_df = pd.DataFrame({
    "id":                      df_test["id"].values,
    "Season_End_Year":         df_test["Season_End_Year"].values,
    "actual_Δoverall":         y_test.values,
    f"pred_{best_model_name}": y_pred_test,
    "mid_season_transfer":     df_test["mid_season_transfer"].values,
})

output_path = "scenario2_results_no_cv.xlsx"
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer,     sheet_name="Model_Comparison",  index=False)
    predictions_df.to_excel(writer, sheet_name="Predictions",       index=False)
    test_results.to_excel(writer,   sheet_name="Transfer_Analysis",  index=False)
    pd.DataFrame({
        "Metric": ["Train rows", "Val rows", "Test rows", "Players",
                   "Best Model", "MAE", "R²",
                   "Transfer Cases", "Transfer MAE", "No-Transfer MAE"],
        "Value":  [len(df_train), len(df_val), len(df_test),
                   df["id"].nunique(), best_model_name,
                   mae_test, r2_test,
                   n_transfer, transfer_mae, no_transfer_mae],
    }).to_excel(writer, sheet_name="Statistics", index=False)

print(f"\nResults saved: {output_path}")

print(f"\n{'='*60}\nCONCLUSIONS\n{'='*60}")
print(f"  Players in final dataset : {df['id'].nunique()}")
print(f"  Best model               : {best_model_name}")
print(f"  Test MAE                 : {mae_test:.3f}")
print(f"  Test R²                  : {r2_test:.3f}")
if   mae_test < 1.0: print("  Accuracy: GOOD     (<1.0 MAE)")
elif mae_test < 2.0: print("  Accuracy: MODERATE (1-2 MAE)")
else:                print("  Accuracy: LOW      (>2.0 MAE)")
print(f"  Transfer MAE diff        : {transfer_mae - no_transfer_mae:.3f}")


# ================================================================
#  SAVE PREDICTIONS
# ================================================================
# Get Random Forest predictions for validation set
rf_model = best_models.get("RandomForest")
if rf_model is not None:
    rf_val_pred = rf_model.predict(X_val)
    
    pd.DataFrame({
        "id": df_val["id"].values,
        "Season_End_Year": df_val["Season_End_Year"].values,
        "y_true": y_val.values,
        "rf_pred": rf_val_pred,
    }).to_csv("rf_val_predictions.csv", index=False)
    
    print("Random Forest validation predictions saved to rf_val_predictions.csv")
else:
    print("Random Forest model not found in best_models")

# Save test predictions 

if rf_model is not None:
    # Re-fit Random Forest on train+val
    rf_final = Pipeline([
        ("preprocessor", preproc_final),
        ("regressor", RandomForestRegressor(
            n_estimators=100, random_state=42,
            n_jobs=-1, max_depth=10, min_samples_split=10)),
    ])
    rf_final.fit(X_tv, y_tv)
    rf_test_pred = rf_final.predict(X_test)
    
    pd.DataFrame({
        "id": df_test["id"].values,
        "Season_End_Year": df_test["Season_End_Year"].values,
        "y_true": y_test.values,
        "rf_pred": rf_test_pred,
    }).to_csv("rf_test_predictions.csv", index=False)
    
    print("Random Forest test predictions saved to rf_test_predictions.csv")