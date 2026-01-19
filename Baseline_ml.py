import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb


#Load Dataset
df = pd.read_excel("C:/Users/evagg/Desktop/Final.xlsx")

df = df.rename(columns={
    "Season_End_Year": "season",
    "overall": "overall_rating",
    "potential": "potential_rating"
    })

df = df.sort_values(["id", "season"])

# Remove goalkeepers
df = df[df["Pos"] != "GK"].copy()

lag_vars = [
    "overall_rating",
    "potential_rating",
    "age",
    "Mins_Per_90"
]

for col in lag_vars:
    df[f"{col}_lag1"] = df.groupby("id")[col].shift(1)
    
# Current-year change
df["overall_delta_t"] = (
    df["overall_rating"] - df["overall_rating_lag1"]
)

# TARGET: next-season change
df["target_delta_t+1"] = (
    df.groupby("id")["overall_rating"].shift(-1)
    - df["overall_rating"]
)

#Drop invalid rows
df = df.dropna(subset=[
    "overall_rating_lag1",
    "target_delta_t+1"
])


#df_role = df[df["role"] == "Defender"].copy()
#print(f"Training Linear Regression model for role: {'Defender'}")


def calculate_metrics(y_true, y_pred, model_name='Model'):
        """
        Calculate comprehensive evaluation metrics
        
        Metrics explained:
        - MAE: Average absolute error (in rating points) - MOST INTERPRETABLE
        - RMSE: Root mean squared error - PENALIZES LARGE ERRORS
        - R²: Variance explained (1.0 = perfect, 0.0 = baseline)
        - MAPE: Mean absolute percentage error
        """
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }


def print_metrics(metrics):
        """Pretty print metrics"""
        print(f"\n📊 {metrics['Model']} Performance:")
        print(f"   MAE:  {metrics['MAE']:.3f} (lower is better)")
        print(f"   RMSE: {metrics['RMSE']:.3f} (lower is better)")
        print(f"   R²:   {metrics['R²']:.3f} (higher is better, max 1.0)")
        print(f"   MAPE: {metrics['MAPE']:.3f}% (lower is better)")


# Dataset Split (Not Random)
train = df[df["season"] <= 2020].copy()
val   = df[df["season"] == 2021].copy()
test  = df[df["season"] == 2022].copy()

print("Train seasons:", train["season"].unique())
print("Val seasons:", val["season"].unique())
print("Test seasons:", test["season"].unique())

num_features = [
    # FIFA
    "overall_rating_lag1",
    "potential_rating_lag1",
    "overall_delta_t",
    "age",
    
    "movement_reactions",
    "mentality_composure",
    "value_eur",
    "attacking_short_passing",
    "mentality_vision",
    "passing",
    "dribbling",
    "mentality_positioning",
    "shooting",
    "attacking_finishing",
    "power_shot_power",
    "attacking_volleys",
    "defending",
    "mentality_interceptions",
    "defending_marking_awareness",
    "skill_long_passing",
    "defending_standing_tackle",
    
    #Understat
    "xGBuildup",
    "xGChain",     
    
    #FBRef
    "Rec_Receiving",   
    "Mid 3rd_Touches",
    "Carries_Carries",
    "Touches_Touches",
    "Live_Touches"   
    ]


# PERSISTENCE MODEL (NAIVE BASELINE)


def persistence_model(train, val, test, target_col='target_delta_t+1'):
    """
    Persistence Model for DELTA prediction:
    Predict that next-season change = 0
    
    Δoverall(t+1) = 0
    """
    print("\n" + "=" * 80)
    print("MODEL 1: PERSISTENCE (ZERO-CHANGE) BASELINE")
    print("=" * 80)

    print("\n📖 Explanation:")
    print("   Assumes player rating remains stable")
    print("   Predicts Δoverall(t+1) = 0")

    # Predictions (ALL ZEROS)
    train_pred = np.zeros(len(train))
    val_pred   = np.zeros(len(val))
    test_pred  = np.zeros(len(test))

    # Ground truth
    y_train = train[target_col].values
    y_val   = val[target_col].values
    y_test  = test[target_col].values

    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, 'Persistence (Train)')
    val_metrics   = calculate_metrics(y_val, val_pred, 'Persistence (Val)')
    test_metrics  = calculate_metrics(y_test, test_pred, 'Persistence (Test)')

    print_metrics(train_metrics)
    print_metrics(val_metrics)
    print_metrics(test_metrics)

    return {
        'model': None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    }

def linear_regression_model(train, val, test, num_features, target_col='target_delta_t+1'):
    
    print("\n" + "=" * 80)
    print("MODEL 3: LINEAR REGRESSION")
    print("=" * 80)
    
    print("\n📖 Explanation:")
    print("   Fits a linear equation: overall = w₁*lag1 + w₂*lag2 + w₃*age + ...")
    print("   Assumes linear relationships between features and target")
    print(f"   Using {len(num_features)} features")
    
    # Prepare data
    X_train = train[num_features].copy()
    y_train = train[target_col].copy()
    X_val = val[num_features].copy()
    y_val = val[target_col].copy()
    X_test = test[num_features].copy()
    y_test = test[target_col].copy()
    
    # Handle missing values (simple imputation with median)
    for col in num_features:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_val[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
    
    # Remove rows with missing target
    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print(f"\n📊 Training samples: {len(X_train):,}")
    print(f"📊 Validation samples: {len(X_val):,}")
    print(f"📊 Test samples: {len(X_test):,}")
    
    # Train model
    print("\n🔧 Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, 'Linear Reg (Train)')
    val_metrics = calculate_metrics(y_val, val_pred, 'Linear Reg (Val)')
    test_metrics = calculate_metrics(y_test, test_pred, 'Linear Reg (Test)')
    
    print_metrics(train_metrics)
    print_metrics(val_metrics)
    print_metrics(test_metrics)
    
    # Feature importance (coefficients)
    print("\n🔍 Top 10 Feature Coefficients:")
    coef_df = pd.DataFrame({
        'Feature': num_features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df.head(10).to_string(index=False))
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'feature_importance': coef_df
    }


def ridge_regression_model(train, val, test, num_features, target_col='target_delta_t+1', alpha=1.0):
   
    print("\n" + "=" * 80)
    print(f"MODEL 4: RIDGE REGRESSION (alpha={alpha})")
    print("=" * 80)
    
    print("\n📖 Explanation:")
    print("   Like Linear Regression but penalizes large coefficients")
    print("   Prevents overfitting, especially with many features")
    print(f"   Alpha={alpha} controls regularization strength")
    
    # Prepare data (same as Linear Regression)
    X_train = train[num_features].copy()
    y_train = train[target_col].copy()
    X_val = val[num_features].copy()
    y_val = val[target_col].copy()
    X_test = test[num_features].copy()
    y_test = test[target_col].copy()
    
    # Handle missing values
    for col in num_features:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_val[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
    
    # Remove missing targets
    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Standardize features (important for Ridge!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\n🔧 Training Ridge Regression (alpha={alpha})...")
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, 'Ridge (Train)')
    val_metrics = calculate_metrics(y_val, val_pred, 'Ridge (Val)')
    test_metrics = calculate_metrics(y_test, test_pred, 'Ridge (Test)')
    
    print_metrics(train_metrics)
    print_metrics(val_metrics)
    print_metrics(test_metrics)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    }

def random_forest_model(train, val, test, num_features, target_col='target_delta_t+1',
                       n_estimators=100, max_depth=15):
    
    print("\n" + "=" * 80)
    print(f"MODEL 5: RANDOM FOREST")
    print("=" * 80)
    
    print("\n📖 Explanation:")
    print("   Ensemble of decision trees")
    print("   Each tree trained on random subset of data and features")
    print("   Final prediction = average of all tree predictions")
    print(f"   Parameters: {n_estimators} trees, max_depth={max_depth}")
    
    # Prepare data
    X_train = train[num_features].copy()
    y_train = train[target_col].copy()
    X_val = val[num_features].copy()
    y_val = val[target_col].copy()
    X_test = test[num_features].copy()
    y_test = test[target_col].copy()
    
    # Handle missing values
    for col in num_features:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_val[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
    
    # Remove missing targets
    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Train model
    print(f"\n🔧 Training Random Forest ({n_estimators} trees, max_depth={max_depth})...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, 'Random Forest (Train)')
    val_metrics = calculate_metrics(y_val, val_pred, 'Random Forest (Val)')
    test_metrics = calculate_metrics(y_test, test_pred, 'Random Forest (Test)')
    
    print_metrics(train_metrics)
    print_metrics(val_metrics)
    print_metrics(test_metrics)
    
    # Feature importance
    print("\n🔍 Top 15 Most Important Features:")
    importance_df = pd.DataFrame({
        'Feature': num_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance_df.head(15).to_string(index=False))
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'feature_importance': importance_df
    }

def xgboost_model(train, val, test, num_features, target_col='target_delta_t+1',
                 n_estimators=100, max_depth=6, learning_rate=0.1):
    
    print("\n" + "=" * 80)
    print(f"MODEL 6: XGBOOST")
    print("=" * 80)
    
    print("\n📖 Explanation:")
    print("   Gradient Boosting: builds trees sequentially")
    print("   Each tree corrects errors of previous trees")
    print("   Often the best performer on tabular data")
    print(f"   Parameters: {n_estimators} rounds, depth={max_depth}, lr={learning_rate}")
    
    # Prepare data
    X_train = train[num_features].copy()
    y_train = train[target_col].copy()
    X_val = val[num_features].copy()
    y_val = val[target_col].copy()
    X_test = test[num_features].copy()
    y_test = test[target_col].copy()
    
    # Handle missing values (XGBoost can handle NaN, but let's be explicit)
    '''
    for col in num_features:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_val[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
    '''
    
    # Remove missing targets
    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Train model
    print(f"\n🔧 Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Fit with validation for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("   ✅ Training complete!")
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, train_pred, 'XGBoost (Train)')
    val_metrics = calculate_metrics(y_val, val_pred, 'XGBoost (Val)')
    test_metrics = calculate_metrics(y_test, test_pred, 'XGBoost (Test)')
    
    print_metrics(train_metrics)
    print_metrics(val_metrics)
    print_metrics(test_metrics)





