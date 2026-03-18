import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# DATA LOADING & BASIC PROFILING

df = pd.read_excel("C:/Users/evagg/Desktop/Final.xlsx")

df = df[df["Pos"] != "GK"].copy()

df = df.rename(columns={
    "Season_End_Year": "season",
    "overall": "overall_rating",
    "potential": "potential_rating"
})

# Remove goalkeepers
df = df[df["Pos"] != "GK"].copy()

print("=" * 80)
print("DATA LOADING & BASIC PROFILING")
print("=" * 80)

print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"📅 Date Range: {df['season'].min()} - {df['season'].max()}")
print(f"👥 Unique Players: {df['id'].nunique():,}")
print(f"⚽ Unique Teams: {df['club_name'].nunique():,}")

print("\n" + "─" * 80)
print("Data Types Summary:")
print(df.dtypes.value_counts())

print("\n" + "─" * 80)
print("Target Variable (Overall) Statistics:")
print(df['overall_rating'].describe())


def analyze_temporal_coverage(df):
    """Analyze temporal aspects of the dataset"""
    print("\n" + "=" * 80)
    print("TEMPORAL COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Players per season
    players_per_season = df.groupby('season')['id'].nunique()
    print("\n📅 Players per Season:")
    print(players_per_season)
    
    # Seasons per player
    seasons_per_player = df.groupby('id').size()
    print("\n👤 Seasons per Player Distribution:")
    print(seasons_per_player.describe())
    
    # Career length analysis
    player_careers = df.groupby('id').agg({
        'season': ['min', 'max', 'count']
    })
    player_careers.columns = ['First_Season', 'Last_Season', 'Total_Seasons']
    player_careers['Career_Span'] = player_careers['Last_Season'] - player_careers['First_Season'] + 1
    
    print("\n⏱️ Career Span Analysis:")
    print(player_careers['Career_Span'].value_counts().sort_index())
    
    # Visualization
    plt.figure(figsize =(8,5))
    
    # Players per season
    plt.bar(players_per_season.index, players_per_season.values, color='steelblue')
    plt.xlabel('Season')
    plt.ylabel('Number of Players')
    plt.title('Players per Season', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Seasons per player histogram
    plt.figure(figsize =(8,5))
    
    plt.hist(seasons_per_player, bins=range(1, seasons_per_player.max()+2), 
                    color='teal', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Seasons')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Seasons per Player', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Career span distribution
    plt.figure(figsize =(8,5))
    plt.hist(player_careers['Career_Span'], bins=range(1, 10), 
                    color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Career Span (years)')
    plt.ylabel('Number of Players')
    plt.title('Career Span Distribution', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # Sequence completeness
    plt.figure(figsize =(8,5))
    complete_sequences = (player_careers['Career_Span'] == player_careers['Total_Seasons']).sum()
    incomplete_sequences = len(player_careers) - complete_sequences
    plt.pie([complete_sequences, incomplete_sequences], 
                   labels=['Complete', 'With Gaps'],
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'],
                   startangle=90)
    plt.title('Career Sequence Completeness', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return player_careers


def analyze_target_variable(df):
    
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS (OVERALL)")
    print("=" * 80)

    # Overall statistics by season
    overall_by_season = df.groupby('season')['overall_rating'].agg(['mean', 'std', 'min', 'max'])
    print("\n📊 Overall Statistics by Season:")
    print(overall_by_season.round(2))

    # Overall by position
    if 'Pos' in df.columns:
        overall_by_position = df.groupby('Pos')['overall_rating'].agg(['mean', 'std', 'count'])
        print("\n⚽ Overall Statistics by Position:")
        print(overall_by_position.round(2))

    # Overall distribution Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['overall_rating'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(df['overall_rating'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["overall_rating"].mean():.1f}')
    plt.xlabel('Overall Rating')
    plt.ylabel('Frequency')
    plt.title('Overall Rating Distribution', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Overall by season (boxplot)
    plt.figure(figsize=(10, 5))
    df.boxplot(column='overall_rating', by='season')
    plt.xlabel('Season')
    plt.ylabel('Overall Rating')
    plt.title('Overall Distribution by Season', fontweight='bold')
    plt.suptitle("")
    plt.xticks(rotation=45)
    plt.show()

    # Overall evolution over time
    plt.figure(figsize=(8, 5))
    plt.plot(overall_by_season.index, overall_by_season['mean'],
             marker='o', linewidth=2, color='green')
    plt.fill_between(
        overall_by_season.index,
        overall_by_season['mean'] - overall_by_season['std'],
        overall_by_season['mean'] + overall_by_season['std'],
        alpha=0.3, color='green'
    )
    plt.xlabel('Season')
    plt.ylabel('Average Overall')
    plt.title('Overall Rating Evolution (Mean ± Std)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Overall vs Age Scatter Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df['age'], df['overall_rating'], alpha=0.3, s=10, color='purple')

    z = np.polyfit(df['age'].dropna(), df.loc[df['age'].notna(), 'overall_rating'], 2)
    p = np.poly1d(z)
    age_range = np.linspace(df['age'].min(), df['age'].max(), 100)
    plt.plot(age_range, p(age_range), "r-", linewidth=2, label='Trend')

    plt.xlabel('Age')
    plt.ylabel('Overall Rating')
    plt.title('Overall vs Age', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Overall by position (violin plot)
    if 'Pos' in df.columns and df['Pos'].nunique() < 20:
        positions = df['Pos'].value_counts().head(10).index
        df_pos = df[df['Pos'].isin(positions)]

        plt.figure(figsize=(10, 5))
        sns.violinplot(data=df_pos, x='Pos', y='overall_rating')
        plt.xlabel('Position')
        plt.ylabel('Overall Rating')
        plt.title('Overall Distribution by Position (Top 10)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.show()

    # Q-Q plot
    plt.figure(figsize=(6, 6))
    stats.probplot(df['overall_rating'].dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    '''
    # Normality test
   stat, p_value = shapiro(df['overall_rating'].sample(min(5000, len(df))))
   print(f"\n📈 Shapiro-Wilk Normality Test: statistic={stat:.4f}, p-value={p_value:.4f}")
   if p_value > 0.05:
       print("   → Overall appears to be normally distributed")
   else:
       print("   → Overall does NOT appear to be normally distributed")
   '''

        
        
def analyze_temporal_patterns(df):
    print("\n" + "=" * 80)
    print("5. TEMPORAL PATTERNS IN OVERALL")
    print("=" * 80)


    df_sorted = df.sort_values(['id', 'season'])

    # Calculate year-over-year changes
    df_sorted['overall_change'] = df_sorted.groupby('id')['overall_rating'].diff()
    df_sorted['overall_change_pct'] = (
        df_sorted['overall_change'] / df_sorted.groupby('id')['overall_rating'].shift(1)
    ) * 100

    print("\n📉 Year-over-Year Overall Changes:")
    print(df_sorted['overall_change'].describe())

    # Age groups
    df_sorted['age_group'] = pd.cut(
        df_sorted['age'],
        bins=[0, 22, 26, 30, 50],
        labels=['Young (≤22)', 'Peak (23-26)', 'Mature (27-30)', 'Veteran (31+)']
    )

    change_by_age = df_sorted.groupby('age_group')['overall_change'].agg(['mean', 'std', 'count'])
    print("\n👥 Average Overall Change by Age Group:")
    print(change_by_age.round(3))

    # Distribution of changes
    plt.figure(figsize=(8, 6))
    plt.hist(df_sorted['overall_change'].dropna(), bins=50,
             color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Year-over-Year Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Overall Changes', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Changes by age group
    plt.figure(figsize=(8, 5))
    df_sorted.boxplot(column='overall_change', by='age_group')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Age Group')
    plt.ylabel('Overall Change')
    plt.title('Overall Changes by Age Group', fontweight='bold')
    plt.suptitle("")
    plt.xticks(rotation=15)
    plt.show()

    # Sample career trajectories
    plt.figure(figsize=(10, 6))
    sample_players = df_sorted['id'].value_counts().head(20).index

    for id in sample_players[:15]:
        player_data = df_sorted[df_sorted['id'] == id].sort_values('season')
        plt.plot(
            player_data['season'],
            player_data['overall_rating'],
            marker='o', alpha=0.6, linewidth=1
        )

    plt.xlabel('Season')
    plt.ylabel('Overall Rating')
    plt.title('Sample Career Trajectories (15 players)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Lag plot (overall_t vs overall_t-1)
    df_sorted['overall_lag1'] = df_sorted.groupby('id')['overall_rating'].shift(1)
    valid_lag = df_sorted[['overall_rating', 'overall_lag1']].dropna()

    plt.figure(figsize=(7, 7))
    plt.scatter(valid_lag['overall_lag1'], valid_lag['overall_rating'],
                alpha=0.3, s=10, color='teal')

    min_val = min(valid_lag.min())
    max_val = max(valid_lag.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    corr = valid_lag['overall_rating'].corr(valid_lag['overall_lag1'])
    plt.text(
        0.05, 0.95, f'Correlation: {corr:.3f}',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.xlabel('Overall (t-1)')
    plt.ylabel('Overall (t)')
    plt.title('Lag Plot: Overall(t) vs Overall(t-1)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

    return df_sorted

        
def analyze_key_features(df):
    """Detailed analysis of key features"""
    print("\n" + "=" * 80)
    print("7. KEY FEATURES ANALYSIS")
    print("=" * 80)
    
    # Age analysis
    print("\n👤 AGE ANALYSIS:")
    print(df['age'].describe())
    
    age_overall = df.groupby('age')['overall_rating'].agg(['mean', 'std', 'count'])
    print("\nAverage Overall by Age:")
    print(age_overall.round(2))
    
    # Playing time analysis
    if 'Mins_Per_90' in df.columns:
        print("\n⏱️ PLAYING TIME ANALYSIS:")
        print(df['Mins_Per_90'].describe())
        
        # Correlation with overall
        corr = df[['Mins_Per_90', 'overall_rating']].corr().iloc[0, 1]
        print(f"\nCorrelation (Playing Time vs Overall): {corr:.3f}")
    
    # Performance metrics
    if 'goals' in df.columns and 'xG' in df.columns:
        print("\n⚽ PERFORMANCE METRICS:")
        print(f"Total Goals: {df['goals'].sum():.0f}")
        print(f"Total xG: {df['xG'].sum():.2f}")
        print(f"Goals vs xG Ratio: {(df['goals'].sum() / df['xG'].sum()):.3f}")
    
    # Visualization
    plt.figure(figsize=(8, 5))
    
    # Age vs Overall
    plt.scatter(df['age'], df['overall_rating'], alpha=0.3, s=10)
    plt.plot(age_overall.index, age_overall['mean'], color='red', linewidth=2, label='Mean')
    plt.xlabel('Age')
    plt.ylabel('Overall Rating')
    plt.title('Age vs Overall (with Mean Trend)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Age distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['age'], bins=range(16, 45), color='teal', edgecolor='black', alpha=0.7)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # Value vs Overall
    plt.figure(figsize=(8, 5))
    if 'value_eur' in df.columns:
        valid_data = df[(df['value_eur'] > 0) & (df['value_eur'] < df['value_eur'].quantile(0.99))]
        plt.scatter(valid_data['overall_rating'], valid_data['value_eur']/1e6, alpha=0.3, s=10, color='green')
        plt.xlabel('Overall Rating')
        plt.ylabel('Market Value (millions €)')
        plt.title('Overall vs Market Value', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Playing time distribution
    plt.figure(figsize=(8, 5))
    if 'Mins_Per_90' in df.columns:
        plt.hist(df['Mins_Per_90'].dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
        plt.xlabel('Minutes per 90')
        plt.ylabel('Frequency')
        plt.title('Playing Time Distribution', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    
    

# FEATURE IMPORTANCE (PRELIMINARY)

def preliminary_feature_importance(df, n_features=20):
    """Quick Random Forest to identify important features"""
    print("\n" + "=" * 80)
    print("8. PRELIMINARY FEATURE IMPORTANCE")
    print("=" * 80)
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['id', 'season', 'overall_rating']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Remove columns with too many missing values
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
            valid_cols.append(col)
    
    print(f"\n🎯 Using {len(valid_cols)} features for importance analysis")
    
    # Prepare dataset
    X = df[valid_cols].fillna(df[valid_cols].median())
    y = df['overall_rating']
    
    # Remove rows with missing target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"📊 Training set size: {len(X):,} samples")
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n⭐ Top {n_features} Most Important Features:")
    print(importances.head(n_features).to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(8, 5))
    top_features = importances.head(n_features)
    
    plt.barh(range(len(top_features)),
         top_features['Importance'].values,
         color='steelblue')

    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Importance Score')
    plt.title(f'Top {n_features} Features by Importance (Random Forest)',
          fontsize=12, fontweight='bold')
    plt.invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.show()
    return importances


'''
def visualize_missing_seasons_issue(df):
    """
    Visualize how missing seasons affect time series
    """
    
    # Find players with gaps
    player_gaps = df.groupby('id').apply(
        lambda x: (x['season'].diff() > 1).any()
    )
    players_with_gaps = player_gaps[player_gaps].index.tolist()
    
    if not players_with_gaps:
        print("No players with gaps found")
        return
    
    sample_player = players_with_gaps[0]
    player_data = df[df['id'] == sample_player].sort_values('season')
    
    plt.figure(figsize=(8, 5))
    
    # Plot 1: Player's career with gaps
    plt.plot(player_data['season'], 
                   player_data['overall_rating'], 
                   'o-', linewidth=2, markersize=8)
    plt.xlabel('Season End Year')
    plt.ylabel('Overall Rating')
    plt.title(f'Player {sample_player}: Career Timeline')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Highlight gaps
    for i in range(len(player_data)-1):
        year_diff = player_data['season'].iloc[i+1] - player_data['season'].iloc[i]
        if year_diff > 1:
            plt.axvspan(
                player_data['season'].iloc[i] + 0.5,
                player_data['season'].iloc[i+1] - 0.5,
                alpha=0.2, color='red', label='Gap' if i == 0 else ""
            )
    
    # Plot 2: Your approach vs Correct approach
    # Simulate both calculations
    player_data['your_delta'] = player_data['overall_rating'].shift(-1) - player_data['overall_rating']
    
    # Correct delta (accounting for gaps)
    year_diffs = player_data['season'].diff().shift(-1)
    player_data['correct_delta'] = (
        (player_data['overall_rating'].shift(-1) - player_data['overall_rating']) / 
        year_diffs
    ).fillna(0)
    
    plt.figure(figsize=(8, 5))

    
    plt.bar(player_data['season'] - 0.2, 
                   player_data['your_delta'].fillna(0), 
                   width=0.4, label='Your Δ (WRONG)', alpha=0.7)
    plt.bar(player_data['season'] + 0.2, 
                   player_data['correct_delta'], 
                   width=0.4, label='Correct Δ', alpha=0.7)
    plt.xlabel('Season End Year')
    plt.ylabel('ΔOverall')
    plt.title('ΔOverall Calculation Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()
    
    # Plot 3: Gap statistics across dataset
    gap_stats = df.groupby('id').apply(
        lambda x: pd.Series({
            'total_seasons': x['season'].nunique(),
            'has_gaps': (x['season'].diff() > 1).any(),
            'max_gap': x['season'].diff().max() if len(x) > 1 else 0
        })
    ).reset_index()
    
    plt.figure(figsize=(8, 5))

    gap_counts = gap_stats['has_gaps'].value_counts()
    plt.pie(gap_counts.values, labels=['No Gaps', 'Has Gaps'], 
                  autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Players with vs without Gaps')
    plt.show()
    
    # Plot 4: Season distribution
    plt.figure(figsize=(8, 5))

    season_counts = df['season'].value_counts().sort_index()
    plt.bar(season_counts.index, season_counts.values)
    plt.xlabel('Season')
    plt.ylabel('Number of Players')
    plt.title('Player Distribution by Season')
    plt.tick_params(axis='x', rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n📊 GAP ANALYSIS:")
    print(f"Total players: {df['id'].nunique()}")
    print(f"Players with gaps: {gap_counts.get(True, 0)} ({gap_counts.get(True, 0)/df['id'].nunique()*100:.1f}%)")
    print(f"Average seasons per player: {gap_stats['total_seasons'].mean():.1f}")
    print(f"Maximum gap found: {gap_stats['max_gap'].max()} years")
    '''
    



def visualize_season_consistency(df):
    """
    Visualizes:
    1. Players with 1 season
    2. Players with ≥2 consecutive seasons
    3. Players with gaps (multiple time blocks)
    4. Why gap-aware Δoverall is necessary
    """

    df = df.sort_values(["id", "season"]).copy()

    # --------------------------------------------------
    # PLAYER-LEVEL SUMMARY
    # --------------------------------------------------
    df["season_diff"] = df.groupby("id")["season"].diff()
    df["has_gap"] = df["season_diff"] > 1

    player_summary = (
        df.groupby("id")
        .agg(
            num_seasons=("season", "nunique"),
            has_gap=("has_gap", "any")
        )
        .reset_index()
    )

    def categorize(row):
        if row["num_seasons"] == 1:
            return "1 season only"
        elif row["has_gap"]:
            return "≥2 seasons with gaps"
        else:
            return "≥2 consecutive seasons"

    player_summary["category"] = player_summary.apply(categorize, axis=1)

    # --------------------------------------------------
    # PLOT 1 — Player category distribution
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    player_summary["category"].value_counts().plot(kind="bar")
    plt.title("Player Categories by Seasonal Availability")
    plt.ylabel("Number of Players")
    plt.xticks(rotation=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # PLOT 2 — Example players per category
    # --------------------------------------------------
    for category in ["1 season only", "≥2 consecutive seasons", "≥2 seasons with gaps"]:
        example_ids = player_summary[player_summary["category"] == category]["id"]
        if example_ids.empty:
            continue

        pid = example_ids.iloc[0]
        pdata = df[df["id"] == pid]

        plt.figure(figsize=(7, 4))
        plt.plot(pdata["season"], pdata["overall_rating"], "o-", linewidth=2)
        plt.title(f"Example Player ({category}) — ID {pid}")
        plt.xlabel("Season")
        plt.ylabel("Overall Rating")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # PLOT 3 — Consecutive blocks visualization
    # --------------------------------------------------
    df["time_block"] = df.groupby("id")["has_gap"].cumsum()

    sample_gap_player = player_summary[
        player_summary["category"] == "≥2 seasons with gaps"
    ]["id"]

    if not sample_gap_player.empty:
        pid = sample_gap_player.iloc[0]
        pdata = df[df["id"] == pid]

        plt.figure(figsize=(8, 5))

        for block_id, block in pdata.groupby("time_block"):
            plt.plot(
                block["season"],
                block["overall_rating"],
                marker="o",
                linewidth=2,
                label=f"Block {block_id}"
            )

        plt.title(f"Player {pid}: Consecutive Time Blocks")
        plt.xlabel("Season")
        plt.ylabel("Overall Rating")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # PLOT 4 — Why naive Δoverall is wrong
    # --------------------------------------------------
    pdata = pdata.sort_values("season")

    pdata["naive_delta"] = pdata["overall_rating"].shift(-1) - pdata["overall_rating"]

    pdata["valid_delta"] = (
        pdata.groupby("time_block")["overall_rating"].shift(-1)
        - pdata["overall_rating"]
    )

    plt.figure(figsize=(8, 5))
    plt.bar(pdata["season"] - 0.2, pdata["naive_delta"], width=0.4, label="Naive Δoverall")
    plt.bar(pdata["season"] + 0.2, pdata["valid_delta"], width=0.4, label="Gap-aware Δoverall")
    plt.xlabel("Season")
    plt.ylabel("ΔOverall")
    plt.title("Naive vs Gap-Aware ΔOverall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------
    print("\n📊 SEASON CONSISTENCY SUMMARY")
    print(player_summary["category"].value_counts())
    print(f"\nTotal players: {player_summary.shape[0]}")
    print(f"Players with gaps: {player_summary['has_gap'].sum()}")
    print(f"Players with only one season: {(player_summary['num_seasons'] == 1).sum()}")
    
    season_counts = df['season'].value_counts().sort_index()
    plt.bar(season_counts.index, season_counts.values)
    plt.xlabel('Season')
    plt.ylabel('Number of Players')
    plt.title('Player Distribution by Season')
    plt.tick_params(axis='x', rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def analyze_player_season_counts(df, max_seasons=6):
    """
    Shows:
    - Number of players with exactly N seasons
    - Number of players with non-consecutive seasons
    """

    df = df.sort_values(["id", "season"]).copy()

    # --------------------------------------------------
    # Season count per player
    # --------------------------------------------------
    season_counts = df.groupby("id")["season"].nunique()

    season_distribution = (
        season_counts
        .value_counts()
        .sort_index()
        .reindex(range(1, max_seasons + 1), fill_value=0)
    )

    # --------------------------------------------------
    # Detect non-consecutive players
    # --------------------------------------------------
    season_diff = df.groupby("id")["season"].diff()
    has_gap = season_diff > 1

    gap_players = has_gap.groupby(df["id"]).any()

    # --------------------------------------------------
    # Summary table
    # --------------------------------------------------
    summary_df = pd.DataFrame({
        "Players with exactly N seasons": season_distribution
    })
    summary_df.index.name = "Number of Seasons"

    # --------------------------------------------------
    # Plot 1 — Exact season count distribution
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    summary_df.plot(
        kind="bar",
        legend=False,
        figsize=(8, 5)
    )
    plt.title("Players by Exact Number of Seasons Played")
    plt.xlabel("Number of Seasons")
    plt.ylabel("Number of Players")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Plot 2 — Consecutive vs Non-Consecutive
    # --------------------------------------------------
    plt.figure(figsize=(6, 5))
    gap_counts = gap_players.value_counts()

    plt.pie(
        gap_counts.values,
        labels=["Fully consecutive", "Has gaps"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Players with vs without Season Gaps")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Print stats (thesis-friendly)
    # --------------------------------------------------
    print("\n📊 PLAYER SEASON AVAILABILITY")
    print(summary_df)

    print("\n📊 GAP STATISTICS")
    print(f"Total players: {season_counts.shape[0]}")
    print(f"Players with ≥2 seasons: {(season_counts >= 2).sum()}")
    print(f"Players with gaps: {gap_players.sum()}")
    print(f"Fully consecutive players: {(~gap_players).sum()}")

    return summary_df, gap_players



def create_performance_features(df):
    # Goals vs xG (overperformance/underperformance)
    if 'goals' in df.columns and 'xG' in df.columns:
        df['feat_goals_vs_xG'] = df['goals'] - df['xG']
        df['feat_goals_xG_ratio'] = df['goals'] / (df['xG'] + 0.01)  # Avoid division by zero

        # Classify performance
        df['feat_is_overperforming'] = (df['feat_goals_vs_xG'] > 0).astype(int)
        df['feat_is_underperforming'] = (df['feat_goals_vs_xG'] < 0).astype(int)

        print("   - Created xG features with performance indicators")

        # Plot Goals vs xG
        valid = df[['goals', 'xG']].dropna()

        plt.figure(figsize=(8, 5))
        plt.scatter(valid['xG'], valid['goals'], alpha=0.3, s=10)
        max_val = max(valid['xG'].max(), valid['goals'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect match')
        plt.xlabel('Expected Goals (xG)')
        plt.ylabel('Actual Goals')
        plt.title('Goals vs xG', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df


