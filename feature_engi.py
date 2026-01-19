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

print("=" * 80)
print("DATA LOADING & BASIC PROFILING")
print("=" * 80)

print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"📅 Date Range: {df['Season_End_Year'].min()} - {df['Season_End_Year'].max()}")
print(f"👥 Unique Players: {df['id'].nunique():,}")
print(f"⚽ Unique Teams: {df['club_name'].nunique():,}")

print("\n" + "─" * 80)
print("Data Types Summary:")
print(df.dtypes.value_counts())

print("\n" + "─" * 80)
print("Target Variable (Overall) Statistics:")
print(df['overall'].describe())


def analyze_temporal_coverage(df):
    """Analyze temporal aspects of the dataset"""
    print("\n" + "=" * 80)
    print("TEMPORAL COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Players per season
    players_per_season = df.groupby('Season_End_Year')['id'].nunique()
    print("\n📅 Players per Season:")
    print(players_per_season)
    
    # Seasons per player
    seasons_per_player = df.groupby('id').size()
    print("\n👤 Seasons per Player Distribution:")
    print(seasons_per_player.describe())
    
    # Career length analysis
    player_careers = df.groupby('id').agg({
        'Season_End_Year': ['min', 'max', 'count']
    })
    player_careers.columns = ['First_Season', 'Last_Season', 'Total_Seasons']
    player_careers['Career_Span'] = player_careers['Last_Season'] - player_careers['First_Season'] + 1
    
    print("\n⏱️ Career Span Analysis:")
    print(player_careers['Career_Span'].value_counts().sort_index())
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Players per season
    axes[0, 0].bar(players_per_season.index, players_per_season.values, color='steelblue')
    axes[0, 0].set_xlabel('Season')
    axes[0, 0].set_ylabel('Number of Players')
    axes[0, 0].set_title('Players per Season', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Seasons per player histogram
    axes[0, 1].hist(seasons_per_player, bins=range(1, seasons_per_player.max()+2), 
                    color='teal', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Seasons')
    axes[0, 1].set_ylabel('Number of Players')
    axes[0, 1].set_title('Distribution of Seasons per Player', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Career span distribution
    axes[1, 0].hist(player_careers['Career_Span'], bins=range(1, 10), 
                    color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Career Span (years)')
    axes[1, 0].set_ylabel('Number of Players')
    axes[1, 0].set_title('Career Span Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Sequence completeness
    complete_sequences = (player_careers['Career_Span'] == player_careers['Total_Seasons']).sum()
    incomplete_sequences = len(player_careers) - complete_sequences
    axes[1, 1].pie([complete_sequences, incomplete_sequences], 
                   labels=['Complete', 'With Gaps'],
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'],
                   startangle=90)
    axes[1, 1].set_title('Career Sequence Completeness', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return player_careers


def analyze_target_variable(df):
    
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS (OVERALL)")
    print("=" * 80)

    # Overall statistics by season
    overall_by_season = df.groupby('Season_End_Year')['overall'].agg(['mean', 'std', 'min', 'max'])
    print("\n📊 Overall Statistics by Season:")
    print(overall_by_season.round(2))

    # Overall by position
    if 'Pos' in df.columns:
        overall_by_position = df.groupby('Pos')['overall'].agg(['mean', 'std', 'count'])
        print("\n⚽ Overall Statistics by Position:")
        print(overall_by_position.round(2))

    # Overall distribution Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['overall'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(df['overall'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["overall"].mean():.1f}')
    plt.xlabel('Overall Rating')
    plt.ylabel('Frequency')
    plt.title('Overall Rating Distribution', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Overall by season (boxplot)
    plt.figure(figsize=(10, 5))
    df.boxplot(column='overall', by='Season_End_Year')
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
    plt.scatter(df['age'], df['overall'], alpha=0.3, s=10, color='purple')

    z = np.polyfit(df['age'].dropna(), df.loc[df['age'].notna(), 'overall'], 2)
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
        sns.violinplot(data=df_pos, x='Pos', y='overall')
        plt.xlabel('Position')
        plt.ylabel('Overall Rating')
        plt.title('Overall Distribution by Position (Top 10)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.show()

    # Q-Q plot
    plt.figure(figsize=(6, 6))
    stats.probplot(df['overall'].dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

        
        
def analyze_temporal_patterns(df):
    print("\n" + "=" * 80)
    print("5. TEMPORAL PATTERNS IN OVERALL")
    print("=" * 80)


    df_sorted = df.sort_values(['id', 'Season_End_Year'])

    # Calculate year-over-year changes
    df_sorted['overall_change'] = df_sorted.groupby('id')['overall'].diff()
    df_sorted['overall_change_pct'] = (
        df_sorted['overall_change'] / df_sorted.groupby('id')['overall'].shift(1)
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
        player_data = df_sorted[df_sorted['id'] == id].sort_values('Season_End_Year')
        plt.plot(
            player_data['Season_End_Year'],
            player_data['overall'],
            marker='o', alpha=0.6, linewidth=1
        )

    plt.xlabel('Season')
    plt.ylabel('Overall Rating')
    plt.title('Sample Career Trajectories (15 players)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Lag plot (overall_t vs overall_t-1)
    df_sorted['overall_lag1'] = df_sorted.groupby('id')['overall'].shift(1)
    valid_lag = df_sorted[['overall', 'overall_lag1']].dropna()

    plt.figure(figsize=(7, 7))
    plt.scatter(valid_lag['overall_lag1'], valid_lag['overall'],
                alpha=0.3, s=10, color='teal')

    min_val = min(valid_lag.min())
    max_val = max(valid_lag.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    corr = valid_lag['overall'].corr(valid_lag['overall_lag1'])
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
    
    age_overall = df.groupby('age')['overall'].agg(['mean', 'std', 'count'])
    print("\nAverage Overall by Age:")
    print(age_overall.round(2))
    
    # Playing time analysis
    if 'Mins_Per_90' in df.columns:
        print("\n⏱️ PLAYING TIME ANALYSIS:")
        print(df['Mins_Per_90'].describe())
        
        # Correlation with overall
        corr = df[['Mins_Per_90', 'overall']].corr().iloc[0, 1]
        print(f"\nCorrelation (Playing Time vs Overall): {corr:.3f}")
    
    # Performance metrics
    if 'goals' in df.columns and 'xG' in df.columns:
        print("\n⚽ PERFORMANCE METRICS:")
        print(f"Total Goals: {df['goals'].sum():.0f}")
        print(f"Total xG: {df['xG'].sum():.2f}")
        print(f"Goals vs xG Ratio: {(df['goals'].sum() / df['xG'].sum()):.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Age vs Overall
    axes[0, 0].scatter(df['age'], df['overall'], alpha=0.3, s=10)
    axes[0, 0].plot(age_overall.index, age_overall['mean'], color='red', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Overall Rating')
    axes[0, 0].set_title('Age vs Overall (with Mean Trend)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Age distribution
    axes[0, 1].hist(df['age'], bins=range(16, 45), color='teal', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Age Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Value vs Overall
    if 'value_eur' in df.columns:
        valid_data = df[(df['value_eur'] > 0) & (df['value_eur'] < df['value_eur'].quantile(0.99))]
        axes[1, 0].scatter(valid_data['overall'], valid_data['value_eur']/1e6, alpha=0.3, s=10, color='green')
        axes[1, 0].set_xlabel('Overall Rating')
        axes[1, 0].set_ylabel('Market Value (millions €)')
        axes[1, 0].set_title('Overall vs Market Value', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Playing time distribution
    if 'Mins_Per_90' in df.columns:
        axes[1, 1].hist(df['Mins_Per_90'].dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Minutes per 90')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Playing Time Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    

# FEATURE IMPORTANCE (PRELIMINARY)

def preliminary_feature_importance(df, n_features=20):
    """Quick Random Forest to identify important features"""
    print("\n" + "=" * 80)
    print("8. PRELIMINARY FEATURE IMPORTANCE")
    print("=" * 80)
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['id', 'Season_End_Year', 'overall']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Remove columns with too many missing values
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
            valid_cols.append(col)
    
    print(f"\n🎯 Using {len(valid_cols)} features for importance analysis")
    
    # Prepare dataset
    X = df[valid_cols].fillna(df[valid_cols].median())
    y = df['overall']
    
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
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = importances.head(n_features)
    ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {n_features} Features by Importance (Random Forest)', 
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return importances
    




