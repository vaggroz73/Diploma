import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Excel file and strip whitespace from column names
df = pd.read_excel("FIFA_2017-18_merged.xlsx")
df.columns = df.columns.str.strip()

# 2. Remove duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]

# 3. Keep only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# 4. Drop rows with missing 'potential' values
clean_df = numeric_df.dropna(subset=['potential'])

# 5. Define likely FBref stat columns to analyze
fbref_cols = [col for col in ['time', 'goals', 'assists', 'shots', 'key_passes', 'xG', 'xA'] if col in clean_df.columns]

# 6. Keep only rows without missing FBref stats or 'potential'
clean_df = clean_df.dropna(subset=fbref_cols + ['potential'])

# 7. Ensure all relevant columns are numeric
clean_df.loc[:, fbref_cols + ['potential']] = clean_df[fbref_cols + ['potential']].apply(pd.to_numeric, errors='coerce')

# 8. Calculate correlation between FBref stats and 'potential'
fbref_correlation = clean_df[fbref_cols + ['potential']].corr()['potential'].sort_values(ascending=False)

# 9. Print correlation results
print("Correlation between potential and FBref stats:")
print(fbref_correlation)

# 10. Visualize with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(clean_df[fbref_cols + ['potential']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: FBref Stats vs Potential")
plt.tight_layout()
plt.show()
