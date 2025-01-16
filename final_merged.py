import pandas as pd
from rapidfuzz import fuzz, process

# Load the two Excel files
file1 = "big5_player_2024.xlsx"  # Replace with your first file name
file2 = "players_la_liga_2023.xlsx"  # Replace with your second file name

# Load data into pandas DataFrames
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Columns containing names
name_col1 = "Player"  # Column name in the first file
name_col2 = "player_name"  # Column name in the second file

# Columns to keep from both files
# If you want all columns, skip this step
columns_to_keep_df1 = df1.columns
columns_to_keep_df2 = df2.columns


# Results list
merged_rows = []

# Iterate through the names in df1 and match with df2
for index1, row1 in df1.iterrows():
    name1 = row1[name_col1]
    # Find the best match in df2 for the current name in df1
    match, score, index2 = process.extractOne(name1, df2[name_col2], scorer=fuzz.ratio)
    
    if score >= 80:  # Check if the similarity score is 90% or higher
        merged_row = {**row1.to_dict(), **df2.loc[index2].to_dict()}
        merged_rows.append(merged_row)
        
        

# Create a DataFrame with the merged rows
merged_df = pd.DataFrame(merged_rows)

# Save the merged DataFrame to an Excel file
output_file = "la_liga_2023-24.xlsx"
merged_df.to_excel(output_file, index=False)

print(f"Merged file saved as {output_file}")
