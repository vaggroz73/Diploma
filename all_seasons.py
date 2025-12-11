import pandas as pd
import glob
import os

# 1️⃣ Ο φάκελος που περιέχει τα Excel αρχεία σου
path = r"C:\Users\evagg\Desktop\SA"  # <-- άλλαξέ το αν χρειάζεται

# 2️⃣ Βρες όλα τα αρχεία που ξεκινούν με fifa_players_ και τελειώνουν σε .xlsx
files = glob.glob(os.path.join(path, "FIFA_FBref_merged_multiple_sa_*.xlsx"))

if not files:
    raise FileNotFoundError("⚠️ Δεν βρέθηκαν Excel αρχεία! Έλεγξε το path και τα ονόματα αρχείων.")

# 3️⃣ Διάβασε και ενώσε όλα τα DataFrames
dfs = []

for f in files:
    print(f"📄 Διαβάζω το αρχείο: {os.path.basename(f)}")
    df = pd.read_excel(f)

    # Έλεγξε ότι υπάρχουν οι στήλες που χρειάζεσαι
    if "long_name" not in df.columns or "Season_End_Year" not in df.columns:
        raise ValueError(f"❌ Το αρχείο {os.path.basename(f)} δεν περιέχει τις στήλες 'long_name' και 'Season_End_Year'")

    dfs.append(df)

# 4️⃣ Συνένωση όλων των DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# 5️⃣ Ταξινόμηση κατά παίκτη και έτος
combined_df = combined_df.sort_values(by=["long_name", "Season_End_Year"]).reset_index(drop=True)

# 6️⃣ Αποθήκευση του ενιαίου dataset
combined_df.to_excel("fifa_players_all_seasons_sa.xlsx", index=False)

print("✅ Συνένωση ολοκληρώθηκε επιτυχώς!")
print("📊 Τελικό μέγεθος:", combined_df.shape)
print("👥 Μοναδικοί παίκτες:", combined_df["long_name"].nunique())
print("🗓️ Εύρος ετών:", combined_df["Season_End_Year"].min(), "-", combined_df["Season_End_Year"].max())
