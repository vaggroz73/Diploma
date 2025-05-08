import pandas as pd
from rapidfuzz import fuzz, process

# Φόρτωση των excel αρχειών 
file1 = r"C:/Users/evagg/Desktop/FIFA_23.xlsx"  # FIFA dataset
file2 = r"C:/Users/evagg/Desktop/epl_2022-23.xlsx"  # FBref dataset

# Αποθήκευση σε DataFrames
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Ονόματα στηλών που θέλουμε να συγκρίνουμε
name_col1 = "short_name"  # FIFA column
name_col2 = "Player"  # FBref column

# Δημιουργία στήλης που θα περιέχει μόνο τα επώνυμα των παικτών
df1["Surname"] = df1[name_col1].apply(lambda x: x.split(". ")[-1].strip() if isinstance(x, str) else "")
df2["Surname"] = df2[name_col2].apply(lambda x: x.split()[-1].strip() if isinstance(x, str) else "")

# Δημιουργία Λίστας για την αποθήκευση των αποτελεσμάτων
merged_rows = []

# Βρόγχος που ελέγχει όλα τα επώνυμα και τις ομάδες των παικτών
for index1, row1 in df1.iterrows():
    surname1 = row1["Surname"]
    team1 = row1["club_name"]  # Ομάδα στο FIFA

    # Εύρεση πιθανών ταιριασμάτων στο αρχείο FBref με βάση το επίθετο
    possible_matches = df2[df2["Surname"] == surname1]

    # Αρχικοποίηση Μεταβλητών
    best_match = None
    best_score = 0
    best_index = None

    # Κοιτάμε το πλήρες όνομα για τους παίκτες με το ίδιο επίθετο
    for index2, row2 in possible_matches.iterrows():
        score = fuzz.WRatio(row1[name_col1], row2[name_col2])  # Σύγριση πλήρους ονόματος
        if score > best_score:  
            best_match = row2
            best_score = score
            best_index = index2

    # Αν ικανοποιούνται οι συνθήκες ενώνω τα δεδομένα
    if best_match is not None and best_score >= 70:  # Το κατώφλι με τα καλύτερα αποτλέσματα από αυτά που δοκιμάσαμε
        merged_row = {**row1.to_dict(), **df2.loc[best_index].to_dict()}
        merged_rows.append(merged_row)

#  DataFrame με τα συγχωνευμένα δεδομένα
merged_df = pd.DataFrame(merged_rows)

# Διαγραφή στήλης Surname 
merged_df.drop(columns=["Surname"], inplace=True, errors="ignore")

# Αποθηκέυουμε τα συγχωνευμένα DataFrame σε Excel
merged_df.to_excel("FIFA_2022-245_merged.xlsx", index=False)

print(f"Merged file saved")


