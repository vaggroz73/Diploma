import pandas as pd

#  Φόρτωση του dataset
df = pd.read_excel("FIFA_2022-23_merged.xlsx")


# Δημιουργία λίστας με τις θέσεις των παικτών
df['Position_List'] = df['player_positions'].str.split(',').apply(lambda x: [pos.strip() for pos in x])

# One Hot Encoding
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
positions_encoded = pd.DataFrame(mlb.fit_transform(df['Position_List']),
                                 columns=mlb.classes_,
                                 index=df.index)

# Οι θέσεις συνδιάζονται με το αρχικό dataframe
df_encoded = pd.concat([df.drop(['player_positions', 'Position_List'], axis=1), positions_encoded], axis=1)

# Αποθήκευση στο τελικό excel αρχείο
df_encoded.to_excel("fifa_players_23_encoded.xlsx", index=False)

