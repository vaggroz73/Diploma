import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_excel("C:/Users/evagg/Desktop/Final.xlsx")

df = df[df['Pos'] != "GK"].copy()

def map_position(Pos):
    defenders = ["DF", "DF,FW", "DF,MF"]
    midfielders = ["MF,DF", "MF", "MF,FW"]
    attackers = ["FW", "FW,DF", "FW,MF"]
    #strikers = ["S"]

    if Pos in defenders:
        return "Defender"
    elif Pos in midfielders:
        return "Midfielder"
    elif Pos in attackers:
        return "Attacker"
    #elif Pos in strikers:
     #   return "Striker"
    else:
        return "Other"


df["role"] = df["Pos"].apply(map_position)
df = df[df["role"] != "Other"].copy()



#data.info()

#print("Unique players: ",  data["id"].nunique() )
#print("Τελικό μέγεθος:", data.shape)

understat_cols = ["overall", "potential", "time", "goals", "xG", "xA", "assists", "shots", "key_passes", 
                  "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup"]



fbref_cols = ["overall", "potential", "Mins_Per_90", "Touches_Touches", "Def Pen_Touches", "Def 3rd_Touches",
              "Mid 3rd_Touches", "Att 3rd_Touches", "Att Pen_Touches", "Live_Touches",
              "Att_Take", "Succ_Take", "Succ_percent_Take", "Tkld_Take", "Carries_Carries",
              "TotDist_Carries", "PrgDist_Carries", "PrgC_Carries", "Final_Third_Carries",
              "CPA_Carries", "Mis_Carries", "Dis_Carries", "Rec_Receiving", "PrgR_Receiving", 
              
              
              
              "Player", "Squad", "Comp", "Season_End_Year", "Nation", "Pos", "Age"]



fifa_physical = ["overall","pace", "physic", "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_reactions",
                 "movement_balance", "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots"]


             
fifa_skills = ["overall","shooting", "passing", "dribbling", "defending", "attacking_crossing", "attacking_finishing", 
               "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys", "skill_dribbling",
               "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control", "defending_standing_tackle", 
               "defending_sliding_tackle"]



fifa_mental = ["mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision",
               "mentality_penalties", "mentality_composure", "defending_marking_awareness", "value_eur", "wage_eur", 
               "age", "height_cm", "weight_kg", "overall"]


fifa_cols = ["pace", "physic", "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_reactions",
            "movement_balance", "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
            "shooting", "passing", "dribbling", "defending", "attacking_crossing", "attacking_finishing", 
            "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys", "skill_dribbling",
            "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control", "defending_standing_tackle", 
            "defending_sliding_tackle","mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision",
            "mentality_penalties", "mentality_composure", "defending_marking_awareness", "value_eur", "wage_eur", 
            "age", "height_cm", "weight_kg","overall"]


econ = ["value_eur", "wage_eur", "overall"]

"""
df_role = df[df["role"] == "Attacker"].copy()
print(f"Training model for role: {'Attacker'}")
"""

df_fifa = df[fifa_cols].dropna()

# Compute correlation matrix
co_mtx = df_fifa.corr(numeric_only=True)

# Best correlated attributes with Overall
overall_corr = (
    co_mtx['overall']
    .sort_values(key=abs, ascending=False)
    .drop('overall')
)

print("Attributes most correlated with Overall")
print(overall_corr.head(30))

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(co_mtx, square=True, linewidths=.8,cmap="YlGnBu", annot=True)
    
# Εύρεση ζευγαριών πάνω από το threshold

mask = np.triu(np.ones_like(co_mtx, dtype=bool), k=1)

high_corr_pairs = []

threshold=0.8

for i in range(len(co_mtx.columns)):
    for j in range(i+1, len(co_mtx.columns)):
        if mask[i, j] and abs(co_mtx.iloc[i, j]) > threshold:
            high_corr_pairs.append({
                'feature_1': co_mtx.columns[i],
                'feature_2': co_mtx.columns[j],
                'correlation': co_mtx.iloc[i, j]
            })

# Sort by absolute correlation
high_corr_pairs = sorted(high_corr_pairs, 
                        key=lambda x: abs(x['correlation']), 
                        reverse=True)

for pair in high_corr_pairs[:15]:  # Show top 15
    print(f"\n  {pair['feature_1']} ↔ {pair['feature_2']}")
    print(f"  Correlation: {pair['correlation']:.3f}")
