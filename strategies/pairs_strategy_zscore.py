from financial_librairy import FinancialAsset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("--- STRATÉGIE MEAN REVERSION (Z-SCORE) ---")

# 1. CHARGEMENT
# On reprend nos deux actifs cointégrés
asset_a = FinancialAsset("KO", "2020-01-01", "2024-01-01")
asset_b = FinancialAsset("PEP", "2020-01-01", "2024-01-01")
asset_a.download_data()
asset_b.download_data()

# DataFrame aligné
df = pd.DataFrame(index=asset_a.data.index)
df['KO'] = asset_a.data
df['PEP'] = asset_b.data
df.dropna(inplace=True)

# 2. CALCUL DU HEDGE RATIO (STATIQUE POUR L'INSTANT)
# On calcule le ratio sur la première moitié des données (In-Sample) pour être rigoureux
# Mais pour simplifier ici, on prend tout l'historique pour trouver le bêta
x = sm.add_constant(df['PEP'])
model = sm.OLS(df['KO'], x).fit()
hedge_ratio = model.params['PEP']
print(f"Hedge Ratio : {hedge_ratio:.4f}")

# 3. CONSTRUCTION DU Z-SCORE DYNAMIQUE
# Le Spread brut
df['Spread'] = df['KO'] - (hedge_ratio * df['PEP'])

# Moyenne et Ecart-Type GLISSANTS (Rolling Window)
# On regarde les 30 derniers jours pour définir ce qui est "normal"
window = 30
df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
df['Spread_Std'] = df['Spread'].rolling(window=window).std()

# Le Z-Score
df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']

# On supprime les NaN du début (les 30 premiers jours)
df.dropna(inplace=True)

# 4. GÉNÉRATION DES SIGNAUX
# Seuil d'entrée (Entry Threshold) : 2 écarts-types
entry_threshold = 2.0
# Seuil de sortie (Exit Threshold) : Retour à la moyenne (0) ou proche (0.5)
exit_threshold = 0.5

# Signaux (Logique booléenne)
# Long Spread (Achat) : Z < -2
signals_long = (df['Z_Score'] < -entry_threshold)
# Short Spread (Vente) : Z > +2
signals_short = (df['Z_Score'] > entry_threshold)
# Exit (Sortie) : -0.5 < Z < 0.5
signals_exit = (df['Z_Score'].abs() < exit_threshold)

# 5. VISUALISATION INGÉNIEUR
fig, ax = plt.subplots(figsize=(14, 7))

# La courbe du Z-Score
ax.plot(df.index, df['Z_Score'], label='Z-Score', color='blue', alpha=0.6)

# Les Lignes de seuil
ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.axhline(entry_threshold, color='red', linestyle='--', label='Seuil Vente (+2)')
ax.axhline(-entry_threshold, color='green', linestyle='--', label='Seuil Achat (-2)')

# --- MARQUAGE DES ENTRÉES ---
# On ne veut afficher un point que quand on ENTRE en position, pas tous les jours
# Astuce : On prend les endroits où le signal passe de Faux à Vrai
# Mais pour visualiser simple, on affiche juste les points extrêmes

# Points Rouges (Vente du Spread : Short KO / Long PEP)
ax.scatter(df.index[signals_short], df['Z_Score'][signals_short], 
           color='red', marker='v', s=30, label='Zone Vente')

# Points Verts (Achat du Spread : Long KO / Short PEP)
ax.scatter(df.index[signals_long], df['Z_Score'][signals_long], 
           color='green', marker='^', s=30, label='Zone Achat')

ax.set_title(f"Signaux de Trading sur Z-Score (Window={window})")
ax.set_ylabel("Nombre d'Écarts-Types (Sigma)")
ax.legend(loc='upper left')
ax.grid(True)

plt.show()