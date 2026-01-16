from financial_librairy import FinancialAsset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

print("--- ANALYSE DE COINTÉGRATION (PAIRS TRADING) ---")

# 1. CHARGEMENT DES DONNÉES
# On prend deux concurrents historiques
asset_a = FinancialAsset("KO", "2020-01-01", "2024-01-01") # Coca-Cola
asset_b = FinancialAsset("PEP", "2020-01-01", "2024-01-01") # Pepsi

asset_a.download_data()
asset_b.download_data()

# 2. ALIGNEMENT DES DONNÉES
# On crée un DataFrame commun pour être sûr d'avoir les mêmes jours
df = pd.DataFrame(index=asset_a.data.index)
df['KO'] = asset_a.data
df['PEP'] = asset_b.data
# On ajoute le filtre des données manquantes
df.dropna(inplace=True)

# 3. ANALYSE VISUELLE (Normalisée base 100)
# Pour comparer des prix différents (KO vaut ~60$, PEP vaut ~170$), on normalise.
plt.figure(figsize=(12, 6))
plt.plot(df['KO'] / df['KO'].iloc[0] * 100, label='Coca-Cola (Base 100)')
plt.plot(df['PEP'] / df['PEP'].iloc[0] * 100, label='Pepsi (Base 100)')
plt.title("Comparaison des Prix (Normalisés)")
plt.legend()
plt.grid(True)
plt.show()

# 4. LE TEST DE COINTÉGRATION (Engle-Granger)
# H0 : Il n'y a PAS de cointégration.
# Si p-value < 0.05, on rejette H0 -> Il Y A cointégration (L'élastique existe).

print("\n[Calcul] Test de Cointégration en cours...")
# La fonction coint renvoie (t-stat, p-value, critical_values)
score, pvalue, _ = ts.coint(df['KO'], df['PEP'])

print(f"p-value de cointégration : {pvalue:.4f}")

if pvalue < 0.05:
    print(">> RÉSULTAT : Les actifs sont COINTÉGRÉS !")
    print(">> Stratégie possible : Trader l'écart (Spread).")
else:
    print(">> RÉSULTAT : Pas de cointégration significative.")
    print(">> Attention : Trader la paire est risqué (l'écart peut grandir à l'infini).")

# 5. CALCUL DU SPREAD (L'Élastique)
# Pour trouver le "Hedge Ratio" (combien de Pepsi pour 1 Coca ?), on fait une régression linéaire.
# Spread = Y - (beta * X)

x = df['PEP'] # Variable explicative
y = df['KO']  # Variable cible
x = sm.add_constant(x) # On ajoute une constante pour l'intercepte

model = sm.OLS(y, x).fit()
hedge_ratio = model.params['PEP']
print(f"\nHedge Ratio calculé : {hedge_ratio:.4f}")
print(f"(Cela signifie : Acheter 1 action KO et vendre {hedge_ratio:.2f} actions PEP)")

# Visualisation du Spread
spread = df['KO'] - (hedge_ratio * df['PEP'])

plt.figure(figsize=(12, 6))
plt.plot(spread, label='Spread (Écart)', color='purple')
plt.axhline(spread.mean(), color='black', linestyle='--')
plt.axhline(spread.mean() + 2*spread.std(), color='red', linestyle=':', label='Seuil Vente')
plt.axhline(spread.mean() - 2*spread.std(), color='green', linestyle=':', label='Seuil Achat')
plt.title(f"Le Spread (Coca - {hedge_ratio:.2f}*Pepsi)")
plt.legend()
plt.grid(True)
plt.show()

# 6. VÉRIFICATION DE STATIONNARITÉ SUR LE SPREAD
# C'est la preuve finale. Le prix de KO n'est pas stationnaire.
# Mais le SPREAD doit l'être.
print("\n--- VÉRIFICATION FINALE (ADF sur le Spread) ---")
adf_stat, adf_pvalue, _, _, _, _ = ts.adfuller(spread)
print(f"p-value ADF sur le Spread : {adf_pvalue:.4e}")
if adf_pvalue < 0.05:
    print(">> CONFIRMÉ : Le Spread est STATIONNAIRE. C'est un signal tradable.")
else:
    print(">> ÉCHEC : Le Spread dérive. Mauvaise paire.")