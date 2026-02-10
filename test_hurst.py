import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.financial_library import FinancialAsset
from core.fractals import calculate_hurst_exponent, check_mean_reversion_quality
import statsmodels.api as sm

print("--- LABORATOIRE DE PHYSIQUE DES MARCHÉS (HURST) ---")

# 1. Chargement des données
# Assure-toi d'avoir lancé seed_data.py si ta base est vide !
print("Chargement des actifs...")
asset_a = FinancialAsset("KO", "2020-01-01", "2024-01-01")
asset_b = FinancialAsset("PEP", "2020-01-01", "2024-01-01")
asset_c = FinancialAsset("AAPL", "2020-01-01", "2024-01-01")

asset_a.download_data()
asset_b.download_data()
asset_c.download_data()

# 2. Analyse d'un actif seul (Apple)
print(f"\n1. Analyse d'une action seule (AAPL) :")
# Apple est une action de croissance, elle devrait être 'Trending' (H > 0.5)
check_mean_reversion_quality(asset_c.data)

# 3. Analyse du Spread (Coca - Beta*Pepsi)
print(f"\n2. Analyse du Spread (KO / PEP) :")
# On calcule un spread rapide pour le test (Beta statique OLS)
df = pd.DataFrame({'KO': asset_a.data, 'PEP': asset_b.data}).dropna()
model = sm.OLS(df['KO'], sm.add_constant(df['PEP'])).fit()
beta = model.params['PEP']

df['Spread'] = df['KO'] - beta * df['PEP']

# Le spread devrait être 'Mean Reverting' (H < 0.5)
check_mean_reversion_quality(df['Spread'])

# 4. Visualisation de l'Exposant de Hurst glissant
# On veut voir si la "qualité" du spread se dégrade parfois
window = 100 # Fenêtre de 100 jours
df['Rolling_Hurst'] = df['Spread'].rolling(window=window).apply(lambda x: calculate_hurst_exponent(x))

plt.figure(figsize=(12, 8))

# Graphique du Spread (Le Prix synthétique)
plt.subplot(2, 1, 1)
plt.plot(df['Spread'])
plt.title(f"Spread Coca-Pepsi (Beta={beta:.2f})")
plt.grid(True)

# Graphique de l'Exposant de Hurst (La métrique de stabilité)
plt.subplot(2, 1, 2)
plt.plot(df['Rolling_Hurst'], color='purple', label='Exposant de Hurst')
plt.axhline(0.5, color='red', linestyle='--', label='Frontière Aléatoire (0.5)')

# Zone Rouge (DANGER) : Quand H > 0.5, le spread part en tendance (divergence)
plt.fill_between(df.index, df['Rolling_Hurst'], 0.5, where=(df['Rolling_Hurst'] > 0.5), color='red', alpha=0.3, label="DANGER (Trending)")

# Zone Verte (OK) : Quand H < 0.5, le spread oscille (convergence)
plt.fill_between(df.index, df['Rolling_Hurst'], 0.5, where=(df['Rolling_Hurst'] < 0.5), color='green', alpha=0.3, label="OK (Mean Reverting)")

plt.title(f"Stabilité Temporelle : Exposant de Hurst Glissant ({window} jours)")
plt.legend()
plt.grid(True)

plt.tight_layout()
print("Affichage des graphiques...")
plt.show()