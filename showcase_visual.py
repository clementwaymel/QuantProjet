import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

print("--- GÉNÉRATION DU VISUEL SHOWCASE (LINKEDIN) ---")

# 1. TÉLÉCHARGEMENT DES DONNÉES (Coca vs Pepsi)
tickers = ['KO', 'PEP']
print("Téléchargement des données...")
data = yf.download(tickers, start="2022-01-01", end="2024-01-01", auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data = data['Close']

data = data.dropna()

# 2. CALCULS MATHÉMATIQUES (Simulation Rolling OLS)
window = 30
rolling_beta = []
rolling_spread = []
z_scores = []

# On simule la boucle mathématique
Y = data['KO'].values
X = data['PEP'].values
dates = data.index

history_X = []
history_Y = []

for i in range(len(data)):
    history_X.append(X[i])
    history_Y.append(Y[i])
    
    if len(history_X) > window:
        history_X.pop(0)
        history_Y.pop(0)
    
    if len(history_X) == window:
        # Calcul OLS (Régression)
        slope, intercept, _, _, _ = linregress(history_X, history_Y)
        
        # Beta dynamique
        beta = slope
        
        # Spread (Résidu)
        spread = Y[i] - (beta * X[i])
        
        rolling_beta.append(beta)
        rolling_spread.append(spread)
    else:
        rolling_beta.append(np.nan)
        rolling_spread.append(np.nan)

# Calcul Z-Score sur le spread
spread_series = pd.Series(rolling_spread)
rolling_mean = spread_series.rolling(window=30).mean()
rolling_std = spread_series.rolling(window=30).std()
z_score_series = (spread_series - rolling_mean) / rolling_std

# 3. CRÉATION DU DASHBOARD "INGÉNIEUR"
plt.style.use('seaborn-v0_8-darkgrid') # Style pro
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1.5])

# Panel 1 : Les Prix (La Corrélation visible)
ax1 = plt.subplot(gs[0])
ax1.plot(dates, data['KO'], label='Coca-Cola (KO)', color='#e41a1c', linewidth=2)
ax1.set_ylabel('Prix KO ($)', color='#e41a1c', fontweight='bold')
ax1.set_title('QuantEngine : Analyse de Paires & Arbitrage Statistique', fontsize=16, fontweight='bold')

ax1_twin = ax1.twinx() # Axe Y secondaire
ax1_twin.plot(dates, data['PEP'], label='PepsiCo (PEP)', color='#377eb8', linewidth=2, linestyle='--')
ax1_twin.set_ylabel('Prix PEP ($)', color='#377eb8', fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Panel 2 : Le Modèle Mathématique (Beta Dynamique)
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(dates, rolling_beta, color='purple', label='Hedge Ratio Dynamique (Rolling OLS)', linewidth=1.5)
ax2.set_ylabel('Beta (β)', fontweight='bold')
ax2.fill_between(dates, rolling_beta, np.nanmean(rolling_beta), alpha=0.1, color='purple')
ax2.legend()
ax2.set_title('Adaptation du Modèle : Évolution du ratio de couverture', fontsize=10)

# Panel 3 : Le Signal de Trading (Z-Score & Mean Reversion)
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.plot(dates, z_score_series, color='black', linewidth=1, label='Z-Score du Spread')
ax3.axhline(0, color='grey', linestyle='-', alpha=0.5)

# Seuils
ax3.axhline(2.0, color='red', linestyle='--', alpha=0.8, label='Seuil Vente (+2σ)')
ax3.axhline(-2.0, color='green', linestyle='--', alpha=0.8, label='Seuil Achat (-2σ)')

# Coloriage des zones d'opportunité
ax3.fill_between(dates, z_score_series, 2.0, where=(z_score_series >= 2.0), color='red', alpha=0.3, interpolate=True)
ax3.fill_between(dates, z_score_series, -2.0, where=(z_score_series <= -2.0), color='green', alpha=0.3, interpolate=True)

ax3.set_ylabel('Z-Score ($\sigma$)', fontweight='bold')
ax3.set_title('Génération de Signal : Détection d\'anomalies statistiques', fontsize=10)
ax3.legend(loc='lower right')

plt.tight_layout()
print("Affichage du graphique...")
plt.show()