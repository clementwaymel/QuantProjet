import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration affichage
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("--- GÉNÉRATEUR DE FRONTIÈRE EFFICIENTE (MARKOWITZ) ---")

# ==========================================
# 1. ACQUISITION DES DONNÉES (Multi-Assets)
# ==========================================
tickers = ['AAPL', 'TLT', 'GLD']
print(f"[1/4] Téléchargement des données pour {tickers}...")

data = yf.download(tickers, start="2020-01-01", end="2024-01-01", auto_adjust=True, progress=False)

# --- Correction Structure Données (Bug Windows/Yfinance) ---
# Si on a un MultiIndex (ex: Price -> Ticker), on garde juste les Close
if isinstance(data.columns, pd.MultiIndex):
    # On essaie d'accéder au niveau 'Close' s'il existe, sinon on prend tout
    try:
        data = data['Close']
    except KeyError:
        # Si 'Close' n'est pas un niveau, on suppose que c'est déjà bon ou on nettoie autrement
        pass

# Calcul des rendements quotidiens (Log-returns préférables pour l'agrégation, mais restons simples)
returns = data.pct_change().dropna()

# ==========================================
# 2. CALCULS STATISTIQUES ANNUELS
# ==========================================
print("[2/4] Calcul de la Matrice de Covariance...")

# Moyenne des rendements annualisée
mean_returns = returns.mean() * 252

# Matrice de Covariance annualisée
cov_matrix = returns.cov() * 252

print("\nMatrice de Corrélation (Pour voir les liens entre actifs) :")
print(returns.corr())
print("-" * 30)

# ==========================================
# 3. SIMULATION MONTE CARLO (L'Expérience)
# ==========================================
num_portfolios = 10000  # On va tester 10 000 portefeuilles aléatoires
print(f"[3/4] Simulation de {num_portfolios} portefeuilles...")

# Tableaux pour stocker les résultats
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    # A. Création de poids aléatoires
    weights = np.array(np.random.random(len(tickers)))
    # Normalisation pour que la somme fasse 1 (100%)
    weights = weights / np.sum(weights)
    
    # On sauvegarde les poids
    all_weights[i, :] = weights
    
    # B. Rendement attendu du portefeuille
    # Produit scalaire : Poids * Rendements Moyens
    ret_arr[i] = np.sum(mean_returns * weights)
    
    # C. Volatilité (Risque) du portefeuille (Formule Matricielle de Markowitz)
    # Volatilité = Racine carrée de (w.T * Sigma * w)
    var_portfolio = np.dot(weights.T, np.dot(cov_matrix, weights))
    vol_arr[i] = np.sqrt(var_portfolio)
    
    # D. Ratio de Sharpe (Rendement / Risque)
    # On suppose taux sans risque = 0
    sharpe_arr[i] = ret_arr[i] / vol_arr[i]

# ==========================================
# 4. RECHERCHE DU MEILLEUR PORTEFEUILLE
# ==========================================
print("[4/4] Analyse des résultats...")

# Trouver l'index du portefeuille avec le Sharpe le plus élevé
max_sharpe_idx = sharpe_arr.argmax()

max_sharpe_ret = ret_arr[max_sharpe_idx]
max_sharpe_vol = vol_arr[max_sharpe_idx]
max_sharpe_sr = sharpe_arr[max_sharpe_idx]
best_weights = all_weights[max_sharpe_idx, :]

print("\n--- RÉSULTAT OPTIMAL (Tangency Portfolio) ---")
print(f"Rendement Annuel Espéré : {max_sharpe_ret:.2%}")
print(f"Volatilité Annuelle     : {max_sharpe_vol:.2%}")
print(f"Ratio de Sharpe         : {max_sharpe_sr:.2f}")
print("Allocation Optimale :")
for ticker, weight in zip(tickers, best_weights):
    print(f"  - {ticker} : {weight:.2%}")

# ==========================================
# 5. VISUALISATION (La Frontière Efficiente)
# ==========================================
plt.figure(figsize=(12, 8))

# Nuage de points (Tous les portefeuilles testés)
# c=sharpe_arr permet de colorer selon la performance
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', s=10, alpha=0.5)
plt.colorbar(label='Ratio de Sharpe')

# Point Rouge : Le Meilleur Portefeuille
plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', s=100, edgecolors='black', label='Max Sharpe (Optimal)')

# Mise en forme
plt.title('Frontière Efficiente de Markowitz (AAPL + TLT + GLD)')
plt.xlabel('Risque (Volatilité Annuelle)')
plt.ylabel('Rendement Espéré (Annuel)')
plt.legend()
plt.grid(True)
plt.show()