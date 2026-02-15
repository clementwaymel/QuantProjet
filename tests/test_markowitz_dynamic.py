import pandas as pd
import matplotlib.pyplot as plt
from core.financial_library import FinancialAsset
from core.optimization import MarkowitzOptimizer

print("--- TEST OPTIMISATION DYNAMIQUE (MARKOWITZ) ---")

# 1. Chargement des données (4 Actifs différents)
tickers = ['AAPL', 'TLT', 'GLD', 'MSFT']
assets = {}
df_prices = pd.DataFrame()

print("Chargement des historiques...")
for t in tickers:
    asset = FinancialAsset(t, "2020-01-01", "2024-01-01")
    # On suppose que la base est remplie (sinon lancer seed_data.py)
    asset.download_data() 
    if asset.data is not None:
        df_prices[t] = asset.data

df_prices.dropna(inplace=True)

# 2. Initialisation de l'Optimiseur
optimizer = MarkowitzOptimizer(lookback_days=60)

# 3. Simulation du Rééquilibrage (Rebalancing)
# On va recalculer les poids tous les 20 jours ouvrés (~1 mois)
rebalance_freq = 20
history_weights = []
dates = []

print("Lancement de la simulation d'allocation...")

for i in range(60, len(df_prices), rebalance_freq):
    # Date actuelle
    current_date = df_prices.index[i]
    
    # Données disponibles jusqu'à aujourd'hui
    # On coupe le DataFrame pour ne pas voir le futur !
    known_data = df_prices.iloc[:i]
    
    # L'IA calcule les poids optimaux
    weights = optimizer.get_optimal_weights(known_data)
    
    # On stocke pour le graphique
    weights['Date'] = current_date
    history_weights.append(weights)
    dates.append(current_date)
    
    # Affichage Log
    print(f"[{current_date.date()}] Allocation Optimale : {weights}")

# 4. Visualisation de l'Évolution des Poids
df_weights = pd.DataFrame(history_weights).set_index('Date')

plt.figure(figsize=(12, 6))
plt.stackplot(df_weights.index, df_weights.T, labels=df_weights.columns, alpha=0.8)
plt.legend(loc='upper left')
plt.title('Évolution Dynamique de l\'Allocation (Minimum Variance)')
plt.ylabel('Poids du Portefeuille (0 à 1)')
plt.margins(0, 0)
plt.show()