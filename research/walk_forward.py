import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
# Ta liste de paires à auditer (Celles qui semblaient bien)
pairs_to_test = [
    ('XOM', 'CVX'), ('JPM', 'BAC'), ('MSFT', 'ADBE'), ('KO', 'PEP'),
    ('V', 'MA'), ('TGT', 'WMT'), ('CAT', 'DE'), ('MRK', 'PFE'),
    ('GOOGL', 'MSFT'), ('HON', 'MMM'), ('LOW', 'HD'),
    ('TXN', 'ADI'), ('UNH', 'CI'), ('GS', 'MS'), ('COST', 'WMT')
]

# On définit nos époques de test
epochs = {
    "Epoch 1 (Pre-Covid)": ("2015-01-01", "2019-12-31"),
    "Epoch 2 (Covid Crash)": ("2020-01-01", "2021-12-31"),
    "Epoch 3 (Inflation)": ("2022-01-01", "2024-06-01")
}

def get_data():
    print(">>> Téléchargement des données fraîches (2015-2024)...")
    tickers = list(set([p[0] for p in pairs_to_test] + [p[1] for p in pairs_to_test]))
    # On télécharge depuis 2015
    data = yf.download(tickers, start="2015-01-01", end="2024-06-01", group_by='ticker', progress=False)
    
    clean_data = pd.DataFrame()
    for t in tickers:
        try:
            # Gestion des cas où Adj Close ou Close sont utilisés
            if isinstance(data.columns, pd.MultiIndex):
                # Structure yfinance récente
                if 'Adj Close' in data[t]:
                    series = data[t]['Adj Close']
                elif 'Close' in data[t]:
                    series = data[t]['Close']
                else:
                    continue
            else:
                series = data['Adj Close'][t]
            
            clean_data[t] = series
        except Exception as e:
            print(f"Warning: Pas de données pour {t}")
            
    return clean_data.dropna() # On supprime les jours vides

def simulate_pair(y_data, x_data, z_entry=2.0):
    """Simulation rapide vectorisée sur une période donnée"""
    if len(y_data) < 100: return 0 # Pas assez de données
    
    ratio = y_data / x_data
    # Moyenne mobile 60 jours
    mu = ratio.rolling(60).mean()
    std = ratio.rolling(60).std()
    z_score = (ratio - mu) / std
    
    pnl = 0
    position = 0
    entry_val = 0
    
    # Boucle simplifiée pour estimation PnL
    # On simule un trade de 10,000$ à chaque signal
    for i in range(60, len(z_score)):
        z = z_score.iloc[i]
        current_val = ratio.iloc[i]
        
        if pd.isna(z): continue
        
        if position == 0:
            if z > z_entry:
                position = -1 # Short Ratio
                entry_val = current_val
            elif z < -z_entry:
                position = 1 # Long Ratio
                entry_val = current_val
        
        elif position == -1: # Short
            if z < 0: # Take Profit au retour à la moyenne
                pnl += (entry_val - current_val) / entry_val * 10000
                position = 0
            elif z > 4.5: # Stop Loss
                pnl += (entry_val - current_val) / entry_val * 10000
                position = 0

        elif position == 1: # Long
            if z > 0: # Take Profit
                pnl += (current_val - entry_val) / entry_val * 10000
                position = 0
            elif z < -4.5: # Stop Loss
                pnl += (current_val - entry_val) / entry_val * 10000
                position = 0
                
    return pnl

def run_matrix():
    df_data = get_data()
    results = []
    
    print("\n>>> Calcul de la matrice de robustesse...")
    
    for pair in pairs_to_test:
        y_sym, x_sym = pair
        row = {'Pair': f"{y_sym}-{x_sym}"}
        total_score = 0
        
        for epoch_name, (start, end) in epochs.items():
            # Découpage des données
            mask = (df_data.index >= start) & (df_data.index <= end)
            sub_data = df_data.loc[mask]
            
            if y_sym in sub_data and x_sym in sub_data:
                pnl = simulate_pair(sub_data[y_sym], sub_data[x_sym])
                row[epoch_name] = int(pnl)
                # On pénalise lourdement les périodes négatives
                if pnl > 0: total_score += 1
                else: total_score -= 1
            else:
                row[epoch_name] = 0 # Pas de données (ex: IPO récente)
        
        row['Consistency Score'] = total_score
        results.append(row)
    
    df_results = pd.DataFrame(results).set_index('Pair')
    
    # Tri par consistance (ceux qui gagnent tout le temps en haut)
    df_results = df_results.sort_values(by='Consistency Score', ascending=False)
    
    # Affichage Heatmap
    plt.figure(figsize=(12, 8))
    # On retire la colonne Score pour l'affichage couleur du PnL
    heatmap_data = df_results.drop(columns=['Consistency Score'])
    
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="RdYlGn", center=0, cbar_kws={'label': 'PnL Estime ($)'})
    plt.title("Matrice de Robustesse : PnL par Période de Marché")
    plt.tight_layout()
    plt.show()
    
    print("\n--- ANALYSE ---")
    print("Les paires en HAUT de la liste sont 'Robustes' (Gagnantes dans tous les marchés).")
    print("Les paires avec du ROUGE sont 'Instables' (Ont marché par chance sur une période).")
    print(df_results)

if __name__ == "__main__":
    run_matrix()