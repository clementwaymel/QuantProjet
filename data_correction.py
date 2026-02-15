
import yfinance as yf
import pandas as pd
import os

# 1. La liste EXACTE de ton Golden Portfolio
pairs_to_trade = [
    ('GOOGL', 'MSFT'),
    ('MSFT', 'ADBE'),
    ('CAT', 'DE'),
    ('TXN', 'ADI'),
    ('GS', 'MS'),
    ('LOW', 'HD'),
    ('V', 'MA'),
    ('HON', 'MMM'),
    ('CL', 'COST'),
    ('TGT', 'WMT')
]

# 2. Extraction des tickers uniques
unique_tickers = list(set([p[0] for p in pairs_to_trade] + [p[1] for p in pairs_to_trade]))

print(f"--- STANDARDISATION DES DONNÉES ({len(unique_tickers)} Tickers) ---")
print("Objectif : Créer des fichiers CSV avec une colonne unique nommée 'Close'")

if not os.path.exists('data'):
    os.makedirs('data')

for ticker in unique_tickers:
    print(f"Traitement de {ticker}...", end=" ")
    try:
        # Téléchargement brute
        df = yf.download(ticker, start="2015-01-01", end="2024-06-01", progress=False, multi_level_index=False)
        
        # Nettoyage agressif
        target_col = None
        
        # On cherche la meilleure colonne disponible
        if 'Adj Close' in df.columns:
            target_col = df['Adj Close']
        elif 'Close' in df.columns:
            target_col = df['Close']
        else:
            # Cas désespéré : on prend la première colonne (souvent le prix)
            target_col = df.iloc[:, 0]
            
        # On nettoie la Série pour enlever les métadonnées bizarres
        clean_series = target_col.copy()
        clean_series.name = "Close" # On force le nom "Close"
        
        # On supprime les lignes vides
        clean_series = clean_series.dropna()
        
        # Sauvegarde
        if not clean_series.empty:
            clean_series.to_csv(f"data/{ticker}.csv", header=True)
            print("OK (Standardisé) ✅")
        else:
            print("VIDE ⚠️")
            
    except Exception as e:
        print(f"ERREUR ❌ ({e})")

print("\n>>> Données nettoyées. Tu peux lancer le Backtest !")

