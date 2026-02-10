import sys
import os

# Astuce d'ingénieur : On ajoute le dossier courant au chemin Python
# pour éviter les erreurs "ModuleNotFoundError: No module named 'core'"
sys.path.append(os.getcwd())

from core.financial_library import FinancialAsset

print("--- INITIALISATION DES DONNÉES (DATA SEEDING) ---")

# Liste des actifs nécessaires pour le backtest
tickers = ['KO', 'PEP', 'AAPL', 'TLT', 'GLD', 'MSFT', 'AMZN', 'GOOGL'] 

for ticker in tickers:
    print(f"\n[Seed] Traitement de {ticker}...")
    try:
        # On télécharge du 01/01/2018 au 01/01/2024 pour avoir un bon historique
        asset = FinancialAsset(ticker, "2018-01-01", "2024-01-01")
        
        # force_update=True oblige le système à écraser/créer la donnée SQL
        asset.download_data(force_update=True)
        print(f"✅ {ticker} sauvegardé avec succès.")
        
    except Exception as e:
        print(f"❌ Erreur sur {ticker} : {e}")

print("\n--- BASE DE DONNÉES PRÊTE ---")