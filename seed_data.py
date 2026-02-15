import sys
import os
sys.path.append(os.getcwd())
from core.financial_library import FinancialAsset

print("--- MISE À JOUR BASE DE DONNÉES (S&P 100) ---")

# Liste unique de tous les actifs de ta nouvelle stratégie
tickers = [
    'MA', 'WMT', 'V', 'TXN', 'UPS', 'QCOM', 'ABT', 'BKNG', 'GE', 'MS', 
    'CAT', 'UNH', 'ABBV', 'AMGN', 'TSLA', 'UNP', 'PEP', 'TJX', 'GS', 
    'SPGI', 'HD', 'PG', 'LOW', 'JNJ', 'DE', 'TMO', 'GOOGL', 'AMAT', 
    'CVX', 'EOG', 'DHR'
]

for ticker in tickers:
    print(f"\n[Seed] Vérification/Téléchargement de {ticker}...")
    try:
        # On prend large (2015-2024) pour que le scanner et le backtest soient à l'aise
        asset = FinancialAsset(ticker, "2015-01-01", "2024-01-01")
        asset.download_data(force_update=False) # False = Si on l'a déjà, on ne retélécharge pas (gain de temps)
    except Exception as e:
        print(f"❌ Erreur sur {ticker} : {e}")

print("\n--- BASE À JOUR. PRÊT POUR LE BACKTEST ---")