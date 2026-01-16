from core.financial_library import FinancialAsset

print("--- LABORATOIRE D'ANALYSE STATISTIQUE ---")

# 1. Analyse d'une Action (Apple)
aapl = FinancialAsset("AAPL", "2020-01-01", "2024-01-01")
aapl.download_data() # Ça va charger depuis le cache SQL instantanément !
aapl.diagnose()

# 2. Analyse d'une Valeur Refuge (Obligations TLT)
tlt = FinancialAsset("TLT", "2020-01-01", "2024-01-01")
tlt.download_data()
tlt.diagnose()

# (Optionnel) Si tu veux voir un truc fou, essaie "BTC-USD" (Bitcoin)
btc = FinancialAsset("BTC-USD", "2020-01-01", "2024-01-01")
btc.download_data()
btc.diagnose()