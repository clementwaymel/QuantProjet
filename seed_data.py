from core.financial_library import FinancialAsset

print("--- REMPLISSAGE DE LA BASE DE DONNÉES (DATA SEEDING) ---")

tickers = ['AAPL', 'TLT', 'KO', 'PEP'] # J'ajoute Coca/Pepsi pour plus tard

for ticker in tickers:
    print(f"\nTraitement de {ticker}...")
    # On force le téléchargement pour être sûr d'écrire dans la base
    asset = FinancialAsset(ticker, "2020-01-01", "2024-01-01")
    asset.download_data(force_update=True) 

print("\n✅ Base de données remplie avec succès.")