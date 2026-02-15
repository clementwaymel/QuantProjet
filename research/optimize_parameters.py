from queue import Queue
import pandas as pd
import numpy as np
from core.data_handler import SQLDataHandler
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from strategies.multi_pairs_strategy import MultiPairsTradingStrategy

print("--- LABORATOIRE D'OPTIMISATION (GRID SEARCH) ---")

# 1. PARAMÈTRES À TESTER
# On va tester différents seuils d'entrée Z-Score
z_entries_to_test = [3.0, 3.5,4.0,4.5,5.0,5.5,6.0]

# Liste des paires (Celles du Hedge Fund)
pairs = [
    ('XOM', 'CVX'), ('MSFT', 'AMD'), ('JPM', 'BAC'),
    ('KO', 'PEP'), ('WMT', 'TGT'), ('PFE', 'MRK')
]
all_symbols = list(set([t for pair in pairs for t in pair]))

results = []

print(f"Optimisation sur {len(z_entries_to_test)} scénarios...")

for z_entry in z_entries_to_test:
    print(f"\n>> TEST DU PARAMÈTRE : Z_ENTRY = {z_entry}")
    
    # Réinitialisation complète de l'environnement pour chaque test
    events = Queue()
    try:
        # On teste sur une période significative (2020-2023)
        data = SQLDataHandler(events, all_symbols, start_date="2020-01-01")
        
        # On injecte le paramètre testé
        strategy = MultiPairsTradingStrategy(data, events, pairs, z_entry=z_entry, z_exit=0.0)
        
        portfolio = Portfolio(data, events, initial_capital=100000)
        broker = SimulatedExecutionHandler(events, slippage_std=0.0001)
        
        # Exécution silencieuse (pas de print à chaque trade pour aller vite)
        while data.continue_backtest:
            data.update_bars()
            while not events.empty():
                event = events.get()
                if event.type == 'MARKET':
                    strategy.calculate_signals(event)
                    # Mark-to-market moins fréquent pour la vitesse
                    # Mais nécessaire pour le résultat final
                    # On le fait à la fin de la boucle si possible, ou tous les 100 ticks
                    pass 
                elif event.type == 'SIGNAL':
                    portfolio.update_signal(event)
                elif event.type == 'ORDER':
                    broker.execute_order(event, data)
                elif event.type == 'FILL':
                    portfolio.update_fill(event)
        
        # Calcul final précis
        # On force un mark-to-market à la toute dernière date connue
        last_date = data.get_latest_bar(all_symbols[0])['date']
        portfolio.mark_to_market(last_date)
        
        final_val = portfolio.current_equity
        roi = (final_val - 100000) / 100000 * 100
        nb_trades = len(portfolio.get_equity_curve()) # Approximation activité
        
        print(f"   RÉSULTAT : ROI = {roi:.2f}% | Capital = {final_val:.0f}$")
        
        results.append({
            'Z_Entry': z_entry,
            'ROI': roi,
            'Final_Equity': final_val
        })
        
    except Exception as e:
        print(f"Erreur sur le scénario {z_entry}: {e}")

# --- ANALYSE DES RÉSULTATS ---
print("\n" + "="*40)
print("TABLEAU D'HONNEUR DES PARAMÈTRES")
print("="*40)

df_res = pd.DataFrame(results).sort_values(by='ROI', ascending=False)
print(df_res)

best_z = df_res.iloc[0]['Z_Entry']
print(f"\n✅ RECOMMANDATION : Utilisez Z_ENTRY = {best_z} pour le prochain backtest.")