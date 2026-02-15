from queue import Queue
import matplotlib.pyplot as plt
from core.data_handler import SQLDataHandler
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from strategies.multi_pairs_strategy import MultiPairsTradingStrategy

print("--- BACKTEST MULTI-PAIRS (ARBITRAGE STATISTIQUE) ---")

events = Queue()

# Le "Golden Portfolio" de paires cointégrées
pairs_to_trade = [
    ('GOOGL', 'MSFT'),
    ('V', 'MA'),
    ('LOW', 'HD'),
    ('KO', 'PEP'),
    ('CAT', 'DE')
]

# Extraction d'une liste unique de symboles pour le DataHandler
symbols = list(set([sym for pair in pairs_to_trade for sym in pair]))

# Initialisation
print(f"Chargement de {len(symbols)} actifs pour {len(pairs_to_trade)} paires...")
data = SQLDataHandler(events, symbols, start_date="2018-01-01")

# On utilise la stratégie Multi-Pairs (avec des seuils Z-Score modérés)
strategy = MultiPairsTradingStrategy(data, events, pairs_to_trade, z_entry=2.5, z_exit=0.0)

# Portefeuille avec 100 000$ (Le levier est géré en interne)
portfolio = Portfolio(data, events, initial_capital=100000)
broker = SimulatedExecutionHandler(events, slippage_std=0.0001)

print("Démarrage de la simulation globale...")
ticks = 0

while data.continue_backtest:
    data.update_bars()
    
    while not events.empty():
        event = events.get()
        
        if event.type == 'MARKET':
            strategy.calculate_signals(event)
            ticks += 1
            
            # Mark-to-Market basé sur la date du premier symbole disponible
            latest = data.get_latest_bar(symbols[0])
            if latest:
                portfolio.mark_to_market(latest['date'])
            
        elif event.type == 'SIGNAL':
            portfolio.update_signal(event)
        elif event.type == 'ORDER':
            broker.execute_order(event, data)
        elif event.type == 'FILL':
            portfolio.update_fill(event)

# --- ANALYSE DES RÉSULTATS ---
print("Simulation terminée.")

df_result = portfolio.get_equity_curve()
final_equity = df_result['equity'].iloc[-1]
roi = (final_equity - 100000) / 100000 * 100

print(f"Capital Final : {final_equity:.2f} $")
print(f"Rendement (ROI) : {roi:.2f} %")

# --- GRAPHIQUE FINAL ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(df_result.index, df_result['equity'], color='blue', label='Valeur Totale')
plt.axhline(100000, color='black', linestyle='--')
plt.title(f"Performance Multi-Pairs Trading (ROI: {roi:.2f}%)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df_result.index, df_result['cash'], color='green', label='Cash Disponible')
plt.plot(df_result.index, df_result['positions_value'], color='orange', label='Valeur Investie (Spread)')
plt.title("Cash vs Exposition Marché")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()