from queue import Queue
import matplotlib.pyplot as plt # Import pour le graphique
from core.data_handler import SQLDataHandler
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from strategies.pairs_strategy import PairsTradingStrategy

print("--- BACKTEST : PAIRS TRADING (COCA vs PEPSI) ---")

events = Queue()
symbols = ['KO', 'PEP']

# Initialisation
data = SQLDataHandler(events, symbols)
strategy = PairsTradingStrategy(data, events, window_size=30)
portfolio = Portfolio(data, events, initial_capital=100000)
broker = SimulatedExecutionHandler(events)

print("Démarrage de la simulation...")
ticks = 0

while data.continue_backtest:
    # A. Avancer le temps
    data.update_bars()
    
    # B. Traiter les événements
    while not events.empty():
        event = events.get()
        
        if event.type == 'MARKET':
            strategy.calculate_signals(event)
            ticks += 1
            
            # --- NOUVEAU : Calcul Mark-to-Market à chaque tick ---
            # On récupère la date actuelle (via un actif, ex: KO)
            latest = data.get_latest_bar('KO')
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

# On récupère l'historique sous forme de DataFrame
df_result = portfolio.get_equity_curve()

# Calcul de la performance finale
final_equity = df_result['equity'].iloc[-1]
roi = (final_equity - 100000) / 100000 * 100

print(f"Capital Final : {final_equity:.2f} $")
print(f"Rendement (ROI) : {roi:.2f} %")

# --- GRAPHIQUE FINAL ---
plt.figure(figsize=(12, 8))

# Courbe 1 : Valeur Totale (Equity)
plt.subplot(2, 1, 1)
plt.plot(df_result.index, df_result['equity'], color='blue', label='Valeur Totale')
plt.axhline(100000, color='black', linestyle='--')
plt.title(f"Performance Stratégie Pairs Trading (ROI: {roi:.2f}%)")
plt.legend()
plt.grid(True)

# Courbe 2 : Décomposition (Cash vs Investi)
plt.subplot(2, 1, 2)
plt.plot(df_result.index, df_result['cash'], color='green', label='Cash Disponible')
plt.plot(df_result.index, df_result['positions_value'], color='orange', label='Valeur Investie (Spread)')
plt.title("Cash vs Exposition Marché")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()