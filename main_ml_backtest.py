from queue import Queue
import matplotlib.pyplot as plt
from core.data_handler import SQLDataHandler
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from strategies.ml_pairs_strategy import MLPairsTradingStrategy

print("--- BACKTEST LIVE : IA EN PRODUCTION (2021-2024) ---")

events = Queue()
symbols = ['KO', 'PEP']

# 1. On démarre le DataHandler SEULEMENT sur la période de TEST (Inconnue)
# L'IA a été entraînée jusqu'en 2020. On teste sur ce qui suit.
start_date_simulation = "2021-01-01"

try:
    data = SQLDataHandler(events, symbols, start_date=start_date_simulation)
    # On branche la stratégie IA
    strategy = MLPairsTradingStrategy(data, events, model_path="models/meta_model_ko_pep.joblib", z_entry=0.75)
    portfolio = Portfolio(data, events, initial_capital=100000)
    broker = SimulatedExecutionHandler(events)
except Exception as e:
    print(f"Erreur critique : {e}")
    exit()

print(f"Démarrage de la simulation sur données inconnues ({start_date_simulation} -> Aujourd'hui)...")

ticks = 0
while data.continue_backtest:
    data.update_bars()
    while not events.empty():
        event = events.get()
        if event.type == 'MARKET':
            strategy.calculate_signals(event)
            ticks += 1
            latest = data.get_latest_bar('KO')
            if latest: portfolio.mark_to_market(latest['date'])
        elif event.type == 'SIGNAL':
            portfolio.update_signal(event)
        elif event.type == 'ORDER':
            broker.execute_order(event, data)
        elif event.type == 'FILL':
            portfolio.update_fill(event)

# Résultats
df_result = portfolio.get_equity_curve()
final_equity = df_result['equity'].iloc[-1]
roi = (final_equity - 100000) / 100000 * 100

print(f"\n--- RÉSULTATS IA (OUT-OF-SAMPLE) ---")
print(f"Capital Final : {final_equity:.2f} $")
print(f"Performance   : {roi:.2f} %")

plt.figure(figsize=(12, 6))
plt.plot(df_result.index, df_result['equity'], label='Stratégie IA (Meta-Labeling)')
plt.title(f"Performance IA sur Données Inconnues (ROI: {roi:.2f}%)")
plt.legend()
plt.grid(True)
plt.show()