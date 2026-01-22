from queue import Queue
from core.data_handler import SQLDataHandler
from core.strategy import BuyAndHoldStrategy
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler

print("--- TEST SYSTÈME COMPLET (FULL LOOP) ---")

# 1. Infrastructure
events = Queue()
symbols = ['AAPL', 'TLT']

# 2. Initialisation
data = SQLDataHandler(events, symbols)
strategy = BuyAndHoldStrategy(data, events)
portfolio = Portfolio(data, events, initial_capital=100000)
broker = SimulatedExecutionHandler(events)

# 3. Boucle de Simulation
print("Démarrage du moteur...")
ticks = 0

while data.continue_backtest:
    # A. Avancer le temps
    data.update_bars()
    
    # B. Traiter TOUS les événements
    while not events.empty():
        event = events.get()
        
        if event.type == 'MARKET':
            strategy.calculate_signals(event)
            
        elif event.type == 'SIGNAL':
            portfolio.update_signal(event)
            
        elif event.type == 'ORDER':
            # Le portefeuille a envoyé un ordre, le courtier l'exécute
            broker.execute_order(event, data)
            
        elif event.type == 'FILL':
            # Le courtier confirme, le portefeuille met à jour les comptes
            portfolio.update_fill(event)

print("Test terminé.")