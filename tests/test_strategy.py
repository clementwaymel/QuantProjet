from queue import Queue
from core.data_handler import SQLDataHandler
from core.strategy import BuyAndHoldStrategy

print("--- TEST INTÉGRATION : DATA + STRATEGY ---")

# 1. Infrastructure
events = Queue()
symbols = ['AAPL', 'TLT']

# 2. Initialisation des composants
try:
    # Le Coeur (Données)
    data = SQLDataHandler(events, symbols)
    # Le Cerveau (Décision) - On lui donne accès aux données
    strategy = BuyAndHoldStrategy(data, events)
except Exception as e:
    print(f"Erreur Init: {e}")
    exit()

# 3. Boucle de Simulation
print("Démarrage de la boucle événementielle...")
ticks = 0

while data.continue_backtest:
    # A. Avancer le temps (T -> T+1)
    data.update_bars()
    
    # B. Traiter les événements (La "Game Loop")
    while not events.empty():
        event = events.get()
        
        # Cas 1 : Nouvelle donnée de marché
        if event.type == 'MARKET':
            ticks += 1
            # On réveille la stratégie pour qu'elle analyse
            strategy.calculate_signals(event)

        # Cas 2 : La stratégie a généré un signal !
        elif event.type == 'SIGNAL':
            print(f" >> SIGNAL REÇU : {event.signal_type} sur {event.symbol} (Force: {event.strength})")


print(f"Test terminé. {ticks} ticks analysés.")