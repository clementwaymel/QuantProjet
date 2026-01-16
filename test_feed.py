from queue import Queue
from core.data_handler import SQLDataHandler
import time

print("--- TEST DU FLUX EVENT-DRIVEN ---")

# 1. Le canal de communication
events = Queue()

# 2. Les actifs
symbols = ['AAPL', 'TLT'] 

# 3. Initialisation
try:
    feed = SQLDataHandler(events, symbols)
except Exception as e:
    print(f"Erreur init: {e}")
    exit()

# 4. Simulation de la boucle
print("Démarrage du streaming...")
count = 0

while feed.continue_backtest:
    # A. Avance le temps
    feed.update_bars()
    
    # B. Vérifie s'il y a un événement
    while not events.empty():
        event = events.get()
        if event.type == 'MARKET':
            count += 1
            # On affiche tous les 100 jours pour ne pas spammer la console
            if count % 100 == 0:
                # On demande le prix "Actuel"
                aapl_price = feed.get_latest_bar('AAPL')
                tlt_price = feed.get_latest_bar('TLT')
                
                date = aapl_price['date'].strftime('%Y-%m-%d')
                print(f"[{count}] Date: {date} | AAPL: {aapl_price['close']:.2f} $ | TLT: {tlt_price['close']:.2f} $")

print(f"Test terminé. {count} barres traitées avec succès.")