from queue import Queue
from core.execution import SimulatedExecutionHandler
from core.event import OrderEvent

# On mock (simule) un DataHandler pour le test
class MockDataHandler:
    def get_latest_bar(self, symbol):
        return {'date': '2024-01-01', 'close': 100.00} # Prix fixe à 100$

print("--- TEST UNITAIRE : EXÉCUTION RÉALISTE ---")

events = Queue()
# On instancie le simulateur avec un slippage un peu fort pour bien voir l'effet
broker = SimulatedExecutionHandler(events, slippage_std=0.001) # 0.1% de volatilité
mock_data = MockDataHandler()

# 1. Test ACHAT (BUY)
print("\n--- Test 1 : ACHAT 1000 Actions à 100$ (Théorique) ---")
order_buy = OrderEvent("AAPL", "MKT", 1000, "BUY")
broker.execute_order(order_buy, mock_data)

# On récupère le résultat
fill_buy = events.get()
# Prix > 100$ (car on paie le spread + slippage)
print(f"Résultat : Acheté à {fill_buy.fill_cost:.4f} $") 

# 2. Test VENTE (SELL)
print("\n--- Test 2 : VENTE 1000 Actions à 100$ (Théorique) ---")
order_sell = OrderEvent("AAPL", "MKT", 1000, "SELL")
broker.execute_order(order_sell, mock_data)

fill_sell = events.get()
# Prix < 100$ (car on vend au Bid + slippage)
print(f"Résultat : Vendu à {fill_sell.fill_cost:.4f} $")

print("-" * 30)
spread_cost = (fill_buy.fill_cost - fill_sell.fill_cost)
print(f"Coût total du cycle (Spread réalisé) : {spread_cost:.4f} $ par action")