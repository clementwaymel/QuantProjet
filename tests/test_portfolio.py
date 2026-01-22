import pytest
from queue import Queue
from core.portfolio import Portfolio
from core.event import SignalEvent, OrderEvent

# --- 1. LE MOCK (Le Faux Distributeur de Données) ---
class MockDataHandler:
    """Simule le DataHandler pour ne pas dépendre de la base SQL/Yahoo."""
    def __init__(self):
        self.symbol_list = ['AAPL']
        
    def get_latest_bar(self, symbol):
        # On dit que Apple vaut toujours 150$ pour ce test
        return {'symbol': 'AAPL', 'close': 150.0, 'date': '2024-01-01'}

# --- 2. LE TEST (Le Juge) ---
def test_portfolio_allocation():
    """
    Vérifie si le portefeuille respecte l'allocation de 40% du capital.
    """
    # A. Initialisation (Setup)
    mock_bars = MockDataHandler()
    events = Queue()
    initial_cash = 100000.0
    
    # On crée le portefeuille
    pf = Portfolio(mock_bars, events, initial_capital=initial_cash)
    
    # B. Action (Trigger)
    # On simule un signal d'achat LONG sur Apple
    signal = SignalEvent('AAPL', '2024-01-01', 'LONG', strength=1.0)
    pf.update_signal(signal)
    
    # C. Vérification (Assert)
    # Le portefeuille a dû générer un ordre. On regarde dans la file d'attente.
    assert not events.empty(), "Le portefeuille aurait dû générer un ordre !"
    
    event = events.get()
    assert isinstance(event, OrderEvent), "L'événement généré doit être un OrderEvent"
    assert event.symbol == 'AAPL'
    assert event.direction == 'BUY'
    
    # D. Le Calcul Mathématique (C'est là qu'on vérifie l'ingénierie)
    # On voulait investir 40% de 100 000$, soit 40 000$.
    # Le prix est de 150$.
    # Quantité attendue = floor(40000 / 150) = 266 actions.
    expected_qty = int((100000 * 0.40) // 150)
    
    print(f"\nQuantité calculée par le robot : {event.quantity}")
    print(f"Quantité attendue mathématiquement : {expected_qty}")
    
    assert event.quantity == expected_qty, f"Erreur d'allocation ! Reçu {event.quantity}, attendu {expected_qty}"

def test_portfolio_valuation():
    """
    Vérifie que le calcul de l'Equity (Mark-to-Market) est juste.
    """
    mock_bars = MockDataHandler()
    events = Queue()
    pf = Portfolio(mock_bars, events, initial_capital=100000.0)
    
    # On force manuellement une position pour tester le calcul
    pf.current_positions['AAPL'] = 10
    pf.current_cash = 50000.0
    
    # On lance le calcul de valorisation
    pf.mark_to_market('2024-01-01')
    
    # Vérification :
    # Equity = Cash (50k) + (10 actions * 150$)
    # Equity = 50000 + 1500 = 51500
    expected_equity = 51500.0
    
    assert pf.current_equity == expected_equity, f"Erreur de valorisation ! {pf.current_equity} != {expected_equity}"