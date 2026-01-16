class Event:
    """
    Classe de base pour tous les événements (Pattern Observer).
    """
    pass

class MarketEvent(Event):
    """
    Se déclenche quand de nouvelles données de marché sont disponibles.
    """
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    """
    Se déclenche quand la Stratégie veut faire quelque chose.
    """
    def __init__(self, symbol, datetime, signal_type, strength=1.0):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type # 'LONG' ou 'SHORT'
        self.strength = strength # Force du signal (pour Kelly)

class OrderEvent(Event):
    """
    Se déclenche quand le Portefeuille envoie un ordre au courtier.
    """
    def __init__(self, symbol, order_type, quantity, direction):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type # 'MARKET' ou 'LIMIT'
        self.quantity = quantity
        self.direction = direction # 'BUY' ou 'SELL'
        
    def print_order(self):
        print(f"ORDRE: {self.direction} {self.quantity} {self.symbol} au prix {self.order_type}")

class FillEvent(Event):
    """
    Se déclenche quand le courtier a exécuté l'ordre (Réalité).
    C'est ici qu'on paie les frais !
    """
    def __init__(self, timeindex, symbol, exchange, quantity, direction, fill_cost, commission=None):
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        
        # Calcul des commissions (ex: Interactive Brokers Fixed)
        if commission is None:
            self.commission = max(1.0, 0.005 * quantity) # Min 1$, ou 0.005$ par action
        else:
            self.commission = commission