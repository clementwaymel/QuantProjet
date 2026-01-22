from math import floor
from queue import Queue
from core.event import OrderEvent
import pandas as pd

class Portfolio:
    """
    Gestionnaire de positions et de valorisation (Mark-to-Market).
    """
    def __init__(self, bars, events_queue, initial_capital=100000.0):
        self.bars = bars              # Accès au DataHandler
        self.events_queue = events_queue
        self.initial_capital = initial_capital
        
        # État des positions {Ticker: Quantité}
        self.current_positions = {s: 0 for s in self.bars.symbol_list}
        
        # État du Cash
        self.current_cash = initial_capital
        
        # État de la Valeur Totale (Equity)
        self.current_equity = initial_capital
        
        # Historique pour le graphique final (Liste de dictionnaires)
        self.equity_curve = [] 

    def update_signal(self, event):
        """Reçoit un Signal -> Génère un Ordre"""
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            if order_event is not None:
                self.events_queue.put(order_event)

    def generate_naive_order(self, signal):
        """
        VERSION CORRIGÉE : Dynamic Position Sizing.
        On investit proportionnellement à la taille du compte.
        """
        order = None
        symbol = signal.symbol
        direction = signal.signal_type
        order_type = 'MKT'
        
        # 1. On récupère le prix actuel
        bar = self.bars.get_latest_bar(symbol)
        if bar is None:
            return None
        price = bar['close']
        
        # 2. Combien on possède déjà ?
        cur_quantity = self.current_positions[symbol]
        
        # 3. CALCUL DE LA TAILLE DE POSITION (RISK MANAGEMENT)
        # On veut allouer 40% du capital (Equity) sur ce trade
        target_allocation = 0.40 
        
        # Note : On utilise current_equity (Valeur Totale) et pas juste le cash
        target_exposure = self.current_equity * target_allocation
        
        # Nombre d'actions à acheter = Montant Cible / Prix unitaire
        # floor() arrondit à l'entier inférieur (on ne peut pas acheter 0.5 action)
        if price == 0: return None
        target_qty = int(floor(target_exposure / price))
        
        if direction == 'LONG':
            # Achat
            order = OrderEvent(symbol, order_type, target_qty, 'BUY')
            
        elif direction == 'SHORT':
            # Vente à découvert
            order = OrderEvent(symbol, order_type, target_qty, 'SELL')
            
        elif direction == 'EXIT':
            # Fermeture de position
            if cur_quantity > 0:
                order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
            elif cur_quantity < 0:
                order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
            
        return order

    def update_fill(self, event):
        """Met à jour Cash et Positions après exécution"""
        if event.type == 'FILL':
            fill_dir = 1 if event.direction == 'BUY' else -1
            
            # 1. Mise à jour Quantité
            self.current_positions[event.symbol] += fill_dir * event.quantity
            
            # 2. Mise à jour Cash (Prix + Commission)
            cost = fill_dir * event.fill_cost * event.quantity
            self.current_cash -= (cost + event.commission)
            
            # Debug
            # print(f"[Portefeuille] Fill {event.symbol}. Cash: {self.current_cash:.2f}")

    def mark_to_market(self, current_time):
        """
        CALCUL CRUCIAL : Combien vaut mon portefeuille MAINTENANT ?
        Cette méthode doit être appelée à chaque 'Tick' de marché.
        """
        market_value = 0
        
        # Pour chaque actif suivi
        for symbol in self.current_positions:
            qty = self.current_positions[symbol]
            
            # On demande le prix actuel au DataHandler
            bar = self.bars.get_latest_bar(symbol)
            if bar is not None:
                price = bar['close'] # Prix actuel
                market_value += qty * price
            
        # Equity = Cash + Valeur de liquidation des positions
        self.current_equity = self.current_cash + market_value
        
        # On enregistre l'histoire
        self.equity_curve.append({
            'datetime': current_time,
            'equity': self.current_equity,
            'cash': self.current_cash,
            'positions_value': market_value
        })
        
    def get_equity_curve(self):
        """Renvoie un DataFrame Pandas pour l'analyse"""
        return pd.DataFrame(self.equity_curve).set_index('datetime')