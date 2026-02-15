from math import floor
from queue import Queue
from core.event import OrderEvent
import pandas as pd

class Portfolio:
    def __init__(self, bars, events_queue, initial_capital=100000.0, leverage=4.0):
        self.bars = bars
        self.events_queue = events_queue
        self.initial_capital = initial_capital
        self.leverage = leverage # Levier Cible (ex: 4.0)
        
        # --- SÉCURITÉ (NOUVEAU) ---
        # Levier MAX autoritaire (Disjoncteur)
        # On ne dépassera JAMAIS 4x les capitaux propres, quoi qu'il arrive.
        self.max_leverage_cap = 4.0 
        
        self.current_positions = {s: 0 for s in self.bars.symbol_list}
        self.current_cash = initial_capital
        self.current_equity = initial_capital
        self.equity_curve = []

    # ... (Gardes update_fill, update_signal, mark_to_market inchangés) ...
    # Je remets update_signal pour la clarté de l'appel
    def update_signal(self, event):
        if event.type == 'SIGNAL':
            order_event = self.generate_leveraged_order(event)
            if order_event is not None:
                self.events_queue.put(order_event)

    def generate_leveraged_order(self, signal):
        """
        VERSION 5.0 (SAFE) : Avec Disjoncteur de Levier.
        """
        order = None
        symbol = signal.symbol
        direction = signal.signal_type
        order_type = 'MKT'
        
        bar = self.bars.get_latest_bar(symbol)
        if bar is None: return None
        price = bar['close']
        if price == 0: return None
        
        cur_quantity = self.current_positions[symbol]
        
        if direction == 'EXIT':
            if cur_quantity > 0:
                order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
            elif cur_quantity < 0:
                order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
            return order

        # --- LOGIQUE DE GESTION DU RISQUE ---
        
        # 1. Calcul de l'exposition actuelle totale
        current_exposure = 0.0
        for s, qty in self.current_positions.items():
            # On approxime avec le dernier prix connu
            last_bar = self.bars.get_latest_bar(s)
            if last_bar:
                current_exposure += abs(qty * last_bar['close'])
        
        # 2. Vérification du Plafond (Circuit Breaker)
        # Si on est déjà à fond (Levier > 3.8), on interdit d'ouvrir de nouvelles positions
        current_leverage = current_exposure / self.current_equity if self.current_equity > 0 else 999
        
        if direction in ['LONG', 'SHORT'] and current_leverage >= self.max_leverage_cap:
            # print(f"[RISK] Ordre bloqué ! Levier actuel {current_leverage:.2f} >= Max {self.max_leverage_cap}")
            return None

        # 3. Calcul de la taille idéale (Risk Parity)
        target_risk_percent = 0.0010 
        
        # Sécurité sur la volatilité (on évite la division par zéro ou par un nombre minuscule)
        volatility = getattr(signal, 'est_volatility', 0.015)
        if volatility < 0.005: volatility = 0.005 # On force un plancher de 0.5% de vol
        
        # Formule Kelly/Risk Parity
        position_value = (self.current_equity * target_risk_percent) / volatility
        
        # 4. Seconde Sécurité : Cap par Position individuelle
        # On ne met jamais plus de 10% du "Buying Power Max" sur une seule ligne
        max_position_value = (self.current_equity * self.leverage) * 0.10
        
        final_dollar_amount = min(position_value, max_position_value)
        
        target_qty = int(floor(final_dollar_amount / price))
        
        if target_qty == 0: return None

        if direction == 'LONG':
            order = OrderEvent(symbol, order_type, target_qty, 'BUY')
        elif direction == 'SHORT':
            order = OrderEvent(symbol, order_type, target_qty, 'SELL')
            
        return order

    # ... (Le reste : update_fill, mark_to_market restent inchangés) ...
    # N'oublie pas de garder mark_to_market sinon le graph est vide !
    def update_fill(self, event):
        if event.type == 'FILL':
            fill_dir = 1 if event.direction == 'BUY' else -1
            self.current_positions[event.symbol] += fill_dir * event.quantity
            cost = fill_dir * event.fill_cost * event.quantity
            self.current_cash -= (cost + event.commission)

    def mark_to_market(self, current_time):
        market_value = 0
        for symbol in self.current_positions:
            qty = self.current_positions[symbol]
            bar = self.bars.get_latest_bar(symbol)
            if bar is not None:
                market_value += qty * bar['close']
        self.current_equity = self.current_cash + market_value
        self.equity_curve.append({
            'datetime': current_time,
            'equity': self.current_equity,
            'cash': self.current_cash,
            'positions_value': market_value
        })
    
    def get_equity_curve(self):
        return pd.DataFrame(self.equity_curve).set_index('datetime')