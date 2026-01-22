import datetime
import random
from core.event import FillEvent, OrderEvent

class ExecutionHandler:
    """
    Interface abstraite pour l'exécution.
    """
    def execute_order(self, event, data_handler):
        raise NotImplementedError("Doit implémenter execute_order()")

class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulateur réaliste prenant en compte :
    1. La Commission (Frais fixes + variables)
    2. Le Slippage (Modèle Stochastique)
    3. Le Spread implicite
    """
    def __init__(self, events_queue, commission_per_share=0.005, min_commission=1.0, slippage_std=0.0001):
        self.events_queue = events_queue
        
        # Paramètres de coûts
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        
        # Paramètre de volatilité d'exécution (Slippage)
        # 0.0001 = 1 point de base (bps) de volatilité sur le prix d'exécution
        self.slippage_std = slippage_std

    def calculate_commission(self, quantity, price):
        """
        Modèle de commission type Interactive Brokers (Pro).
        Max(1$, 0.005$ par action).
        On plafonne souvent à 1% du montant du trade dans la réalité, mais restons simples.
        """
        full_cost = max(self.min_commission, self.commission_per_share * abs(quantity))
        return full_cost

    def calculate_slippage(self, price, direction):
        """
        Simule l'impact de marché et le spread.
        On suppose que le prix reçu (price) est le 'Mid-Price'.
        On ajoute un bruit aléatoire (Loi Normale) pour simuler le prix réel d'exécution.
        """
        # Biais de spread : on paie toujours un peu plus cher qu'on ne le voudrait
        # 5 points de base de pénalité fixe (Spread)
        spread_penalty = 0.0005 
        
        # Bruit de marché (Slippage aléatoire)
        market_noise = random.gauss(0, self.slippage_std)
        
        slippage_factor = spread_penalty + market_noise
        
        if direction == 'BUY':
            # À l'achat, le prix monte (en notre défaveur)
            execution_price = price * (1 + slippage_factor)
        else:
            # À la vente, le prix baisse (en notre défaveur)
            execution_price = price * (1 - slippage_factor)
            
        return execution_price

    def execute_order(self, event, data_handler):
        """
        Transforme un Ordre en 'Fill' avec des prix dégradés par la réalité.
        """
        if event.type == 'ORDER':
            # Récupération du prix théorique (Mid-Price)
            latest_bar = data_handler.get_latest_bar(event.symbol)
            
            if latest_bar:
                theoretical_price = latest_bar['close']
                date = latest_bar['date']
                
                # 1. Application du Slippage
                real_price = self.calculate_slippage(theoretical_price, event.direction)
                
                # 2. Calcul des Commissions
                commission = self.calculate_commission(event.quantity, real_price)
                
                # 3. Création de la confirmation (Fill)
                fill_event = FillEvent(
                    timeindex=date,
                    symbol=event.symbol,
                    exchange='ARCA_SIM',
                    quantity=event.quantity,
                    direction=event.direction,
                    fill_cost=real_price,
                    commission=commission
                )
                
                self.events_queue.put(fill_event)
                
                # Log Ingénieur pour vérifier le réalisme
                diff = abs(real_price - theoretical_price)
                print(f"[Exec] {event.direction} {event.quantity} {event.symbol}")
                print(f"       Théorique: {theoretical_price:.2f} | Exécuté: {real_price:.2f} (Slippage: {diff:.4f}$)")
                print(f"       Commission: {commission:.2f}$")