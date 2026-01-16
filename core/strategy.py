from core.event import SignalEvent

class Strategy:
    """
    Classe abstraite. Toutes tes futures stratégies (Markowitz, Z-Score, ML)
    devront hériter de cette classe.
    """
    def calculate_signals(self, event):
        raise NotImplementedError("Doit implémenter calculate_signals()")

class BuyAndHoldStrategy(Strategy):
    """
    Stratégie la plus simple du monde :
    Dès qu'on reçoit la première donnée, on achète et on ne bouge plus.
    Sert à valider le pipeline d'ordres.
    """
    def __init__(self, bars, events_queue):
        self.bars = bars             # Référence au DataHandler (pour lire les prix)
        self.events_queue = events_queue # Là où on envoie les signaux
        self.bought = {}             # Mémoire : "Est-ce que j'ai déjà acheté ?"
        
        # On initialise la mémoire à False pour tous les symboles
        for s in self.bars.symbol_list:
            self.bought[s] = False

    def calculate_signals(self, event):
        """
        Appelé à chaque "Tick" de marché.
        """
        if event.type == 'MARKET':
            for symbol in self.bars.symbol_list:
                # On récupère le dernier prix
                bar = self.bars.get_latest_bar(symbol)
                if bar is None:
                    continue
                
                # LOGIQUE DE TRADING :
                # Si on n'a pas encore acheté et qu'on a un prix -> ACHAT
                if self.bought[symbol] == False:
                    print(f"[Stratégie] Signal d'achat généré pour {symbol} à {bar['date']}")
                    
                    # Création du Signal (Type 'LONG' = Achat)
                    signal = SignalEvent(symbol, bar['date'], 'LONG', strength=1.0)
                    
                    # Envoi dans le système
                    self.events_queue.put(signal)
                    
                    # On note qu'on a acheté pour ne pas racheter demain
                    self.bought[symbol] = True