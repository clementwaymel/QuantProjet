import pandas as pd
import numpy as np
from queue import Queue
from core.event import MarketEvent
from core.database_manager import QuantDatabase

class DataHandler:
    """
    Classe abstraite (Interface) qui force à respecter une structure.
    """
    def get_latest_bar(self, symbol):
        raise NotImplementedError("Doit implémenter get_latest_bar()")

    def update_bars(self):
        raise NotImplementedError("Doit implémenter update_bars()")

class SQLDataHandler(DataHandler):
    """
    Simulateur de flux temps réel basé sur l'historique SQL.
    """
    def __init__(self, events_queue, symbol_list):
        self.events_queue = events_queue # La file où on enverra les événements 'MARKET'
        self.symbol_list = symbol_list
        
        self.db = QuantDatabase()
        self.symbol_data = {} # Stockage de toutes les données (Source)
        self.latest_symbol_data = {} # Ce que le robot "sait" à l'instant T (Mémoire tampon)
        self.continue_backtest = True # Tant que c'est True, la boucle tourne
        
        # Le curseur temporel (Générateur)
        self.bar_generator = None 
        
        # Chargement immédiat
        self._load_data()
        
    def _load_data(self):
        """
        Charge les données depuis SQL et aligne les dates de tous les actifs.
        """
        print("[DataHandler] Chargement et synchronisation des données...")
        combined_index = None
        
        # 1. Récupération SQL
        for symbol in self.symbol_list:
            raw_data = self.db.get_ticker_data(symbol)
            
            if raw_data is None:
                print(f"ERREUR CRITIQUE: Pas de données SQL pour {symbol}")
                continue
            
            # Transformation en DataFrame si c'est une Série (pour standardiser)
            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame(name='Close')
            
            self.symbol_data[symbol] = raw_data
            
            # Construction de l'axe du temps global (Union des dates)
            if combined_index is None:
                combined_index = raw_data.index
            else:
                combined_index = combined_index.union(raw_data.index)
                
            # Init du tampon
            self.latest_symbol_data[symbol] = []

        # 2. Réindexation (Pour gérer les trous, ex: jours fériés différents)
        # On utilise 'ffill' (Forward Fill) : si pas de prix ajd, on prend celui d'hier.
        for symbol in self.symbol_list:
            self.symbol_data[symbol] = self.symbol_data[symbol].reindex(index=combined_index, method='ffill').dropna()
            
            # Transformation en itérateur (Pour pouvoir faire 'next()')
            self.symbol_data[symbol] = self.symbol_data[symbol].itertuples()

        # Le métronome global est prêt
        self.bar_generator = iter(combined_index)
        print(f"[DataHandler] Prêt. {len(combined_index)} barres chargées.")

    def _get_new_bar(self, symbol):
        """Méthode interne pour piocher la prochaine ligne."""
        # Dans cette version simplifiée, update_bars fait le travail.
        pass

    def get_latest_bar(self, symbol):
        """
        Permet à la Stratégie de demander : "Combien vaut Apple MAINTENANT ?"
        Retourne la dernière ligne ajoutée au tampon.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
            return bars_list[-1]
        except (KeyError, IndexError):
            return None

    def update_bars(self):
        """
        AVANCE LE TEMPS D'UN CRAN (t -> t+1).
        C'est ici que la magie opère.
        """
        try:
            # 1. On avance l'heure globale
            current_date = next(self.bar_generator)
        except StopIteration:
            # Fin de l'historique
            self.continue_backtest = False
            return
        
        # 2. On met à jour chaque actif
        for symbol in self.symbol_list:
            try:
                # On récupère la ligne suivante du générateur
                bar = next(self.symbol_data[symbol])
                
                # On met à jour la "mémoire" du robot
                # bar[0] est l'index (Date), bar[1] est le Close (car colonne 1)
                self.latest_symbol_data[symbol].append({
                    'symbol': symbol,
                    'date': current_date,
                    'close': bar.Close if hasattr(bar, 'Close') else bar[1]
                })
                
                # Optimisation mémoire : on ne garde que les 100 derniers jours en cache actif
                if len(self.latest_symbol_data[symbol]) > 100:
                    self.latest_symbol_data[symbol].pop(0)
                    
            except StopIteration:
                self.continue_backtest = False
                return

        # 3. On crie "NOUVEAU PRIX !" dans le système (Event)
        self.events_queue.put(MarketEvent())