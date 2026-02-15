import pandas as pd
import numpy as np
from queue import Queue
from core.event import MarketEvent
from core.database_manager import QuantDatabase

class DataHandler:
    """
    Classe abstraite (Interface) pour gérer la distribution des données.
    """
    def get_latest_bar(self, symbol):
        raise NotImplementedError("Doit implémenter get_latest_bar()")

    def update_bars(self):
        raise NotImplementedError("Doit implémenter update_bars()")

class SQLDataHandler(DataHandler):
    """
    Charge l'historique depuis SQLite et le distribue barre par barre
    pour simuler le temps réel (Drip feed).
    """
    def __init__(self, events_queue, symbol_list, start_date=None):
        self.events_queue = events_queue # La file d'attente
        self.symbol_list = symbol_list
        self.start_date = start_date     # <--- NOUVEAU PARAMÈTRE
        
        self.db = QuantDatabase()
        self.symbol_data = {} # Stocke tout le DataFrame en cache
        self.latest_symbol_data = {} # Stocke juste la DERNIÈRE barre connue
        self.continue_backtest = True 
        
        self.bar_generator = None 
        
        # Chargement initial
        self._load_data()
        
    def _load_data(self):
        """Charge toutes les données de la DB en mémoire et prépare l'itérateur."""
        print("[DataHandler] Chargement des données historiques...")
        
        combined_index = None
        
        for symbol in self.symbol_list:
            raw_data = self.db.get_ticker_data(symbol)
            
            if raw_data is None:
                print(f"ERREUR: Pas de données pour {symbol}")
                continue
            
            # Transformation en DataFrame si nécessaire
            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame(name='Close')

            # --- FILTRE DE DATE (NOUVEAU) ---
            if self.start_date is not None:
                # On filtre pour ne garder que ce qui est APRÈS la start_date
                # Utile pour tester l'IA sur des données inconnues
                mask = (raw_data.index >= self.start_date)
                raw_data = raw_data.loc[mask]
            
            self.symbol_data[symbol] = raw_data
            
            # Construction de l'index global
            if combined_index is None:
                combined_index = raw_data.index
            else:
                combined_index = combined_index.union(raw_data.index)
            
            self.latest_symbol_data[symbol] = []
        
        # Réindexation globale
        if combined_index is not None:
            for symbol in self.symbol_list:
                self.symbol_data[symbol] = self.symbol_data[symbol].reindex(index=combined_index, method='ffill')
                self.symbol_data[symbol] = self.symbol_data[symbol].itertuples()

            self.bar_generator = iter(combined_index)
            print(f"[DataHandler] Prêt. {len(combined_index)} périodes chargées (Début: {combined_index[0]}).")
        else:
            print("ERREUR CRITIQUE : Aucune donnée chargée (vérifiez les dates ou la base).")
            self.continue_backtest = False

    def get_latest_bar(self, symbol):
        """Retourne la dernière info prix connue"""
        try:
            bars_list = self.latest_symbol_data[symbol]
            return bars_list[-1]
        except (KeyError, IndexError):
            return None

    def update_bars(self):
        """Avance le temps d'un cran (t -> t+1)."""
        try:
            current_date = next(self.bar_generator)
        except (StopIteration, TypeError):
            self.continue_backtest = False
            return
        
        for symbol in self.symbol_list:
            try:
                bar = next(self.symbol_data[symbol])
                
                # bar[0] est l'index, bar[1] est le Close (ou bar.Close)
                close_price = bar.Close if hasattr(bar, 'Close') else bar[1]
                
                self.latest_symbol_data[symbol].append({
                    'symbol': symbol,
                    'date': current_date,
                    'close': close_price
                })
                
                if len(self.latest_symbol_data[symbol]) > 100:
                    self.latest_symbol_data[symbol].pop(0)
                    
            except StopIteration:
                self.continue_backtest = False
                return

        self.events_queue.put(MarketEvent())