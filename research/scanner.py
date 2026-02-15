import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
from itertools import combinations
from core.financial_library import FinancialAsset

class CointegrationScanner:
    """
    Explorateur de marché.
    Cherche des paires cointégrées (liées par un élastique invisible) 
    au sein d'un panier d'actifs.
    """
    def __init__(self, tickers_list, start_date, end_date):
        self.tickers = tickers_list
        self.start = start_date
        self.end = end_date
        self.data_matrix = pd.DataFrame()
        
    def load_data(self):
        """Télécharge tout l'univers d'investissement."""
        print(f"[Scanner] Chargement de {len(self.tickers)} actifs...")
        
        for t in self.tickers:
            # On utilise notre librairie (et le cache SQL !)
            asset = FinancialAsset(t, self.start, self.end)
            asset.download_data()
            
            if asset.data is not None and not asset.data.empty:
                self.data_matrix[t] = asset.data
        
        # Nettoyage : On ne garde que les lignes où tout le monde est coté
        self.data_matrix.dropna(inplace=True)
        print(f"[Scanner] Matrice de prix prête : {self.data_matrix.shape}")

    def find_best_pairs(self, p_value_threshold=0.05):
        """
        Teste toutes les combinaisons possibles (Brute Force).
        Retourne les paires avec une p-value < 0.05 (Cointégration prouvée).
        """
        if self.data_matrix.empty:
            self.load_data()
            
        keys = self.data_matrix.columns
        # Génère toutes les paires uniques (A, B) sans doublons
        # Si 10 actifs -> 45 paires à tester. Si 20 actifs -> 190 paires.
        pairs_to_test = list(combinations(keys, 2))
        
        print(f"[Scanner] Lancement des tests de Engle-Granger sur {len(pairs_to_test)} paires...")
        
        valid_pairs = []
        
        for asset_a, asset_b in pairs_to_test:
            series_a = self.data_matrix[asset_a]
            series_b = self.data_matrix[asset_b]
            
            # Test de Cointégration (coint)
            # Retourne : t-stat, p-value, crit_values
            score, pvalue, _ = ts.coint(series_a, series_b)
            
            if pvalue < p_value_threshold:
                # C'est une pépite !
                print(f" >> DÉCOUVERTE : {asset_a} vs {asset_b} (p-value={pvalue:.4f})")
                valid_pairs.append({
                    'Y': asset_a,
                    'X': asset_b,
                    'p_value': pvalue
                })
        
        # On trie par qualité (p-value la plus basse = lien le plus fort)
        valid_pairs.sort(key=lambda x: x['p_value'])
        
        return valid_pairs

