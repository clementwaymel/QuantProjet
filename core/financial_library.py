import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from database_manager import QuantDatabase

class FinancialAsset:
    """
    Cette classe représente un actif financier (Action, ETF, Forex).
    Elle encapsule les données et les calculs mathématiques de base.
    """
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None
        self.daily_returns = None
        self.log_returns = None
        # On instancie la base de données
        self.db = QuantDatabase()

    def download_data(self, force_update=False):
        """
        Stratégie Intelligente : Local First, Network Second.
        Si force_update=True, on force le téléchargement Yahoo.
        """
        # 1. Essai de lecture locale (Si pas force_update)
        if not force_update:
            print(f"[{self.ticker}] Recherche dans la base locale...")
            local_data = self.db.get_ticker_data(self.ticker)
            
            if local_data is not None and not local_data.empty:
                # On vérifie si les dates correspondent (approximativement)
                # Si la dernière date locale est < à self.end, il faudrait mettre à jour (on verra plus tard)
                print(f"[{self.ticker}] Chargé depuis le disque (SQL).")
                self.data = local_data
                self.clean_data()
                return # On s'arrête là, pas besoin d'internet

        # 2. Si on est ici, c'est qu'on n'a pas de données ou qu'on force la mise à jour
        print(f"[{self.ticker}] Téléchargement Yahoo Finance...")
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=True, progress=False)
            
            # Gestion MultiIndex (Ton correctif)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
                df = df.iloc[:, 0]

            self.data = df
            self.clean_data()
            
            # 3. SAUVEGARDE EN BASE pour la prochaine fois
            print(f"[{self.ticker}] Sauvegarde en base SQL...")
            self.db.save_ticker_data(self.ticker, self.data)
            
        except Exception as e:
            print(f"ERREUR CRITIQUE sur {self.ticker} : {e}")

    def clean_data(self):
        """Nettoyage des données (Ingénierie)"""
        # Suppression des NaNs (Jours fériés, bugs)
        self.data.dropna(inplace=True)
        # On pourrait ajouter ici : détection des outliers (prix aberrants)
        
    def compute_returns(self):
        """Calcule les rendements Arithmétiques ET Logarithmiques."""
        if self.data is None:
            raise ValueError("Données non chargées. Lancez download_data() d'abord.")
            
        # 1. Rendements Simples (Arithmétiques) : R_t = (P_t / P_{t-1}) - 1
        self.daily_returns = self.data.pct_change().dropna()
        
        # 2. Log Rendements (Géométriques) : r_t = ln(P_t / P_{t-1})
        # MATHS : np.log est le logarithme népérien
        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()

    def get_volatility(self, annualized=True):
        """Retourne la volatilité (Standard Deviation)"""
        if self.log_returns is None:
            self.compute_returns()
            
        vol = self.log_returns.std()
        if annualized:
            vol = vol * np.sqrt(252) # Annualisation (Racine du temps)
        return vol

    def plot_analysis(self):
        """Affiche un dashboard complet de l'actif"""
        if self.log_returns is None:
            self.compute_returns()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Prix
        ax1.plot(self.data, color='navy')
        ax1.set_title(f"Prix de {self.ticker}")
        ax1.grid(True)
        
        # Distribution des Rendements (Histogramme)
        # C'est ici qu'on vérifie la "Normalité"
        ax2.hist(self.log_returns, bins=50, color='orange', alpha=0.7, density=True)
        ax2.set_title(f"Distribution des Log-Rendements (Volatilité: {self.get_volatility():.2%})")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


    def diagnose(self):
        """
        Effectue une analyse statistique complète (Moments 3 & 4 + Test de Normalité).
        Affiche un rapport et un graphique de distribution.
        """
        if self.log_returns is None:
            self.compute_returns()
            
        series = self.log_returns
        
        # 1. Calcul des Moments
        mu = series.mean()
        sigma = series.std()
        skew = series.skew()
        kurt = series.kurtosis() # Pandas donne l'Excess Kurtosis (Normal = 0)
        
        # 2. Test de Jarque-Bera (Test de Normalité)
        # H0 : La distribution est Normale (Gaussienne)
        jb_stat, p_value = stats.jarque_bera(series)
        is_normal = p_value > 0.05
        
        # 3. Affichage du Rapport "Ingénieur"
        print(f"\n--- DIAGNOSTIC STATISTIQUE : {self.ticker} ---")
        print(f"Période analysée : {len(series)} jours")
        print(f"1. Moyenne (Drift)    : {mu:.6f} (proche de 0 ?)")
        print(f"2. Volatilité (Sigma) : {sigma:.6f}")
        print(f"3. Skewness (Biais)   : {skew:.4f} " + ("(Danger: Négatif)" if skew < -0.5 else "(OK)"))
        print(f"4. Kurtosis (Queues)  : {kurt:.4f} " + ("(Danger: Fat Tails)" if kurt > 2.0 else "(Normal)"))
        print("-" * 30)
        print(f"Test Jarque-Bera      : p-value = {p_value:.4e}")
        if is_normal:
            print(">> CONCLUSION : Distribution NORMALE (Modèles classiques valides).")
        else:
            print(">> CONCLUSION : Distribution ANORMALE (Risque extrême sous-estimé).")
            print(">> Les modèles Gaussiens (Sharpe, Black-Scholes) doivent être ajustés.")

        # 4. Visualisation Avancée (QQ-Plot + Histogramme)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogramme vs Loi Normale Théorique
        count, bins, ignored = ax1.hist(series, 50, density=True, alpha=0.6, color='blue', label='Réel')
        # Courbe théorique
        plt_x = np.linspace(min(bins), max(bins), 100)
        plt_y = stats.norm.pdf(plt_x, mu, sigma)
        ax1.plot(plt_x, plt_y, linewidth=2, color='r', label='Loi Normale Théorique')
        ax1.set_title(f"Distribution des Rendements ({self.ticker})")
        ax1.legend()
        ax1.grid(True)
        
        # QQ-Plot (Quantile-Quantile Plot)
        # Si les points bleus suivent la ligne rouge, c'est normal.
        # S'ils s'écartent aux extrémités, c'est des Fat Tails.
        stats.probplot(series, dist="norm", plot=ax2)
        ax2.set_title("QQ-Plot (Détection des Fat Tails)")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()




class Portfolio:
    """
    Gestionnaire de portefeuille multi-actifs (Version Debugging)
    """
    
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.positions = {}
        self.assets = {}
        self.history = None
        
    def add_position(self, asset_object, quantity):
        """Ajoute un actif. Si l'actif existe déjà, on met à jour la quantité."""
        if asset_object.data is None:
            print(f"Erreur : Pas de données pour {asset_object.ticker}")
            return

        ticker = asset_object.ticker
        self.assets[ticker] = asset_object
        self.positions[ticker] = quantity
        print(f"[Portefeuille] Position mise à jour : {ticker} = {quantity} unités")
        
    def compute_portfolio_value(self):
        # 1. Alignement des dates (Union)
        all_dates = pd.Index([])
        for asset in self.assets.values():
            all_dates = all_dates.union(asset.data.index)
            
        df_portfolio = pd.DataFrame(index=all_dates.sort_values())
        
        # 2. Calcul Valeur Investie
        df_portfolio['Invested_Value'] = 0.0 # Initialisation à 0
        
        cost_basis = 0
        
        for ticker, qty in self.positions.items():
            # Alignement des prix
            asset_data = self.assets[ticker].data.reindex(df_portfolio.index).ffill()
            
            # Valeur = Prix * Quantité
            # C'est ici que si qty=0, la valeur DOIT être 0
            current_value = asset_data * qty
            
            # Stockage individuel pour vérification
            df_portfolio[f'Value_{ticker}'] = current_value.fillna(0)
            
            # Somme totale investie
            df_portfolio['Invested_Value'] += df_portfolio[f'Value_{ticker}']
            
            # Calcul du coût initial (pour le cash)
            # On prend le premier prix valide
            first_valid_price = asset_data.dropna().iloc[0]
            cost_basis += first_valid_price * qty

        # 3. Calcul du Cash et Total
        start_cash = self.initial_capital - cost_basis
        df_portfolio['Cash'] = start_cash # Le cash est constant dans cette version simple
        df_portfolio['Total_Value'] = df_portfolio['Invested_Value'] + df_portfolio['Cash']
        
        self.history = df_portfolio.dropna()
        return self.history

    def plot_portfolio(self):
        """Nouvelle version : Lignes séparées pour bien voir qui fait quoi"""
        if self.history is None:
            self.compute_portfolio_value()
            
        # --- PHASE DE DEBUG (Console) ---
        print("\n--- AUDIT DU PORTEFEUILLE (Dernier Jour Connu) ---")
        last_row = self.history.iloc[-1]
        for col in self.history.columns:
            if col.startswith('Value_'):
                print(f"  -> Actif {col} : {last_row[col]:.2f} $")
        print(f"  -> Cash Restant : {last_row['Cash']:.2f} $")
        print(f"  -> VALEUR TOTALE : {last_row['Total_Value']:.2f} $")
        print("-" * 40)
            
        # --- GRAPHIQUE ---
        plt.figure(figsize=(12, 8))
        
        # 1. La courbe principale (Total)
        plt.plot(self.history['Total_Value'], label='Total Portefeuille', color='black', linewidth=3)
        
        # 2. Les composants (Lignes fines)
        # On ne les empile plus, on les affiche telles quelles
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        color_idx = 0
        
        value_cols = [c for c in self.history.columns if c.startswith('Value_')]
        for col in value_cols:
            # On n'affiche que si la valeur n'est pas nulle
            if self.history[col].sum() != 0: 
                plt.plot(self.history[col], label=col, linestyle='--', alpha=0.7, linewidth=1.5)
            else:
                print(f"Info Graphique : {col} est une ligne plate à 0 (non affichée)")

        plt.title("Décomposition du Portefeuille (Lignes non empilées)")
        plt.ylabel("Valeur en $")
        plt.legend()
        plt.grid(True)
        plt.show()