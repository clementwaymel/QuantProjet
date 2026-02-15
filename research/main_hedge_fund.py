import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

print("--- HEDGE FUND 'DYNAMIC HALF-LIFE' ---")

# --- 1. CONFIGURATION ---
pairs_to_trade = [
    ('GOOGL', 'MSFT'), ('MSFT', 'ADBE'), ('CAT', 'DE'),
    ('TXN', 'ADI'), ('GS', 'MS'), ('LOW', 'HD'),
    ('V', 'MA'), ('HON', 'MMM'), ('CL', 'COST'),
    ('TGT', 'WMT')
]

Z_ENTRY = 2.0  
Z_EXIT = 0.0
LEVERAGE = 4.0 # On monte un peu le levier car la méthode est plus précise

# --- 2. FONCTION MATHÉMATIQUE AVANCÉE ---
def calculate_half_life(spread_series):
    """
    Calcule la demi-vie d'un spread via un processus d'Ornstein-Uhlenbeck.
    Formule: dx(t) = theta * (mu - x(t)) * dt + sigma * dW(t)
    """
    # 1. On crée le décalage (Lag)
    spread_lag = spread_series.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    
    # 2. On calcule la variation (Ret)
    spread_ret = spread_series - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    
    # 3. Régression Linéaire pour trouver Theta (Mean Reversion Speed)
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    
    # Theta est le coefficient de la pente (négatif pour mean reversion)
    theta = res.params.iloc[1]
    
    if theta >= 0: return 60 # Pas de mean reversion détectée (Random Walk)
    
    # 4. Calcul Demi-Vie : -ln(2) / theta
    half_life = -np.log(2) / theta
    return max(5, min(120, int(half_life))) # On borne entre 5 et 120 jours

# --- 3. CHARGEMENT DONNÉES ---
def load_data(tickers):
    data_dict = {}
    for t in tickers:
        path = f"data/{t}.csv"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if 'Close' in df.columns: s = df['Close']
                elif 'Adj Close' in df.columns: s = df['Adj Close']
                else: s = df.iloc[:, 0]
                data_dict[t] = s.dropna()
            except: pass
    return pd.DataFrame(data_dict).dropna()

# --- 4. MOTEUR ---
class BacktestEngine:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = {} 
        self.equity_curve = []
        self.cash = initial_capital
        
    def run(self, data, strategy):
        timestamps = data.index
        print(f">>> Simulation avec Fenêtres Dynamiques (Half-Life)...")
        
        for t in timestamps:
            # Valorisation
            portfolio_value = self.cash
            for pair_id, pos in self.positions.items():
                py = data.at[t, pos['symbol_y']]
                px = data.at[t, pos['symbol_x']]
                pos_value = (pos['qty_y'] * py) + (pos['qty_x'] * px)
                portfolio_value += pos_value - pos['cost_basis']
            
            self.equity_curve.append({'date': t, 'equity': portfolio_value})
            
            # Trading
            signals = strategy.on_bar(t, self.positions)
            for sig in signals:
                self.execute(sig, data, t)
                
        return pd.DataFrame(self.equity_curve).set_index('date')

    def execute(self, signal, data, t):
        pair_id = signal['pair_id']
        equity = self.equity_curve[-1]['equity']
        
        if signal['action'] == 'OPEN':
            if pair_id in self.positions or equity <= 0: return
            
            # Allocation
            allocation = (equity / len(pairs_to_trade)) * LEVERAGE
            
            py = data.at[t, signal['y']]
            px = data.at[t, signal['x']]
            
            # DOLLAR NEUTRAL (Plus robuste que Beta Neutre)
            qty_y = (allocation / 2) / py * (-1 if signal['side'] == 'SHORT_Y' else 1)
            qty_x = (allocation / 2) / px * (1 if signal['side'] == 'SHORT_Y' else -1)
            
            self.positions[pair_id] = {
                'symbol_y': signal['y'], 'symbol_x': signal['x'],
                'qty_y': qty_y, 'qty_x': qty_x,
                'cost_basis': (qty_y * py) + (qty_x * px)
            }
            self.cash -= (abs(qty_y*py) + abs(qty_x*px)) * 0.0005

        elif signal['action'] == 'CLOSE':
            if pair_id not in self.positions: return
            pos = self.positions[pair_id]
            py = data.at[t, pos['symbol_y']]
            px = data.at[t, pos['symbol_x']]
            pnl = ((pos['qty_y'] * py) + (pos['qty_x'] * px)) - pos['cost_basis']
            
            self.cash += pnl
            self.cash -= (abs(pos['qty_y']*py) + abs(pos['qty_x']*px)) * 0.0005
            del self.positions[pair_id]

# --- 5. STRATÉGIE ADAPTATIVE ---
class AdaptiveStrategy:
    def __init__(self, data, pairs):
        self.data = data
        self.pairs = pairs
        self.pair_params = {} # Pour stocker la Half-Life de chaque paire
        
        # Phase d'apprentissage (Calibration)
        print(">>> Calibration mathématique (Ornstein-Uhlenbeck)...")
        for y, x in pairs:
            # On prend la première année pour calibrer
            ratio = data[y].iloc[:252] / data[x].iloc[:252]
            hl = calculate_half_life(ratio)
            # On fixe la fenêtre à 1x ou 2x la demi-vie (ici 1x pour être réactif)
            window = int(hl) 
            self.pair_params[f"{y}_{x}"] = window
            print(f"   Pair {y}-{x} : Half-Life détectée = {hl} jours -> Window = {window}")
        
    def on_bar(self, t, current_positions):
        signals = []
        idx = self.data.index.get_loc(t)
        
        for y_sym, x_sym in self.pairs:
            pair_id = f"{y_sym}_{x_sym}"
            window = self.pair_params[pair_id] # Paramètre dynamique !
            
            if idx < window: continue
            
            # Extraction des données
            s_y = self.data[y_sym].iloc[idx-window:idx+1]
            s_x = self.data[x_sym].iloc[idx-window:idx+1]
            
            # Ratio
            ratio = s_y / s_x
            
            # Z-Score sur la fenêtre adaptée
            mu = ratio.mean()
            std = ratio.std()
            
            if std == 0: continue
            z = (ratio.iloc[-1] - mu) / std
            
            # Trading
            if pair_id in current_positions:
                pos = current_positions[pair_id]
                if (pos['qty_y'] < 0 and z <= Z_EXIT) or \
                   (pos['qty_y'] > 0 and z >= -Z_EXIT):
                    signals.append({'action': 'CLOSE', 'pair_id': pair_id})
                elif abs(z) > 4.5:
                    signals.append({'action': 'CLOSE', 'pair_id': pair_id})
            else:
                if z > Z_ENTRY:
                    signals.append({'action': 'OPEN', 'pair_id': pair_id, 'y': y_sym, 'x': x_sym, 'side': 'SHORT_Y'})
                elif z < -Z_ENTRY:
                    signals.append({'action': 'OPEN', 'pair_id': pair_id, 'y': y_sym, 'x': x_sym, 'side': 'LONG_Y'})
                    
        return signals

# --- 6. EXÉCUTION ---
tickers = list(set([p[0] for p in pairs_to_trade] + [p[1] for p in pairs_to_trade]))
full_data = load_data(tickers)

if not full_data.empty:
    strategy = AdaptiveStrategy(full_data, pairs_to_trade)
    engine = BacktestEngine()
    results = engine.run(full_data, strategy)

    final_equity = results['equity'].iloc[-1]
    roi = (final_equity - 100000) / 100000 * 100
    peak = results['equity'].cummax()
    max_dd = ((results['equity'] - peak) / peak).min() * 100

    print("\n" + "="*40)
    print(f"RÉSULTAT STRATÉGIE ADAPTATIVE (HALF-LIFE)")
    print(f"ROI Total     : {roi:.2f}%")
    print(f"Max Drawdown  : {max_dd:.2f}%")
    print("="*40)

    plt.figure(figsize=(12, 6))
    plt.plot(results['equity'], label='Equity Half-Life', color='#007acc')
    plt.title(f"Stratégie Adaptative (Ornstein-Uhlenbeck) - ROI: {roi:.2f}%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()