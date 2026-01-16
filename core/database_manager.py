import sqlite3
import pandas as pd
import os

class QuantDatabase:
    """
    Gère le stockage local des données financières (SQLite).
    Permet de sauvegarder et de relire les historiques sans internet.
    """
    
    def __init__(self, db_name="quant_data.db"):
        self.db_name = db_name
        self.conn = None
        self.initialize_db()
        
    def connect(self):
        """Ouvre la connexion au fichier de base de données."""
        self.conn = sqlite3.connect(self.db_name)
        
    def close(self):
        """Ferme la connexion proprement."""
        if self.conn:
            self.conn.close()

    def initialize_db(self):
        """
        Création de l'architecture des tables (Schema).
        On utilise du SQL brut pour créer la table si elle n'existe pas.
        """
        self.connect()
        cursor = self.conn.cursor()
        
        # SQL : CREATE TABLE
        # On définit les colonnes : Date (Texte), Ticker (Texte), Prix (Réel)
        # PRIMARY KEY (Date, Ticker) empêche d'avoir deux fois la même donnée pour le même jour
        query = """
        CREATE TABLE IF NOT EXISTS market_data (
            date TEXT,
            ticker TEXT,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (date, ticker)
        )
        """
        cursor.execute(query)
        self.conn.commit()
        self.close()
        print(f"[Database] Base '{self.db_name}' initialisée/vérifiée.")

    def save_ticker_data(self, ticker, df):
        """
        Prend un DataFrame (Yahoo) et l'insère dans la base SQL.
        """
        if df is None or df.empty:
            print(f"[Database] Pas de données à sauvegarder pour {ticker}")
            return

        self.connect()
        
        # Préparation du DataFrame pour qu'il colle au format SQL
        # On s'assure d'avoir 'date', 'ticker', 'close', 'volume'
        
        # Copie pour ne pas casser l'original
        df_sql = df.copy()
        
        # Si 'Close' est une Série, on la transforme en DataFrame
        if isinstance(df_sql, pd.Series):
            df_sql = df_sql.to_frame(name='close')
        
        # On s'assure que 'close' est bien le nom de la colonne
        if 'Close' in df_sql.columns:
            df_sql.rename(columns={'Close': 'close'}, inplace=True)
            
        # On ajoute la colonne Ticker
        df_sql['ticker'] = ticker
        
        # On remet la date en colonne (si elle est en index)
        df_sql.reset_index(inplace=True)
        # On renomme 'Date' en 'date' (minuscule)
        df_sql.rename(columns={'Date': 'date'}, inplace=True)
        
        # On garde uniquement les colonnes utiles
        cols_to_keep = ['date', 'ticker', 'close']
        # Si on a le volume, on le garde, sinon on met 0
        if 'Volume' in df_sql.columns:
            df_sql.rename(columns={'Volume': 'volume'}, inplace=True)
            cols_to_keep.append('volume')
        
        # Écriture dans la base (Magie de Pandas)
        # if_exists='replace' remplacerait tout la table (Dangereux !)
        # if_exists='append' ajoute à la suite.
        try:
            # On utilise 'append' mais attention aux doublons.
            # Pour faire propre, on supprime d'abord les vieilles données de ce ticker
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM market_data WHERE ticker = ?", (ticker,))
            
            df_sql[cols_to_keep].to_sql('market_data', self.conn, if_exists='append', index=False)
            print(f"[Database] {len(df_sql)} lignes sauvegardées pour {ticker}.")
            
        except Exception as e:
            print(f"[Database] Erreur sauvegarde {ticker} : {e}")
            
        self.conn.commit()
        self.close()

    def get_ticker_data(self, ticker):
        """
        Lit les données depuis la base SQL et renvoie un DataFrame prêt à l'emploi.
        """
        self.connect()
        
        query = f"SELECT date, close FROM market_data WHERE ticker = '{ticker}' ORDER BY date ASC"
        
        try:
            df = pd.read_sql(query, self.conn, parse_dates=['date'], index_col='date')
            
            # On renomme pour que ça ressemble à Yahoo (Compatible avec notre ancien code)
            df.rename(columns={'close': 'Close'}, inplace=True)
            
            self.close()
            
            if df.empty:
                print(f"[Database] Aucune donnée trouvée pour {ticker} en local.")
                return None
                
            return df['Close'] # On renvoie une Série comme avant
            
        except Exception as e:
            print(f"[Database] Erreur lecture {ticker} : {e}")
            self.close()
            return None