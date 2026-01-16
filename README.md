# QuantEngine : Infrastructure de Trading Algorithmique & Recherche Quantitative

**Auteur :** Clément Waymel (Étudiant 3eme année en Mathématiques Appliquées - Polytech)  
**Statut :** En Développement Actif (Mois 2)



## À Propos du Projet

**QuantEngine** est un projet d'ingénierie financière visant à construire un moteur de backtesting complet en Python, en partant de zéro. 

Contrairement à l'utilisation de librairies existantes, ce projet est une démarche pédagogique et technique pour :
1.  **Comprendre les mathématiques financières** (Optimisation,Processus stochastiques, Statistiques).
2.  **Appliquer les concepts d'ingénierie logicielle** (OOP, Design Patterns, Data Engineering).
3.  **Construire une simulation réaliste** (Architecture Événementielle vs Vectorisée).

> **Approche :** Ce projet lie mes connaissances académiques en Algèbre Linéaire Numérique et Systèmes Dynamiques avec la pratique de la finance de marché.



## Phase 1 : Laboratoire Mathématique (Prototypes)
Objectif : Comprendre les concepts financiers fondamentaux avant de coder l'architecture.

Au début du projet, plusieurs scripts isolés ont été développés pour valider la théorie. Ces concepts sont les fondations mathématiques du moteur actuel.

### 1. Gestion du Risque & Volatilité
* **Concepts :** Calcul des rendements logarithmiques (Log-Returns), Volatilité annualisée.
* **VaR (Value at Risk) :** Implémentation de la méthode paramétrique (Loi Normale) pour estimer la perte maximale potentielle avec un intervalle de confiance de 95%.
* **Fichier :** `prototypes/risk_analysis.py` (Archivé)

### 2. Théorie Moderne du Portefeuille (Markowitz)
* **Concepts :** Matrice de Covariance, Diversification, Frontière Efficiente.
* **Réalisation :** Simulation de Monte Carlo (10 000 portefeuilles) pour visualiser le couple Risque/Rendement optimal.
* **Maths :** Minimisation de la variance $\sigma_p^2 = w^T \Sigma w$.

### 3. Stratégies de Régime ("Risk On / Risk Off")
* **Concepts :** Utilisation de données alternatives (Indice VIX - "L'indice de la peur") pour changer dynamiquement d'allocation.
* **Stratégie :** Si VIX > 25 (Crise) $\rightarrow$ Obligations (TLT). Sinon $\rightarrow$ Actions (AAPL).
* **Résultat :** Lissage de la courbe de capital (Drawdown réduit) par rapport au Buy & Hold.

### 4. Gestion de la Mise (Critère de Kelly)
* **Concepts :** Probabilités, Espérance Mathématique.
* **Objectif :** Calculer la taille de mise optimale $f^*$ pour maximiser la croissance géométrique du capital sans risque de ruine.



## Phase 2 : Data Engineering & Statistiques Avancées
*Objectif : Fiabiliser les données et quitter les approximations "naïves".*

### Infrastructure de Données (SQL)
Pour s'affranchir de la dépendance aux API (Yahoo Finance) et accélérer les calculs, un système de cache local a été mis en place.
* **SQLite :** Base de données relationnelle locale.
* **Logiciel :** Classe `QuantDatabase` et `DataHandler` pour gérer les lectures/écritures.
* **Avantage :** Persistance des données et simulation "Offline".

### Analyses Statistiques Rigoureuses
Avant de trader un actif, nous vérifions ses propriétés statistiques (Moments 3 et 4) :
* **Test de Stationnarité (ADF) :** Distinction entre une Marche Aléatoire (Prix) et une série prédictible (Spread).
* **Normalité (Jarque-Bera) :** Détection des **Fat Tails** (Queues de distribution épaisses).
    * *Découverte :* Le Bitcoin et les Actions ont un Kurtosis > 3, invalidant partiellement les modèles Gaussiens classiques (Black-Scholes).
* **Cointégration (Pairs Trading) :** Identification de paires d'actifs (ex: Coca vs Pepsi) liées par une relation long terme, permettant des stratégies de **Mean Reversion** via le Z-Score.



## Phase 3 : Architecture Événementielle (En cours)
*Objectif : Simulation professionnelle réaliste (Event-Driven Backtester).*

Nous sommes passés d'un code "Vectorisé" (calcul sur colonnes) à une **Boucle Événementielle** (`while True`). Cela simule le temps réel et empêche le "Look-ahead Bias" (tricher en connaissant le futur).

### Le Cycle de Vie d'un "Tick" :
1.  **DataHandler :** Lit la base SQL et envoie un événement `MARKET` (Nouveau prix).
2.  **Queue (File d'attente) :** Transporte les messages entre les modules.
3.  **Strategy (Cerveau) :** Reçoit le prix, analyse, et envoie un événement `SIGNAL`.
4.  **Portfolio (Risque) :** Reçoit le signal, vérifie le cash, applique le Kelly Criterion, et génère un `ORDER`. *(En cours de dév)*
5.  **Execution (Broker) :** Simule l'exécution au marché et renvoie un `FILL`. *(En cours de dév)*



## Stack Technique

* **Langage :** Python 3.11+
* **Noyau Numérique :** NumPy, Pandas (Vectorisation, Algèbre Linéaire).
* **Statistiques :** SciPy, Statsmodels (Tests ADF, Cointégration, Régressions).
* **Données :** SQLite3, Yfinance.
* **Outils :** Git/GitHub, VS Code.



## Prochaines Étapes (Roadmap)

- [x] Mise en place de l'environnement et Git.
- [x] Création de la librairie financière (Asset / Portfolio).
- [x] Implémentation de la Base de Données SQL.
- [x] Module de Statistique Avancée (Jarque-Bera, Cointégration).
- [x] Cœur du système événementiel (DataHandler + Strategy).
- [ ] Finalisation du Portefeuille et de l'Exécution.
- [ ] Implémentation de la stratégie Pairs Trading (Z-Score) dans le moteur événementiel.
- [ ] Ajout d'un module de Machine Learning pour la prédiction de tendance.



Ce projet est développé dans une démarche d'apprentissage continu, en appliquant la rigueur des Mathématiques Appliquées aux problématiques de la Finance Quantitative.