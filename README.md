# 🏦 CreditScore Pro

**Application professionnelle d'analyse de crédit avec Machine Learning**

Une solution complète d'évaluation de demandes de crédit combinant intelligence artificielle et règles métier bancaires françaises (HCSF 2022).

---

## 📋 Table des matières

- [Caractéristiques](#-caractéristiques)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
  - [Notebook Jupyter](#1-notebook-jupyter)
  - [API REST](#2-api-rest)
  - [CLI](#3-interface-en-ligne-de-commande)
- [Entraînement du modèle](#-entraînement-du-modèle)
- [Structure du projet](#-structure-du-projet)
- [Détails techniques](#-détails-techniques)
- [Licence](#-licence)

---

## ✨ Caractéristiques

### 🤖 Machine Learning
- Modèle hybride : **HistGradientBoostingClassifier** + Règles métier
- Feature engineering avancé (ratios financiers, métriques calculées)
- AUC Test > 0.75
- Entraîné sur 300 000+ dossiers réels

### 📊 Analyse financière
- Calcul de mensualités
- Taux d'endettement
- Reste à vivre
- Capacité d'emprunt
- Tableau d'amortissement

### 🎯 Moteur de décision
- **3 décisions possibles** : ACCEPTÉ / ACCEPTÉ SOUS CONDITIONS / REFUSÉ
- Respect des normes HCSF (35% max d'endettement)
- Critères : âge, revenus, charges, ancienneté emploi, apport
- Score combiné (60% règles métier + 40% ML)

### 🚀 Interfaces multiples
1. **Notebook Jupyter interactif** avec widgets et visualisations Plotly
2. **API REST** (FastAPI) pour intégration
3. **CLI** (Command Line Interface) pour usage terminal

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER INPUT                        │
│  (Revenu, Montant, Durée, Âge, Ancienneté, etc.)  │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│            CALCULATEUR FINANCIER                     │
│  • Mensualité  • Taux d'endettement                │
│  • Reste à vivre  • Capacité max                   │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│            MOTEUR DE DÉCISION                        │
│                                                      │
│  ┌──────────────┐        ┌──────────────┐          │
│  │  RÈGLES      │  60%   │   MODÈLE     │  40%     │
│  │  MÉTIER      ├────────┤     ML       │          │
│  │  (Banking)   │        │  (Gradient   │          │
│  │              │        │   Boosting)  │          │
│  └──────────────┘        └──────────────┘          │
│         │                       │                   │
│         └───────────┬───────────┘                   │
│                     ▼                               │
│              SCORE FINAL                            │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│                   DÉCISION                           │
│  • ACCEPTÉ (score ≥ 70)                             │
│  • ACCEPTÉ SOUS CONDITIONS (50 ≤ score < 70)        │
│  • REFUSÉ (score < 50 ou critères bloquants)        │
└─────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. **Cloner le repository**
   ```bash
   git clone <url-du-repo>
   cd test-claude
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Entraîner le modèle** (optionnel si déjà fait)
   ```bash
   python train_model.py
   ```
   Le modèle sera sauvegardé dans `models/credit_model.pkl`

---

## 🎮 Utilisation

### 1. Notebook Jupyter

Interface interactive avec visualisations élaborées.

```bash
jupyter notebook notebooks/CreditScore_Pro.ipynb
```

**Fonctionnalités** :
- Formulaire interactif avec widgets
- Analyse en temps réel
- Visualisations Plotly (jauges, graphiques, tableaux d'amortissement)
- Explications pédagogiques du modèle

### 2. API REST

API FastAPI pour intégration dans vos applications.

**Démarrage**:
```bash
cd api
python main.py
```

L'API sera accessible sur `http://localhost:8000`

**Documentation interactive** : `http://localhost:8000/docs`

**Endpoints principaux** :

```bash
# Health check
GET /health

# Analyser une demande de crédit
POST /analyser
{
  "revenu_annuel": 50000,
  "montant_credit": 200000,
  "duree_annees": 20,
  "age": 35,
  "anciennete_emploi": 5.0,
  "nb_enfants": 2,
  "charges_existantes": 500,
  "apport": 20000
}

# Calculer une mensualité
GET /calculer/mensualite?capital=200000&taux_annuel=0.035&duree_annees=20

# Calculer la capacité d'emprunt
GET /calculer/capacite?revenu_mensuel=4000&taux_annuel=0.035&duree_annees=20
```

**Exemple avec curl** :
```bash
curl -X POST "http://localhost:8000/analyser" \
  -H "Content-Type: application/json" \
  -d '{
    "revenu_annuel": 50000,
    "montant_credit": 200000,
    "duree_annees": 20,
    "age": 35,
    "anciennete_emploi": 5.0,
    "nb_enfants": 2,
    "charges_existantes": 500,
    "apport": 20000
  }'
```

### 3. Interface en ligne de commande

CLI pour analyse rapide en terminal.

**Analyser un dossier** :
```bash
python cli.py analyser \
  --revenu 50000 \
  --montant 200000 \
  --duree 20 \
  --age 35 \
  --anciennete 5 \
  --enfants 2 \
  --charges 500 \
  --apport 20000
```

**Calculer une mensualité** :
```bash
python cli.py mensualite --capital 200000 --taux 0.035 --duree 20
```

**Calculer la capacité d'emprunt** :
```bash
python cli.py capacite --revenu 4000 --taux 0.035 --duree 20
```

**Sortie JSON** :
```bash
python cli.py analyser --revenu 50000 --montant 200000 --duree 20 --age 35 --json-output
```

---

## 🎓 Entraînement du modèle

### Données

Le dataset `application_train.csv` (Home Credit Default Risk) doit être placé dans `data/`.

**Features utilisées** :
- **Démographiques** : âge, situation familiale, éducation, logement
- **Professionnelles** : type de contrat, revenus, ancienneté emploi, type d'organisation
- **Financières** : montant crédit, annuités, biens, ratios calculés
- **Engineered** : taux d'endettement, reste à vivre, âge fin de prêt, etc.

### Lancer l'entraînement

```bash
python train_model.py
```

**Sortie** :
```
============================================================
🏦 CreditScore Pro - Entraînement du modèle
============================================================
⏳ Chargement des données...
✅ 307,511 dossiers chargés (122 variables)
🔧 Feature engineering...
✅ 54 features préparées
📈 Taux de défaut historique: 8.07%
🔀 Division des données...
  Train: 184,506 samples
  Val:   61,503 samples
  Test:  61,502 samples
🏗️  Construction du pipeline...
✅ Pipeline créé
🚀 Entraînement du modèle...
✅ Modèle entraîné

📊 Évaluation du modèle:
  AUC Validation: 0.7623
  AUC Test:       0.7589

💾 Sauvegarde du modèle dans models/credit_model.pkl...
✅ Modèle sauvegardé (45.3 MB)

✅ Entraînement terminé avec succès!
============================================================
```

---

## 📁 Structure du projet

```
test-claude/
├── data/
│   └── application_train.csv       # Dataset (166 MB, Git LFS)
│
├── src/                             # Code source
│   ├── __init__.py
│   ├── config.py                    # Configuration et paramètres
│   ├── calculator.py                # Calculateur de crédit
│   ├── decision_engine.py           # Moteur de décision
│   └── model_trainer.py             # Entraînement ML
│
├── api/                             # API REST
│   ├── __init__.py
│   ├── main.py                      # Application FastAPI
│   └── schemas.py                   # Schémas Pydantic
│
├── notebooks/
│   └── CreditScore_Pro.ipynb        # Notebook interactif
│
├── models/
│   └── credit_model.pkl             # Modèle entraîné (généré)
│
├── cli.py                           # Interface CLI
├── train_model.py                   # Script d'entraînement
├── requirements.txt                 # Dépendances
├── .gitignore
└── README.md                        # Ce fichier
```

---

## 🔬 Détails techniques

### Modèle ML

**Pipeline** :
```python
Pipeline([
    Preprocessing:
      - Imputation (médiane pour num, mode fréquent pour cat)
      - Standardisation (StandardScaler)
      - OneHotEncoding (catégories)

    Model:
      - HistGradientBoostingClassifier
      - max_depth=6, learning_rate=0.05
      - max_iter=300, class_weight="balanced"
])
```

**Métriques** :
- AUC ROC : ~0.76
- Précision : ~92%
- Recall : ~24% (volontairement conservateur)

### Règles métier (HCSF 2022)

| Critère | Valeur |
|---------|--------|
| Taux d'endettement max | 35% |
| Reste à vivre min | 700€/personne + 300€/enfant |
| Âge max fin de prêt | 75 ans |
| Durée max immobilier | 25 ans |
| Durée max consommation | 7 ans |
| Taux immobilier | 3.5% |
| Taux consommation | 6.5% |

### Technologies

- **ML** : scikit-learn, numpy, pandas
- **Viz** : plotly, matplotlib, ipywidgets
- **API** : FastAPI, Pydantic, Uvicorn
- **CLI** : Click, Rich

---

## ⚠️ Avertissement

Cette application est un **outil éducatif et de démonstration**. Elle ne constitue en aucun cas :
- Une offre de prêt
- Un conseil financier personnalisé
- Une garantie d'obtention de crédit

Pour toute demande de crédit réelle, veuillez consulter un établissement bancaire agréé.

---

## 📝 Licence

Ce projet est développé à des fins éducatives.

---

## 👨‍💻 Auteur

Développé dans le cadre d'un projet de Data Science - M2

---

## 🙏 Remerciements

- Dataset : [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)
- Normes bancaires françaises HCSF 2022
