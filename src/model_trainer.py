"""
Module d'entraînement du modèle ML
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import CONFIG, CAT_COLS, NUM_COLS, DATA_PATH, MODEL_PATH


class ModelTrainer:
    """Classe pour entraîner et sauvegarder le modèle ML."""

    def __init__(self, data_path: str = DATA_PATH):
        """
        Initialise le trainer.

        Args:
            data_path: Chemin vers le fichier CSV de données
        """
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.features_final = None

    def load_and_prepare_data(self):
        """Charge et prépare les données avec feature engineering."""
        print("⏳ Chargement des données...")
        data = pd.read_csv(self.data_path)
        print(f"✅ {data.shape[0]:,} dossiers chargés ({data.shape[1]} variables)")

        # Préparation
        features = CAT_COLS + NUM_COLS
        df = data[['SK_ID_CURR', 'TARGET'] + features].copy()

        # Feature engineering
        print("🔧 Feature engineering...")
        df['AGE_YEARS'] = (-df['DAYS_BIRTH']) / 365.25
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED']) / 365.25
        df['REGISTRATION_YEARS'] = (-df['DAYS_REGISTRATION']) / 365.25
        df['ID_PUBLISH_YEARS'] = (-df['DAYS_ID_PUBLISH']) / 365.25
        df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'],
                inplace=True)

        num_cols_updated = [c for c in NUM_COLS if c not in
                           ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']]
        num_cols_updated += ['AGE_YEARS', 'EMPLOYED_YEARS', 'REGISTRATION_YEARS', 'ID_PUBLISH_YEARS']

        # Ratios et métriques calculées
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['INCOME_MONTHLY'] = df['AMT_INCOME_TOTAL'] / 12
        df['DEBT_RATIO'] = (df['AMT_ANNUITY'] / 12) / df['INCOME_MONTHLY'].replace(0, np.nan)
        df['RESTE_A_VIVRE'] = df['INCOME_MONTHLY'] - (df['AMT_ANNUITY'] / 12)
        df['DUREE_PRET_YEARS'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)) / 12
        df['AGE_FIN_PRET'] = df['AGE_YEARS'] + df['DUREE_PRET_YEARS']
        df['CREDIT_TERM_MONTHS'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)
        df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT'].replace(0, np.nan)

        engineered = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO',
                     'CREDIT_TERM_MONTHS', 'INCOME_MONTHLY', 'DEBT_RATIO', 'RESTE_A_VIVRE',
                     'DUREE_PRET_YEARS', 'AGE_FIN_PRET']

        self.features_final = CAT_COLS + num_cols_updated + engineered

        X = df[self.features_final].copy()
        y = df['TARGET'].astype(int)

        print(f"✅ {len(self.features_final)} features préparées")
        print(f"📈 Taux de défaut historique: {y.mean()*100:.2f}%")

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.25, random_state=42):
        """Divise les données en train/val/test."""
        print("🔀 Division des données...")

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        print(f"  Train: {len(self.X_train):,} samples")
        print(f"  Val:   {len(self.X_val):,} samples")
        print(f"  Test:  {len(self.X_test):,} samples")

    def create_pipeline(self):
        """Crée le pipeline de preprocessing et le modèle."""
        print("🏗️  Construction du pipeline...")

        numeric_features = [c for c in self.features_final if c not in CAT_COLS]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ])

        preprocess = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CAT_COLS),
        ])

        to_dense = FunctionTransformer(
            lambda x: x.toarray() if hasattr(x, 'toarray') else x,
            accept_sparse=True
        )

        # Modèle
        self.model = Pipeline([
            ("preprocess", preprocess),
            ("to_dense", to_dense),
            ("model", HistGradientBoostingClassifier(
                random_state=42,
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                class_weight="balanced"
            ))
        ])

        print("✅ Pipeline créé")

    def train(self):
        """Entraîne le modèle."""
        print("🚀 Entraînement du modèle...")

        self.model.fit(self.X_train, self.y_train)

        print("✅ Modèle entraîné")

    def evaluate(self):
        """Évalue le modèle sur validation et test."""
        print("\n📊 Évaluation du modèle:")

        # Validation
        proba_val = self.model.predict_proba(self.X_val)[:, 1]
        auc_val = roc_auc_score(self.y_val, proba_val)
        print(f"  AUC Validation: {auc_val:.4f}")

        # Test
        proba_test = self.model.predict_proba(self.X_test)[:, 1]
        auc_test = roc_auc_score(self.y_test, proba_test)
        print(f"  AUC Test:       {auc_test:.4f}")

        # Prédictions binaires pour le rapport
        y_pred = self.model.predict(self.X_test)
        print("\n📋 Rapport de classification (Test):")
        print(classification_report(self.y_test, y_pred))

        return auc_val, auc_test

    def save_model(self, model_path: str = MODEL_PATH):
        """Sauvegarde le modèle entraîné."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Sauvegarde du modèle dans {model_path}...")

        model_data = {
            'model': self.model,
            'features': self.features_final,
            'auc_val': roc_auc_score(self.y_val, self.model.predict_proba(self.X_val)[:, 1]),
            'auc_test': roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Modèle sauvegardé ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    def run_full_training(self):
        """Lance le pipeline complet d'entraînement."""
        print("=" * 60)
        print("🏦 CreditScore Pro - Entraînement du modèle")
        print("=" * 60)

        # 1. Chargement et préparation
        X, y = self.load_and_prepare_data()

        # 2. Division
        self.split_data(X, y)

        # 3. Création du pipeline
        self.create_pipeline()

        # 4. Entraînement
        self.train()

        # 5. Évaluation
        self.evaluate()

        # 6. Sauvegarde
        self.save_model()

        print("\n✅ Entraînement terminé avec succès!")
        print("=" * 60)

        return self.model, self.features_final


def load_model(model_path: str = MODEL_PATH):
    """
    Charge un modèle sauvegardé.

    Returns:
        (model, features_list, metadata)
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data['model'], model_data['features'], {
        'auc_val': model_data.get('auc_val'),
        'auc_test': model_data.get('auc_test')
    }
