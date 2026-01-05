#!/usr/bin/env python3
"""
Script d'entraînement du modèle ML
"""

import sys
from pathlib import Path

# Ajout du chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from src.model_trainer import ModelTrainer
from src.config import DATA_PATH, MODEL_PATH


def main():
    """Lance l'entraînement du modèle."""

    # Vérification du fichier de données
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"❌ Fichier de données introuvable: {DATA_PATH}")
        print("Assurez-vous que le fichier application_train.csv est dans le dossier data/")
        return 1

    # Création du trainer
    trainer = ModelTrainer(str(data_path))

    try:
        # Entraînement complet
        model, features = trainer.run_full_training()

        print("\n🎉 Modèle prêt à être utilisé!")
        print(f"📁 Chemin: {MODEL_PATH}")
        print(f"📊 Features: {len(features)}")

        return 0

    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
