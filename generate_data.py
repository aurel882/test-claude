"""
Générateur de données synthétiques pour l'entraînement
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generer_donnees_credit(n_samples=10000, random_state=42):
    """
    Génère un dataset synthétique réaliste pour l'analyse de crédit.

    Args:
        n_samples: Nombre d'échantillons à générer
        random_state: Seed pour reproductibilité

    Returns:
        DataFrame avec les données générées
    """
    np.random.seed(random_state)

    print(f"🔧 Génération de {n_samples:,} dossiers synthétiques...")

    # ID
    data = {
        'SK_ID_CURR': range(1, n_samples + 1)
    }

    # Démographie
    data['CODE_GENDER'] = np.random.choice(['M', 'F'], n_samples, p=[0.48, 0.52])
    data['FLAG_OWN_CAR'] = np.random.choice(['Y', 'N'], n_samples, p=[0.35, 0.65])
    data['FLAG_OWN_REALTY'] = np.random.choice(['Y', 'N'], n_samples, p=[0.70, 0.30])
    data['CNT_CHILDREN'] = np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                           p=[0.40, 0.25, 0.20, 0.10, 0.04, 0.01])
    data['CNT_FAM_MEMBERS'] = data['CNT_CHILDREN'] + np.random.choice([1, 2], n_samples, p=[0.3, 0.7])

    # Âge (en jours négatifs comme dans le dataset original)
    ages_years = np.random.normal(40, 12, n_samples)
    ages_years = np.clip(ages_years, 21, 68)
    data['DAYS_BIRTH'] = -(ages_years * 365.25).astype(int)

    # Emploi
    data['NAME_INCOME_TYPE'] = np.random.choice(
        ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed'],
        n_samples, p=[0.50, 0.25, 0.15, 0.08, 0.02]
    )

    # Ancienneté emploi (en jours négatifs)
    employed_years = np.random.exponential(5, n_samples)
    employed_years = np.clip(employed_years, 0, 40)
    # Valeur spéciale pour les retraités/chômeurs
    is_not_working = np.isin(data['NAME_INCOME_TYPE'], ['Pensioner', 'Unemployed'])
    data['DAYS_EMPLOYED'] = np.where(is_not_working, 365243, -(employed_years * 365.25).astype(int))

    # Revenus (corrélés avec l'âge et le type d'emploi)
    base_income = np.random.lognormal(10.5, 0.6, n_samples)
    income_multiplier = np.where(data['NAME_INCOME_TYPE'] == 'State servant', 1.2,
                        np.where(data['NAME_INCOME_TYPE'] == 'Commercial associate', 1.3,
                        np.where(data['NAME_INCOME_TYPE'] == 'Pensioner', 0.6,
                        np.where(data['NAME_INCOME_TYPE'] == 'Unemployed', 0.4, 1.0))))
    data['AMT_INCOME_TOTAL'] = (base_income * income_multiplier * 1000).astype(int)

    # Montant du crédit (corrélé avec les revenus)
    credit_ratio = np.random.lognormal(1.2, 0.8, n_samples)
    credit_ratio = np.clip(credit_ratio, 0.5, 10)
    data['AMT_CREDIT'] = (data['AMT_INCOME_TOTAL'] * credit_ratio).astype(int)

    # Annuités (mensualités * 12)
    # Durée aléatoire entre 1 et 25 ans
    duree_mois = np.random.choice([12, 24, 36, 60, 84, 120, 180, 240, 300], n_samples,
                                   p=[0.05, 0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.07, 0.03])
    # Taux d'intérêt approximatif
    taux_mensuel = 0.035 / 12  # ~3.5% annuel
    mensualite = data['AMT_CREDIT'] * (taux_mensuel * (1 + taux_mensuel)**duree_mois) / \
                 ((1 + taux_mensuel)**duree_mois - 1)
    data['AMT_ANNUITY'] = (mensualite * 12).astype(int)

    # Prix des biens
    data['AMT_GOODS_PRICE'] = (data['AMT_CREDIT'] * np.random.uniform(0.9, 1.1, n_samples)).astype(int)

    # Éducation
    data['NAME_EDUCATION_TYPE'] = np.random.choice(
        ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary'],
        n_samples, p=[0.70, 0.20, 0.07, 0.03]
    )

    # Situation familiale
    data['NAME_FAMILY_STATUS'] = np.random.choice(
        ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
        n_samples, p=[0.60, 0.20, 0.10, 0.07, 0.03]
    )

    # Type de logement
    data['NAME_HOUSING_TYPE'] = np.random.choice(
        ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment'],
        n_samples, p=[0.85, 0.07, 0.04, 0.03, 0.01]
    )

    # Type de contrat
    data['NAME_CONTRACT_TYPE'] = np.random.choice(['Cash loans', 'Revolving loans'],
                                                   n_samples, p=[0.90, 0.10])

    # Occupation
    occupations = ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers',
                  'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
                  'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers']
    data['OCCUPATION_TYPE'] = np.random.choice(occupations + [np.nan], n_samples,
                                               p=[0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.20])

    # Organisation
    organizations = ['Business Entity Type 3', 'XNA', 'Self-employed', 'Other', 'Medicine',
                    'Government', 'School', 'Trade: type 7', 'Industry: type 9']
    data['ORGANIZATION_TYPE'] = np.random.choice(organizations, n_samples)

    # Autres dates
    data['DAYS_REGISTRATION'] = -(np.random.uniform(0, 25, n_samples) * 365.25).astype(int)
    data['DAYS_ID_PUBLISH'] = -(np.random.uniform(0, 20, n_samples) * 365.25).astype(int)

    # Âge de la voiture (si possède)
    data['OWN_CAR_AGE'] = np.where(data['FLAG_OWN_CAR'] == 'Y',
                                   np.random.exponential(7, n_samples),
                                   np.nan)

    # Flags binaires
    data['FLAG_MOBIL'] = 1
    data['FLAG_EMP_PHONE'] = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    data['FLAG_WORK_PHONE'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['FLAG_CONT_MOBILE'] = np.random.choice([0, 1], n_samples, p=[0.01, 0.99])
    data['FLAG_PHONE'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['FLAG_EMAIL'] = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])

    # Région
    data['REGION_POPULATION_RELATIVE'] = np.random.uniform(0.0, 0.1, n_samples)
    data['REGION_RATING_CLIENT'] = np.random.choice([1, 2, 3], n_samples, p=[0.15, 0.75, 0.10])

    # Informations sur le logement
    data['WEEKDAY_APPR_PROCESS_START'] = np.random.choice(
        ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'],
        n_samples
    )
    data['NAME_TYPE_SUITE'] = np.random.choice(
        ['Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_B'],
        n_samples, p=[0.80, 0.10, 0.05, 0.03, 0.02]
    )

    # Informations sur le bâtiment (avec NaN)
    data['FONDKAPREMONT_MODE'] = np.random.choice(['reg oper account', 'org spec account', 'reg oper spec account', np.nan],
                                                   n_samples, p=[0.40, 0.20, 0.05, 0.35])
    data['HOUSETYPE_MODE'] = np.random.choice(['block of flats', 'terraced house', 'specific housing', np.nan],
                                               n_samples, p=[0.50, 0.15, 0.05, 0.30])
    data['WALLSMATERIAL_MODE'] = np.random.choice(['Panel', 'Stone, brick', 'Block', 'Wooden', 'Mixed', np.nan],
                                                   n_samples, p=[0.30, 0.25, 0.15, 0.05, 0.05, 0.20])
    data['EMERGENCYSTATE_MODE'] = np.random.choice(['No', 'Yes', np.nan],
                                                    n_samples, p=[0.70, 0.05, 0.25])

    # TARGET - Calculé selon des critères réalistes
    # Facteurs de risque
    debt_ratio = data['AMT_ANNUITY'] / (data['AMT_INCOME_TOTAL'] + 1)
    income_ratio = data['AMT_CREDIT'] / (data['AMT_INCOME_TOTAL'] + 1)
    age_years = -np.array(data['DAYS_BIRTH']) / 365.25

    # Score de risque (plus c'est haut, plus c'est risqué)
    risk_score = (
        (debt_ratio > 0.40) * 0.3 +  # Endettement élevé
        (debt_ratio > 0.50) * 0.3 +  # Endettement critique
        (income_ratio > 8) * 0.2 +   # Crédit très élevé vs revenu
        (age_years < 25) * 0.15 +    # Jeune
        (age_years > 65) * 0.15 +    # Âgé
        (data['NAME_INCOME_TYPE'] == 'Unemployed') * 0.4 +  # Chômeur
        (data['CNT_CHILDREN'] > 3) * 0.1 +  # Beaucoup d'enfants
        (data['FLAG_OWN_REALTY'] == 'N') * 0.1 +  # Pas de propriété
        np.random.uniform(0, 0.3, n_samples)  # Facteur aléatoire
    )

    # Convertir en probabilité puis en target binaire
    default_proba = 1 / (1 + np.exp(-3 * (risk_score - 0.5)))
    data['TARGET'] = (np.random.random(n_samples) < default_proba).astype(int)

    df = pd.DataFrame(data)

    print(f"✅ Dataset généré:")
    print(f"   - {len(df):,} dossiers")
    print(f"   - {df.shape[1]} colonnes")
    print(f"   - Taux de défaut: {df['TARGET'].mean()*100:.2f}%")

    return df


def sauvegarder_dataset(output_path='data/application_train.csv'):
    """Génère et sauvegarde le dataset."""
    df = generer_donnees_credit(n_samples=10000)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"💾 Dataset sauvegardé: {output_path}")
    print(f"   Taille: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return df


if __name__ == '__main__':
    sauvegarder_dataset()
