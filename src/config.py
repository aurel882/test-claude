"""
Configuration et paramètres bancaires réalistes (normes françaises HCSF 2022)
"""

# ============================================================
# PARAMÈTRES BANCAIRES RÉALISTES
# ============================================================

CONFIG = {
    'MAX_DEBT_RATIO': 0.35,           # Taux d'endettement max 35%
    'MIN_RESTE_A_VIVRE': 700,         # € par personne
    'MIN_RESTE_A_VIVRE_ENFANT': 300,  # € par enfant
    'MAX_AGE_FIN_PRET': 75,           # Âge max fin de prêt
    'MIN_AGE': 18,
    'MAX_DUREE_IMMO': 25,             # ans
    'MAX_DUREE_CONSO': 7,             # ans
    'TAUX_IMMO': 0.035,               # 3.5%
    'TAUX_CONSO': 0.065,              # 6.5%
    'APPORT_MIN_RECOMMANDE': 0.10,    # 10%
    'SEUIL_IMMO': 75000,              # € seuil crédit immo
}

# ============================================================
# COLONNES POUR LE MACHINE LEARNING
# ============================================================

CAT_COLS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
]

NUM_COLS = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
    'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
    'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT'
]

# Colonnes numériques après feature engineering
NUM_COLS_UPDATED = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'OWN_CAR_AGE',
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
    'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
    'AGE_YEARS', 'EMPLOYED_YEARS', 'REGISTRATION_YEARS', 'ID_PUBLISH_YEARS'
]

# Features engineered
ENGINEERED_COLS = [
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO',
    'CREDIT_TERM_MONTHS', 'INCOME_MONTHLY', 'DEBT_RATIO', 'RESTE_A_VIVRE',
    'DUREE_PRET_YEARS', 'AGE_FIN_PRET'
]

# Toutes les features finales
FEATURES_FINAL = CAT_COLS + NUM_COLS_UPDATED + ENGINEERED_COLS

# ============================================================
# CHEMINS
# ============================================================

DATA_PATH = "data/application_train.csv"
MODEL_PATH = "models/credit_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
