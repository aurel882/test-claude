"""
Schémas Pydantic pour l'API
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class DossierCredit(BaseModel):
    """Modèle pour une demande de crédit."""

    revenu_annuel: float = Field(
        ...,
        gt=0,
        description="Revenu annuel net en euros",
        example=50000
    )
    montant_credit: float = Field(
        ...,
        gt=0,
        description="Montant du crédit demandé en euros",
        example=200000
    )
    duree_annees: int = Field(
        ...,
        ge=1,
        le=25,
        description="Durée du prêt en années",
        example=20
    )
    age: int = Field(
        ...,
        ge=18,
        le=75,
        description="Âge du demandeur",
        example=35
    )
    anciennete_emploi: float = Field(
        default=0,
        ge=0,
        description="Ancienneté dans l'emploi actuel en années",
        example=5.0
    )
    nb_enfants: int = Field(
        default=0,
        ge=0,
        description="Nombre d'enfants à charge",
        example=2
    )
    charges_existantes: float = Field(
        default=0,
        ge=0,
        description="Charges mensuelles existantes en euros",
        example=500
    )
    apport: float = Field(
        default=0,
        ge=0,
        description="Apport personnel en euros",
        example=20000
    )

    class Config:
        json_schema_extra = {
            "example": {
                "revenu_annuel": 50000,
                "montant_credit": 200000,
                "duree_annees": 20,
                "age": 35,
                "anciennete_emploi": 5.0,
                "nb_enfants": 2,
                "charges_existantes": 500,
                "apport": 20000
            }
        }


class DetailsFinanciers(BaseModel):
    """Détails financiers du crédit."""

    type_credit: str
    taux: float
    mensualite: float
    taux_endettement: float
    reste_a_vivre: float
    cout_total: float
    interets: float
    capacite_max: float
    age_fin_pret: float


class ResultatAnalyse(BaseModel):
    """Résultat de l'analyse de crédit."""

    decision: str
    score_final: float
    score_metier: float
    score_ml: float
    proba_defaut: float
    alertes: List[Tuple[str, str]]
    points_forts: List[str]
    refus_auto: bool
    raison_refus: Optional[str]
    details: DetailsFinanciers


class HealthCheck(BaseModel):
    """Health check de l'API."""

    status: str
    version: str
    model_loaded: bool
