"""
Moteur de décision crédit hybride (ML + règles métier)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from .calculator import CalculateurCredit
from .config import CONFIG, FEATURES_FINAL


class MoteurDecision:
    """Moteur de décision crédit hybride (ML + règles métier)."""

    def __init__(self, model, features_list: List[str] = None):
        """
        Initialise le moteur de décision.

        Args:
            model: Modèle ML entraîné (pipeline sklearn)
            features_list: Liste des features attendues par le modèle
        """
        self.model = model
        self.features = features_list or FEATURES_FINAL
        self.calc = CalculateurCredit()

    def analyser(self, dossier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse complète d'un dossier de crédit.

        Args:
            dossier: Dictionnaire contenant les informations du demandeur
                - revenu_annuel: Revenu annuel en €
                - montant_credit: Montant demandé en €
                - duree_annees: Durée du prêt en années
                - age: Âge du demandeur
                - anciennete_emploi: Ancienneté dans l'emploi actuel (années)
                - nb_enfants: Nombre d'enfants à charge
                - charges_existantes: Charges mensuelles existantes en €
                - apport: Apport personnel en €

        Returns:
            Dictionnaire contenant la décision et tous les détails
        """
        # Extraction des données
        revenu_annuel = dossier.get('revenu_annuel', 0)
        revenu_mensuel = revenu_annuel / 12
        montant = dossier.get('montant_credit', 0)
        duree = dossier.get('duree_annees', 20)
        age = dossier.get('age', 30)
        anciennete = dossier.get('anciennete_emploi', 0)
        nb_enfants = dossier.get('nb_enfants', 0)
        charges = dossier.get('charges_existantes', 0)
        apport = dossier.get('apport', 0)

        # Calculs financiers
        type_credit = self.calc.type_credit(montant)
        taux = self.calc.taux_interet(montant)
        mensualite = self.calc.mensualite(montant, taux, duree)
        mensualite_totale = mensualite + charges
        taux_endettement = self.calc.taux_endettement(mensualite_totale, revenu_mensuel)
        reste_a_vivre = revenu_mensuel - mensualite_totale
        cout_total, interets = self.calc.cout_total(montant, taux, duree)
        capacite_max = self.calc.capacite_emprunt(revenu_mensuel, taux, duree, charges)
        age_fin_pret = age + duree

        # Score règles métier
        score_metier = 100
        alertes = []
        points_forts = []

        # Règle 1: Taux d'endettement
        if taux_endettement > 0.50:
            score_metier -= 40
            alertes.append(('danger', f"Taux d'endettement critique: {taux_endettement*100:.1f}%"))
        elif taux_endettement > CONFIG['MAX_DEBT_RATIO']:
            score_metier -= 25
            alertes.append(('warning', f"Taux d'endettement élevé: {taux_endettement*100:.1f}% (max {CONFIG['MAX_DEBT_RATIO']*100}%)"))
        elif taux_endettement <= 0.25:
            score_metier += 10
            points_forts.append(f"Excellent taux d'endettement: {taux_endettement*100:.1f}%")
        elif taux_endettement <= 0.33:
            points_forts.append(f"Bon taux d'endettement: {taux_endettement*100:.1f}%")

        # Règle 2: Reste à vivre
        seuil_rav = CONFIG['MIN_RESTE_A_VIVRE'] + (nb_enfants * CONFIG['MIN_RESTE_A_VIVRE_ENFANT'])
        if reste_a_vivre < 400:
            score_metier -= 35
            alertes.append(('danger', f"Reste à vivre insuffisant: {reste_a_vivre:.0f}€"))
        elif reste_a_vivre < seuil_rav:
            score_metier -= 20
            alertes.append(('warning', f"Reste à vivre limite: {reste_a_vivre:.0f}€ (recommandé: {seuil_rav}€)"))
        elif reste_a_vivre > seuil_rav * 2:
            score_metier += 10
            points_forts.append(f"Excellent reste à vivre: {reste_a_vivre:,.0f}€")

        # Règle 3: Âge
        if age < CONFIG['MIN_AGE']:
            score_metier -= 50
            alertes.append(('danger', f"Âge insuffisant: {age} ans"))
        if age_fin_pret > CONFIG['MAX_AGE_FIN_PRET']:
            score_metier -= 15
            alertes.append(('warning', f"Âge en fin de prêt élevé: {age_fin_pret} ans"))

        # Règle 4: Ancienneté emploi
        if anciennete < 0.5:
            score_metier -= 15
            alertes.append(('warning', f"Ancienneté emploi faible: {anciennete:.1f} ans"))
        elif anciennete >= 5:
            score_metier += 10
            points_forts.append(f"Excellente stabilité professionnelle: {anciennete:.0f} ans")
        elif anciennete >= 2:
            points_forts.append(f"Bonne ancienneté: {anciennete:.1f} ans")

        # Règle 5: Apport (immobilier)
        if type_credit == "immobilier":
            taux_apport = apport / (montant + apport) if (montant + apport) > 0 else 0
            if taux_apport >= 0.20:
                score_metier += 10
                points_forts.append(f"Apport conséquent: {taux_apport*100:.0f}%")
            elif taux_apport < CONFIG['APPORT_MIN_RECOMMANDE']:
                score_metier -= 10
                alertes.append(('warning', f"Apport faible: {taux_apport*100:.1f}%"))

        # Score ML
        proba_defaut = self._score_ml(dossier, mensualite, duree)
        score_ml = (1 - proba_defaut) * 100

        # Score final (60% métier, 40% ML)
        score_metier = max(0, min(100, score_metier))
        score_final = 0.6 * score_metier + 0.4 * score_ml

        # Décision
        refus_auto = False
        raison_refus = None

        if taux_endettement > 0.50:
            refus_auto = True
            raison_refus = "Taux d'endettement excessif"
        elif reste_a_vivre < 400:
            refus_auto = True
            raison_refus = "Reste à vivre insuffisant"
        elif age < CONFIG['MIN_AGE']:
            refus_auto = True
            raison_refus = "Âge minimum non atteint"

        if refus_auto:
            decision = "REFUSÉ"
        elif score_final >= 70:
            decision = "ACCEPTÉ"
        elif score_final >= 50:
            decision = "ACCEPTÉ SOUS CONDITIONS"
        else:
            decision = "REFUSÉ"

        return {
            'decision': decision,
            'score_final': score_final,
            'score_metier': score_metier,
            'score_ml': score_ml,
            'proba_defaut': proba_defaut,
            'alertes': alertes,
            'points_forts': points_forts,
            'refus_auto': refus_auto,
            'raison_refus': raison_refus,
            'details': {
                'type_credit': type_credit,
                'taux': taux,
                'mensualite': mensualite,
                'taux_endettement': taux_endettement,
                'reste_a_vivre': reste_a_vivre,
                'cout_total': cout_total,
                'interets': interets,
                'capacite_max': capacite_max,
                'age_fin_pret': age_fin_pret
            }
        }

    def _score_ml(self, dossier: Dict[str, Any], mensualite: float, duree: int) -> float:
        """
        Calcule le score ML (probabilité de défaut).

        Args:
            dossier: Informations du demandeur
            mensualite: Mensualité calculée
            duree: Durée du prêt

        Returns:
            Probabilité de défaut (entre 0 et 1)
        """
        # Création d'une ligne avec toutes les features
        row = {c: np.nan for c in self.features}

        # Features de base
        row['AMT_INCOME_TOTAL'] = dossier.get('revenu_annuel', np.nan)
        row['AMT_CREDIT'] = dossier.get('montant_credit', np.nan)
        row['AGE_YEARS'] = dossier.get('age', np.nan)
        row['EMPLOYED_YEARS'] = dossier.get('anciennete_emploi', np.nan)
        row['CNT_CHILDREN'] = dossier.get('nb_enfants', 0)
        row['CNT_FAM_MEMBERS'] = dossier.get('nb_enfants', 0) + 1
        row['AMT_ANNUITY'] = mensualite * 12

        # Features calculées
        revenu = dossier.get('revenu_annuel', 1)
        montant = dossier.get('montant_credit', 0)

        row['CREDIT_INCOME_RATIO'] = montant / revenu if revenu > 0 else np.nan
        row['ANNUITY_INCOME_RATIO'] = (mensualite * 12) / revenu if revenu > 0 else np.nan
        row['INCOME_MONTHLY'] = revenu / 12
        row['DEBT_RATIO'] = mensualite / (revenu / 12) if revenu > 0 else np.nan
        row['RESTE_A_VIVRE'] = (revenu / 12) - mensualite
        row['DUREE_PRET_YEARS'] = duree
        row['AGE_FIN_PRET'] = dossier.get('age', 30) + duree
        row['CREDIT_TERM_MONTHS'] = duree * 12

        # Prédiction
        X_pred = pd.DataFrame([row], columns=self.features)

        try:
            proba = float(self.model.predict_proba(X_pred)[:, 1][0])
        except:
            # Si le modèle n'est pas disponible, retourne une valeur neutre
            proba = 0.5

        return proba
