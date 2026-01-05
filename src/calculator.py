"""
Calculateur de crédit - Toutes les fonctions de calcul financier
"""

import pandas as pd
from .config import CONFIG


class CalculateurCredit:
    """Classe utilitaire pour tous les calculs de crédit."""

    @staticmethod
    def mensualite(capital: float, taux_annuel: float, duree_annees: int) -> float:
        """
        Calcule la mensualité d'un prêt.

        Args:
            capital: Montant emprunté
            taux_annuel: Taux d'intérêt annuel (ex: 0.035 pour 3.5%)
            duree_annees: Durée du prêt en années

        Returns:
            Mensualité en euros
        """
        if taux_annuel == 0:
            return capital / (duree_annees * 12)

        taux_mensuel = taux_annuel / 12
        nb_mois = duree_annees * 12

        return capital * (taux_mensuel * (1 + taux_mensuel)**nb_mois) / \
               ((1 + taux_mensuel)**nb_mois - 1)

    @staticmethod
    def cout_total(capital: float, taux_annuel: float, duree_annees: int) -> tuple:
        """
        Calcule le coût total et les intérêts.

        Returns:
            (cout_total, interets)
        """
        mens = CalculateurCredit.mensualite(capital, taux_annuel, duree_annees)
        total = mens * duree_annees * 12
        return total, total - capital

    @staticmethod
    def taux_endettement(mensualite: float, revenu_mensuel: float) -> float:
        """Calcule le taux d'endettement."""
        if revenu_mensuel <= 0:
            return float('inf')
        return mensualite / revenu_mensuel

    @staticmethod
    def capacite_emprunt(revenu_mensuel: float, taux_annuel: float,
                        duree_annees: int, charges: float = 0) -> float:
        """
        Calcule la capacité d'emprunt maximale.

        Args:
            revenu_mensuel: Revenu mensuel net
            taux_annuel: Taux d'intérêt annuel
            duree_annees: Durée souhaitée
            charges: Charges mensuelles existantes

        Returns:
            Capacité d'emprunt maximale
        """
        mensualite_max = (revenu_mensuel * CONFIG['MAX_DEBT_RATIO']) - charges

        if mensualite_max <= 0:
            return 0

        taux_mensuel = taux_annuel / 12
        nb_mois = duree_annees * 12

        if taux_annuel == 0:
            return mensualite_max * nb_mois

        return mensualite_max * ((1 + taux_mensuel)**nb_mois - 1) / \
               (taux_mensuel * (1 + taux_mensuel)**nb_mois)

    @staticmethod
    def type_credit(montant: float) -> str:
        """Détermine le type de crédit."""
        return "immobilier" if montant >= CONFIG['SEUIL_IMMO'] else "consommation"

    @staticmethod
    def taux_interet(montant: float) -> float:
        """Retourne le taux selon le type de crédit."""
        return CONFIG['TAUX_IMMO'] if montant >= CONFIG['SEUIL_IMMO'] else CONFIG['TAUX_CONSO']

    @staticmethod
    def tableau_amortissement(capital: float, taux_annuel: float,
                             duree_annees: int) -> pd.DataFrame:
        """
        Génère le tableau d'amortissement annuel.

        Returns:
            DataFrame avec colonnes: annee, capital_rembourse, interets, solde_restant
        """
        mensualite = CalculateurCredit.mensualite(capital, taux_annuel, duree_annees)
        taux_mensuel = taux_annuel / 12
        solde = capital
        tableau = []

        for annee in range(1, duree_annees + 1):
            interets_annee = 0
            capital_rembourse = 0

            for mois in range(12):
                if solde <= 0:
                    break

                interet = solde * taux_mensuel
                principal = mensualite - interet
                solde -= principal
                interets_annee += interet
                capital_rembourse += principal

            tableau.append({
                'annee': annee,
                'capital_rembourse': capital_rembourse,
                'interets': interets_annee,
                'solde_restant': max(0, solde)
            })

        return pd.DataFrame(tableau)
