#!/usr/bin/env python3
"""
Interface en ligne de commande pour CreditScore Pro
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from pathlib import Path
import json

from src.model_trainer import load_model
from src.decision_engine import MoteurDecision
from src.calculator import CalculateurCredit
from src.config import MODEL_PATH

console = Console()
calc = CalculateurCredit()


def afficher_decision(resultat, dossier):
    """Affiche la décision de manière formatée."""

    decision = resultat['decision']
    d = resultat['details']

    # Couleur selon la décision
    if decision == "ACCEPTÉ":
        color = "green"
        emoji = "✅"
    elif decision == "ACCEPTÉ SOUS CONDITIONS":
        color = "yellow"
        emoji = "⚠️"
    else:
        color = "red"
        emoji = "❌"

    # Panel de décision
    console.print()
    console.print(Panel.fit(
        f"[bold {color}]{emoji} {decision}[/bold {color}]",
        title="🏦 Décision",
        border_style=color
    ))

    # Scores
    console.print()
    table_scores = Table(title="📊 Scores", box=box.ROUNDED)
    table_scores.add_column("Métrique", style="cyan")
    table_scores.add_column("Valeur", justify="right", style="magenta")

    table_scores.add_row("Score Final", f"{resultat['score_final']:.1f}/100")
    table_scores.add_row("Score Règles Métier", f"{resultat['score_metier']:.1f}/100")
    table_scores.add_row("Score ML", f"{resultat['score_ml']:.1f}/100")
    table_scores.add_row("Probabilité de défaut", f"{resultat['proba_defaut']*100:.1f}%")

    console.print(table_scores)

    # Détails financiers
    console.print()
    table_finance = Table(title="💰 Détails Financiers", box=box.ROUNDED)
    table_finance.add_column("Élément", style="cyan")
    table_finance.add_column("Valeur", justify="right", style="green")

    table_finance.add_row("Type de crédit", d['type_credit'].title())
    table_finance.add_row("Taux d'intérêt", f"{d['taux']*100:.2f}%")
    table_finance.add_row("Mensualité", f"{d['mensualite']:,.2f}€")
    table_finance.add_row("Coût total", f"{d['cout_total']:,.2f}€")
    table_finance.add_row("Intérêts", f"{d['interets']:,.2f}€")
    table_finance.add_row("", "")
    table_finance.add_row("Taux d'endettement", f"{d['taux_endettement']*100:.1f}%")
    table_finance.add_row("Reste à vivre", f"{d['reste_a_vivre']:,.0f}€/mois")
    table_finance.add_row("Capacité max", f"{d['capacite_max']:,.0f}€")
    table_finance.add_row("Âge fin de prêt", f"{d['age_fin_pret']:.0f} ans")

    console.print(table_finance)

    # Points forts
    if resultat['points_forts']:
        console.print()
        console.print("[bold green]✅ Points forts:[/bold green]")
        for point in resultat['points_forts']:
            console.print(f"  • {point}")

    # Alertes
    if resultat['alertes']:
        console.print()
        console.print("[bold yellow]⚠️  Points d'attention:[/bold yellow]")
        for severity, msg in resultat['alertes']:
            color = "red" if severity == "danger" else "yellow"
            console.print(f"  • [{color}]{msg}[/{color}]")

    # Raison de refus
    if resultat['raison_refus']:
        console.print()
        console.print(f"[bold red]❌ Raison du refus: {resultat['raison_refus']}[/bold red]")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🏦 CreditScore Pro - Analyse de crédit intelligente"""
    pass


@cli.command()
@click.option('--revenu', '-r', type=float, required=True, help="Revenu annuel (€)")
@click.option('--montant', '-m', type=float, required=True, help="Montant du crédit (€)")
@click.option('--duree', '-d', type=int, required=True, help="Durée (années)")
@click.option('--age', '-a', type=int, required=True, help="Âge du demandeur")
@click.option('--anciennete', type=float, default=0, help="Ancienneté emploi (années)")
@click.option('--enfants', type=int, default=0, help="Nombre d'enfants")
@click.option('--charges', type=float, default=0, help="Charges mensuelles (€)")
@click.option('--apport', type=float, default=0, help="Apport personnel (€)")
@click.option('--json-output', is_flag=True, help="Sortie en format JSON")
def analyser(revenu, montant, duree, age, anciennete, enfants, charges, apport, json_output):
    """Analyse une demande de crédit."""

    dossier = {
        'revenu_annuel': revenu,
        'montant_credit': montant,
        'duree_annees': duree,
        'age': age,
        'anciennete_emploi': anciennete,
        'nb_enfants': enfants,
        'charges_existantes': charges,
        'apport': apport
    }

    # Chargement du modèle
    try:
        if not json_output:
            console.print("⏳ Chargement du modèle...", style="blue")

        model_path = Path(MODEL_PATH)
        if model_path.exists():
            model, features, _ = load_model(str(model_path))
            moteur = MoteurDecision(model, features)
            if not json_output:
                console.print("✅ Modèle chargé", style="green")
        else:
            moteur = MoteurDecision(model=None)
            if not json_output:
                console.print("⚠️  Modèle non trouvé, utilisation des règles métier uniquement",
                            style="yellow")

    except Exception as e:
        console.print(f"❌ Erreur lors du chargement: {e}", style="red")
        return

    # Analyse
    try:
        if not json_output:
            console.print("🔍 Analyse en cours...", style="blue")

        resultat = moteur.analyser(dossier)

        if json_output:
            # Sortie JSON
            print(json.dumps(resultat, indent=2, ensure_ascii=False))
        else:
            # Affichage formaté
            afficher_decision(resultat, dossier)

    except Exception as e:
        console.print(f"❌ Erreur lors de l'analyse: {e}", style="red")


@cli.command()
@click.option('--capital', '-c', type=float, required=True, help="Capital emprunté (€)")
@click.option('--taux', '-t', type=float, required=True, help="Taux annuel (ex: 0.035)")
@click.option('--duree', '-d', type=int, required=True, help="Durée (années)")
def mensualite(capital, taux, duree):
    """Calcule la mensualité d'un prêt."""

    mens = calc.mensualite(capital, taux, duree)
    cout_total, interets = calc.cout_total(capital, taux, duree)

    console.print()
    table = Table(title="💰 Calcul de mensualité", box=box.ROUNDED)
    table.add_column("Élément", style="cyan")
    table.add_column("Valeur", justify="right", style="green")

    table.add_row("Capital emprunté", f"{capital:,.2f}€")
    table.add_row("Taux annuel", f"{taux*100:.2f}%")
    table.add_row("Durée", f"{duree} ans")
    table.add_row("", "")
    table.add_row("[bold]Mensualité[/bold]", f"[bold]{mens:,.2f}€[/bold]")
    table.add_row("Coût total", f"{cout_total:,.2f}€")
    table.add_row("Intérêts totaux", f"{interets:,.2f}€")

    console.print(table)
    console.print()


@cli.command()
@click.option('--revenu', '-r', type=float, required=True, help="Revenu mensuel (€)")
@click.option('--taux', '-t', type=float, required=True, help="Taux annuel (ex: 0.035)")
@click.option('--duree', '-d', type=int, required=True, help="Durée (années)")
@click.option('--charges', type=float, default=0, help="Charges mensuelles (€)")
def capacite(revenu, taux, duree, charges):
    """Calcule la capacité d'emprunt."""

    cap = calc.capacite_emprunt(revenu, taux, duree, charges)

    console.print()
    table = Table(title="🎯 Capacité d'emprunt", box=box.ROUNDED)
    table.add_column("Élément", style="cyan")
    table.add_column("Valeur", justify="right", style="green")

    table.add_row("Revenu mensuel", f"{revenu:,.2f}€")
    table.add_row("Taux annuel", f"{taux*100:.2f}%")
    table.add_row("Durée", f"{duree} ans")
    table.add_row("Charges existantes", f"{charges:,.2f}€")
    table.add_row("", "")
    table.add_row("[bold]Capacité maximale[/bold]", f"[bold]{cap:,.2f}€[/bold]")

    console.print(table)
    console.print()


if __name__ == '__main__':
    cli()
