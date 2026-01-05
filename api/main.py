"""
API REST FastAPI pour CreditScore Pro
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import sys

# Ajout du chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_trainer import load_model
from src.decision_engine import MoteurDecision
from src.calculator import CalculateurCredit
from src.config import MODEL_PATH
from api.schemas import DossierCredit, ResultatAnalyse, HealthCheck

# ============================================================
# INITIALISATION
# ============================================================

app = FastAPI(
    title="CreditScore Pro API",
    description="API d'analyse de crédit avec ML et règles métier",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
moteur = None
calc = CalculateurCredit()

# Chemins
BASE_DIR = Path(__file__).parent.parent
WEBAPP_DIR = BASE_DIR / "webapp"

# Montage des fichiers statiques
if WEBAPP_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage."""
    global moteur

    try:
        print("🚀 Démarrage de l'API CreditScore Pro...")
        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            print("⚠️  Modèle non trouvé. L'API fonctionnera en mode règles métier uniquement.")
            # Créer un moteur sans modèle (utilisera des valeurs par défaut)
            moteur = MoteurDecision(model=None)
        else:
            print(f"📦 Chargement du modèle depuis {model_path}...")
            model, features, metadata = load_model(str(model_path))
            moteur = MoteurDecision(model, features)
            print(f"✅ Modèle chargé (AUC Test: {metadata.get('auc_test', 'N/A'):.4f})")

        print("✅ API prête à recevoir des requêtes")

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("⚠️  L'API démarrera en mode dégradé (règles métier uniquement)")
        moteur = MoteurDecision(model=None)


@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage au shutdown."""
    print("👋 Arrêt de l'API CreditScore Pro")


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Page d'accueil - Interface web."""
    index_path = WEBAPP_DIR / "templates" / "index.html"

    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return {
            "message": "Bienvenue sur CreditScore Pro API",
            "version": "1.0.0",
            "documentation": "/docs",
            "web_interface": "Interface web non disponible",
            "endpoints": {
                "health": "/health",
                "analyser": "/analyser (POST)",
                "calculer_mensualite": "/calculer/mensualite",
                "calculer_capacite": "/calculer/capacite"
            }
        }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": moteur is not None and moteur.model is not None
    }


@app.post("/analyser", response_model=ResultatAnalyse, tags=["Analyse"])
async def analyser_credit(dossier: DossierCredit):
    """
    Analyse complète d'une demande de crédit.

    Retourne une décision (ACCEPTÉ / ACCEPTÉ SOUS CONDITIONS / REFUSÉ)
    avec tous les détails financiers et les scores.
    """
    if moteur is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporairement indisponible. Modèle non chargé."
        )

    try:
        resultat = moteur.analyser(dossier.model_dump())
        return resultat

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )


@app.get("/calculer/mensualite", tags=["Calculateurs"])
async def calculer_mensualite(
    capital: float,
    taux_annuel: float,
    duree_annees: int
):
    """
    Calcule la mensualité d'un prêt.

    Args:
        capital: Montant emprunté
        taux_annuel: Taux d'intérêt annuel (ex: 0.035 pour 3.5%)
        duree_annees: Durée en années
    """
    try:
        mensualite = calc.mensualite(capital, taux_annuel, duree_annees)
        cout_total, interets = calc.cout_total(capital, taux_annuel, duree_annees)

        return {
            "mensualite": round(mensualite, 2),
            "cout_total": round(cout_total, 2),
            "interets": round(interets, 2),
            "capital": capital,
            "taux_annuel": taux_annuel,
            "duree_annees": duree_annees
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de calcul: {str(e)}"
        )


@app.get("/calculer/capacite", tags=["Calculateurs"])
async def calculer_capacite(
    revenu_mensuel: float,
    taux_annuel: float,
    duree_annees: int,
    charges: float = 0
):
    """
    Calcule la capacité d'emprunt maximale.

    Args:
        revenu_mensuel: Revenu mensuel net
        taux_annuel: Taux d'intérêt annuel
        duree_annees: Durée souhaitée
        charges: Charges mensuelles existantes (optionnel)
    """
    try:
        capacite = calc.capacite_emprunt(revenu_mensuel, taux_annuel, duree_annees, charges)

        return {
            "capacite_emprunt": round(capacite, 2),
            "revenu_mensuel": revenu_mensuel,
            "taux_annuel": taux_annuel,
            "duree_annees": duree_annees,
            "charges": charges
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de calcul: {str(e)}"
        )


@app.get("/calculer/tableau-amortissement", tags=["Calculateurs"])
async def tableau_amortissement(
    capital: float,
    taux_annuel: float,
    duree_annees: int
):
    """
    Génère le tableau d'amortissement annuel.
    """
    try:
        tableau = calc.tableau_amortissement(capital, taux_annuel, duree_annees)
        return tableau.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de calcul: {str(e)}"
        )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
