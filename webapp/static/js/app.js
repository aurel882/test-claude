// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Elements
const form = document.getElementById('creditForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoader = submitBtn.querySelector('.btn-loader');
const resultsSection = document.getElementById('resultsSection');

// Formatage des nombres
function formatNumber(num, decimals = 0) {
    return new Intl.NumberFormat('fr-FR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(num);
}

function formatCurrency(num) {
    return formatNumber(num, 0) + ' €';
}

function formatPercent(num) {
    return formatNumber(num * 100, 1) + '%';
}

// Update slider value display
document.getElementById('duree').addEventListener('input', (e) => {
    document.getElementById('dureeValue').textContent = e.target.value;
});

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Show loading state
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'flex';
    resultsSection.style.display = 'none';

    // Collect form data
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value) || 0;
    });

    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/analyser`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Erreur lors de l\'analyse');
        }

        const result = await response.json();

        // Display results
        displayResults(result, data);

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

    } catch (error) {
        console.error('Error:', error);
        alert('Une erreur est survenue lors de l\'analyse. Veuillez réessayer.');
    } finally {
        // Hide loading state
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

function displayResults(result, inputData) {
    const { decision, score_final, details, alertes, points_forts, raison_refus } = result;

    // Decision badge
    const decisionBadge = document.getElementById('decisionBadge');
    decisionBadge.textContent = decision;

    if (decision === 'ACCEPTÉ') {
        decisionBadge.className = 'decision-badge decision-accepted';
    } else if (decision.includes('CONDITIONS')) {
        decisionBadge.className = 'decision-badge decision-conditional';
    } else {
        decisionBadge.className = 'decision-badge decision-refused';
    }

    // Metrics
    document.getElementById('mensualite').textContent = formatCurrency(details.mensualite);
    document.getElementById('taux').textContent = formatPercent(details.taux);
    document.getElementById('score').textContent = score_final.toFixed(0) + '/100';
    document.getElementById('endettement').textContent = formatPercent(details.taux_endettement);

    // Details
    const detailsGrid = document.getElementById('detailsGrid');
    detailsGrid.innerHTML = '';

    const detailsToShow = [
        { label: 'Type de crédit', value: details.type_credit, format: 'text' },
        { label: 'Coût total', value: details.cout_total, format: 'currency' },
        { label: 'Intérêts totaux', value: details.interets, format: 'currency' },
        { label: 'Reste à vivre', value: details.reste_a_vivre, format: 'currency' },
        { label: 'Capacité maximale', value: details.capacite_max, format: 'currency' },
        { label: 'Âge fin de prêt', value: details.age_fin_pret, format: 'years' }
    ];

    detailsToShow.forEach(detail => {
        const item = document.createElement('div');
        item.className = 'detail-item';

        let formattedValue;
        if (detail.format === 'currency') {
            formattedValue = formatCurrency(detail.value);
        } else if (detail.format === 'percent') {
            formattedValue = formatPercent(detail.value);
        } else if (detail.format === 'years') {
            formattedValue = detail.value.toFixed(0) + ' ans';
        } else {
            formattedValue = detail.value;
        }

        item.innerHTML = `
            <span class="detail-label">${detail.label}</span>
            <span class="detail-value">${formattedValue}</span>
        `;
        detailsGrid.appendChild(item);
    });

    // Alerts section
    const alertsSection = document.getElementById('alertsSection');
    alertsSection.innerHTML = '';

    // Points forts
    if (points_forts && points_forts.length > 0) {
        points_forts.forEach(point => {
            const alert = createAlert('success', '✅', point);
            alertsSection.appendChild(alert);
        });
    }

    // Alertes
    if (alertes && alertes.length > 0) {
        alertes.forEach(([severity, message]) => {
            const icon = severity === 'danger' ? '⛔' : '⚠️';
            const alert = createAlert(severity, icon, message);
            alertsSection.appendChild(alert);
        });
    }

    // Raison de refus
    if (raison_refus) {
        const alert = createAlert('danger', '❌', `Raison du refus: ${raison_refus}`);
        alertsSection.appendChild(alert);
    }

    // Recommendations
    const recommendationsSection = document.getElementById('recommendationsSection');
    recommendationsSection.innerHTML = '';

    const recTitle = document.createElement('h4');
    recTitle.innerHTML = '💡 Recommandations personnalisées';
    recommendationsSection.appendChild(recTitle);

    const recommendations = getRecommendations(result, inputData);
    const recList = document.createElement('ul');
    recList.style.marginLeft = '1.5rem';
    recList.style.marginTop = '1rem';

    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        li.style.marginBottom = '0.75rem';
        recList.appendChild(li);
    });

    recommendationsSection.appendChild(recList);

    // Capacity highlight
    const capacityDiv = document.createElement('div');
    capacityDiv.className = 'capacity-highlight';
    capacityDiv.innerHTML = `
        <div class="label">Votre capacité d'emprunt maximale</div>
        <div class="value">${formatCurrency(details.capacite_max)}</div>
        <div class="subtext">sur ${inputData.duree_annees} ans à ${formatPercent(details.taux)}</div>
    `;
    recommendationsSection.appendChild(capacityDiv);

    // Show results
    resultsSection.style.display = 'block';
}

function createAlert(type, icon, message) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <div class="alert-icon">${icon}</div>
        <div>${message}</div>
    `;
    return alert;
}

function getRecommendations(result, inputData) {
    const { decision, details, score_final } = result;
    const recommendations = [];

    if (decision === 'ACCEPTÉ') {
        recommendations.push('Excellent profil ! N\'hésitez pas à négocier le taux auprès de plusieurs banques.');
        recommendations.push('Vous êtes en position de force pour obtenir les meilleures conditions.');

        if (details.taux_endettement < 0.30) {
            recommendations.push('Votre taux d\'endettement est très confortable, vous pourriez emprunter davantage si nécessaire.');
        }
    } else if (decision.includes('CONDITIONS')) {
        recommendations.push('Votre dossier est acceptable mais nécessite quelques améliorations.');

        if (details.taux_endettement > 0.33) {
            recommendations.push('Réduisez vos charges existantes ou augmentez votre apport pour améliorer votre taux d\'endettement.');
        }

        if (inputData.anciennete_emploi < 2) {
            recommendations.push('Stabilisez votre situation professionnelle pour renforcer votre dossier.');
        }

        if (inputData.apport < inputData.montant_credit * 0.10) {
            recommendations.push('Augmentez votre apport personnel (idéalement 10% minimum).');
        }

        recommendations.push('Une assurance emprunteur pourra être exigée.');
    } else {
        recommendations.push('Votre demande nécessite des ajustements importants.');

        if (inputData.montant_credit > details.capacite_max) {
            recommendations.push(`Réduisez le montant emprunté à ${formatCurrency(details.capacite_max * 0.9)} maximum.`);
        }

        if (details.taux_endettement > 0.40) {
            recommendations.push('Remboursez vos crédits en cours avant de faire une nouvelle demande.');
        }

        if (inputData.duree_annees < 20 && details.type_credit === 'immobilier') {
            recommendations.push('Allongez la durée du prêt pour réduire les mensualités.');
        }

        if (inputData.apport < inputData.montant_credit * 0.10) {
            recommendations.push('Constituez un apport personnel plus important.');
        }

        recommendations.push('Attendez une amélioration de vos revenus ou une diminution de vos charges.');
    }

    return recommendations;
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            console.warn('API not responding');
        }
    } catch (error) {
        console.error('Cannot connect to API:', error);
        alert('⚠️ Impossible de se connecter à l\'API. Assurez-vous que le serveur est démarré (python api/main.py)');
    }
});
