# ArboVision - Dashboard Patrimoine Arboricole

Application web moderne pour la gestion et la visualisation du patrimoine arboricole.

## Installation

```bash
cd webapp
pip install -r requirements.txt
```

## Lancement

1. Demarrer le serveur API Flask:
```bash
python api.py
```

2. Ouvrir `index.html` dans un navigateur (ou utiliser un serveur local):
```bash
# Option 1: Python simple server
python -m http.server 8080

# Option 2: Ouvrir directement index.html dans le navigateur
```

3. Acceder a l'application: http://localhost:8080

## Fonctionnalites

- **Tableau de bord**: KPIs, graphiques par secteur et travaux
- **Carte interactive**: Visualisation geographique avec clusters et filtres
- **Analyse Secteurs**: Tableau comparatif et graphiques par secteur
- **Analyse Travaux**: Distribution des travaux et comparaison global/urgents
- **Arbres Prioritaires**: Liste des arbres necessitant une intervention

## Stack technique

- Frontend: Vue.js 3, Chart.js, Leaflet
- Backend: Flask, Pandas, GeoPandas
