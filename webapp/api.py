"""
API Flask pour le dashboard arboricole
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

app = Flask(__name__)
CORS(app)

# Variable globale pour les donnees
gdf = None


def load_data():
    """Charge et traite les donnees"""
    global gdf

    df = pd.read_csv("../donnees-defi-egc.csv", low_memory=False)
    df = df.replace("?", np.nan)

    for c in ["ANNEEDEPLANTATION", "ANNEEREALISATIONDIAGNOSTIC", "ANNEETRAVAUXPRECONISESDIAG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    gdf_temp = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["coord_x"], df["coord_y"])],
        crs="EPSG:3945"
    )

    gdf_wgs = gdf_temp.to_crs(4326)

    loc_cols = ["Collet", "Houppier", "Racine", "Tronc"]
    for c in loc_cols:
        if c in gdf_temp.columns:
            gdf_temp[c] = pd.to_numeric(gdf_temp[c], errors="coerce")

    gdf_temp["NB_LOCALISATIONS_DEFAUT"] = gdf_temp[loc_cols].fillna(0).sum(axis=1)
    gdf_temp["A_UN_DEFAUT_LOCALISE"] = (gdf_temp["NB_LOCALISATIONS_DEFAUT"] > 0).astype(int)

    ref_year = gdf_temp["ANNEEREALISATIONDIAGNOSTIC"].fillna(2025)
    gdf_temp["AGE_ESTIME"] = ref_year - gdf_temp["ANNEEDEPLANTATION"]

    def norm(s):
        return str(s).strip().lower()

    gdf_temp["VIGUEUR_N"] = gdf_temp["VIGUEUR"].map(norm) if "VIGUEUR" in gdf_temp.columns else ""
    gdf_temp["NOTE_N"] = gdf_temp["NOTEDIAGNOSTIC"].map(norm) if "NOTEDIAGNOSTIC" in gdf_temp.columns else ""
    gdf_temp["PRIO_N"] = gdf_temp["PRIORITEDERENOUVELLEMENT"].map(norm) if "PRIORITEDERENOUVELLEMENT" in gdf_temp.columns else ""

    gdf_temp["SCORE_URGENCE"] = 0

    if "DEFAUT" in gdf_temp.columns:
        gdf_temp["DEFAUT"] = pd.to_numeric(gdf_temp["DEFAUT"], errors="coerce")
        gdf_temp.loc[gdf_temp["DEFAUT"] == 1, "SCORE_URGENCE"] += 3

    gdf_temp["SCORE_URGENCE"] += gdf_temp["NB_LOCALISATIONS_DEFAUT"].clip(0, 3)

    if "VIGUEUR" in gdf_temp.columns:
        gdf_temp.loc[gdf_temp["VIGUEUR_N"].str.contains("faible", na=False), "SCORE_URGENCE"] += 2

    if "PRIORITEDERENOUVELLEMENT" in gdf_temp.columns:
        gdf_temp.loc[gdf_temp["PRIO_N"].str.contains("moins de 5|< ?5", na=False, regex=True), "SCORE_URGENCE"] += 2

    if "NOTEDIAGNOSTIC" in gdf_temp.columns:
        gdf_temp.loc[gdf_temp["NOTE_N"].str.contains("mauvais|danger|urgent", na=False), "SCORE_URGENCE"] += 2

    gdf_temp.loc[gdf_temp["AGE_ESTIME"] >= 60, "SCORE_URGENCE"] += 1
    gdf_temp.loc[gdf_temp["AGE_ESTIME"] >= 80, "SCORE_URGENCE"] += 1

    gdf_temp["lat"] = gdf_wgs.geometry.y
    gdf_temp["lon"] = gdf_wgs.geometry.x

    gdf = gdf_temp
    print(f"Donnees chargees: {len(gdf)} arbres")


@app.route('/api/kpis')
def get_kpis():
    """Retourne les KPIs principaux"""
    sector = request.args.get('sector', None)

    data = gdf if sector is None or sector == 'all' else gdf[gdf["ADR_SECTEUR"] == int(sector)]

    total = len(data)
    with_defect = int((data["DEFAUT"] == 1).sum())
    pct_defect = round(with_defect / total * 100, 1) if total > 0 else 0
    avg_score = round(data["SCORE_URGENCE"].mean(), 2)
    urgent_count = int((data["SCORE_URGENCE"] >= 4).sum())

    return jsonify({
        "total_arbres": total,
        "arbres_defaut": with_defect,
        "pct_defaut": pct_defect,
        "score_moyen": avg_score,
        "cas_urgents": urgent_count,
        "nb_secteurs": int(data["ADR_SECTEUR"].nunique())
    })


@app.route('/api/sectors')
def get_sectors():
    """Retourne la liste des secteurs"""
    sectors = sorted([int(s) for s in gdf["ADR_SECTEUR"].dropna().unique()])
    return jsonify(sectors)


@app.route('/api/sector-stats')
def get_sector_stats():
    """Retourne les stats par secteur"""
    grp = (gdf
           .groupby("ADR_SECTEUR", dropna=False)
           .agg(
                nb_arbres=("geometry", "size"),
                pct_defaut=("DEFAUT", lambda s: round(np.nanmean(s == 1) * 100, 1)),
                score_moy=("SCORE_URGENCE", lambda s: round(s.mean(), 2)),
                age_moy=("AGE_ESTIME", lambda s: round(s.mean(), 1)),
           )
           .sort_values("score_moy", ascending=False)
           .reset_index()
          )

    grp["ADR_SECTEUR"] = grp["ADR_SECTEUR"].astype(int)

    return jsonify(grp.to_dict(orient='records'))


@app.route('/api/work-stats')
def get_work_stats():
    """Retourne les stats par type de travaux"""
    limit = request.args.get('limit', 0, type=int)

    tab_trav = (gdf.groupby("TRAVAUXPRECONISESDIAG")
                .agg(
                    nb=("geometry", "size"),
                    score_moy=("SCORE_URGENCE", lambda s: round(s.mean(), 2)),
                    pct_defaut=("DEFAUT", lambda s: round(np.nanmean(s == 1) * 100, 1))
                )
                .sort_values("nb", ascending=False)
                .reset_index())

    if limit > 0:
        tab_trav = tab_trav.head(limit)

    return jsonify(tab_trav.to_dict(orient='records'))


@app.route('/api/priority-trees')
def get_priority_trees():
    """Retourne les arbres prioritaires"""
    limit = request.args.get('limit', 20, type=int)
    sector = request.args.get('sector', None)

    data = gdf if sector is None or sector == 'all' else gdf[gdf["ADR_SECTEUR"] == int(sector)]

    top_trees = (data
                 .sort_values("SCORE_URGENCE", ascending=False)
                 .head(limit)[["ADR_SECTEUR", "GENRE_BOTA", "ESPECE", "AGE_ESTIME", "DEFAUT",
                               "NB_LOCALISATIONS_DEFAUT", "TRAVAUXPRECONISESDIAG",
                               "SCORE_URGENCE", "lat", "lon"]])

    top_trees = top_trees.fillna("")
    top_trees["ADR_SECTEUR"] = top_trees["ADR_SECTEUR"].apply(lambda x: int(x) if x != "" else "")
    top_trees["AGE_ESTIME"] = top_trees["AGE_ESTIME"].apply(lambda x: int(x) if x != "" and not pd.isna(x) else "")
    top_trees["DEFAUT"] = top_trees["DEFAUT"].apply(lambda x: int(x) if x != "" and not pd.isna(x) else 0)
    top_trees["NB_LOCALISATIONS_DEFAUT"] = top_trees["NB_LOCALISATIONS_DEFAUT"].apply(lambda x: int(x) if x != "" else 0)
    top_trees["SCORE_URGENCE"] = top_trees["SCORE_URGENCE"].apply(lambda x: int(x) if x != "" else 0)

    return jsonify(top_trees.to_dict(orient='records'))


@app.route('/api/map-data')
def get_map_data():
    """Retourne les donnees pour la carte"""
    limit = request.args.get('limit', 1000, type=int)
    min_score = request.args.get('min_score', 0, type=int)
    sector = request.args.get('sector', None)

    data = gdf.copy()

    if sector and sector != 'all':
        data = data[data["ADR_SECTEUR"] == int(sector)]

    if min_score > 0:
        data = data[data["SCORE_URGENCE"] >= min_score]

    data = data.sort_values("SCORE_URGENCE", ascending=False).head(limit)

    result = data[["lat", "lon", "SCORE_URGENCE", "ADR_SECTEUR", "GENRE_BOTA",
                   "ESPECE", "TRAVAUXPRECONISESDIAG", "AGE_ESTIME"]].copy()

    result = result.fillna("")
    result["ADR_SECTEUR"] = result["ADR_SECTEUR"].apply(lambda x: int(x) if x != "" and not pd.isna(x) else "")
    result["SCORE_URGENCE"] = result["SCORE_URGENCE"].apply(lambda x: int(x) if not pd.isna(x) else 0)
    result["AGE_ESTIME"] = result["AGE_ESTIME"].apply(lambda x: int(x) if x != "" and not pd.isna(x) else "")

    return jsonify(result.to_dict(orient='records'))


@app.route('/api/comparison')
def get_comparison():
    """Retourne la comparaison global vs urgents"""
    q = gdf["SCORE_URGENCE"].quantile(0.90)
    high = gdf[gdf["SCORE_URGENCE"] >= q]

    global_dist = (gdf["TRAVAUXPRECONISESDIAG"].value_counts(normalize=True) * 100).head(8)
    high_dist = (high["TRAVAUXPRECONISESDIAG"].value_counts(normalize=True) * 100).head(8)

    result = []
    all_works = set(global_dist.index) | set(high_dist.index)

    for work in all_works:
        result.append({
            "travaux": work,
            "global_pct": round(global_dist.get(work, 0), 1),
            "urgent_pct": round(high_dist.get(work, 0), 1)
        })

    result.sort(key=lambda x: x["global_pct"], reverse=True)

    return jsonify(result)


if __name__ == '__main__':
    load_data()
    app.run(debug=True, port=5000)
