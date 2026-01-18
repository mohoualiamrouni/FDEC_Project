"""
Application de gestion du patrimoine arboricole
Dashboard interactif pour l'analyse des arbres et la priorisation des interventions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import Point

# Configuration de la page
st.set_page_config(
    page_title="Gestion Patrimoine Arboricole",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c7d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a5f;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data():
    """Charge et traite les donnees"""
    df = pd.read_csv("donnees-defi-egc.csv", low_memory=False)
    df = df.replace("?", np.nan)

    # Conversions numeriques
    for c in ["ANNEEDEPLANTATION", "ANNEEREALISATIONDIAGNOSTIC", "ANNEETRAVAUXPRECONISESDIAG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Creation du GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["coord_x"], df["coord_y"])],
        crs="EPSG:3945"
    )

    # Conversion WGS84 pour affichage carte
    gdf_wgs = gdf.to_crs(4326)

    # Colonnes de localisation des defauts
    loc_cols = ["Collet", "Houppier", "Racine", "Tronc"]
    for c in loc_cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

    # Calcul des indicateurs
    gdf["NB_LOCALISATIONS_DEFAUT"] = gdf[loc_cols].fillna(0).sum(axis=1)
    gdf["A_UN_DEFAUT_LOCALISE"] = (gdf["NB_LOCALISATIONS_DEFAUT"] > 0).astype(int)

    # Age estime
    ref_year = gdf["ANNEEREALISATIONDIAGNOSTIC"].fillna(2025)
    gdf["AGE_ESTIME"] = ref_year - gdf["ANNEEDEPLANTATION"]

    # Normalisation des textes
    def norm(s):
        return str(s).strip().lower()

    gdf["VIGUEUR_N"] = gdf["VIGUEUR"].map(norm) if "VIGUEUR" in gdf.columns else ""
    gdf["NOTE_N"] = gdf["NOTEDIAGNOSTIC"].map(norm) if "NOTEDIAGNOSTIC" in gdf.columns else ""
    gdf["PRIO_N"] = gdf["PRIORITEDERENOUVELLEMENT"].map(norm) if "PRIORITEDERENOUVELLEMENT" in gdf.columns else ""

    # Calcul du score d'urgence
    gdf["SCORE_URGENCE"] = 0

    if "DEFAUT" in gdf.columns:
        gdf["DEFAUT"] = pd.to_numeric(gdf["DEFAUT"], errors="coerce")
        gdf.loc[gdf["DEFAUT"] == 1, "SCORE_URGENCE"] += 3

    gdf["SCORE_URGENCE"] += gdf["NB_LOCALISATIONS_DEFAUT"].clip(0, 3)

    if "VIGUEUR" in gdf.columns:
        gdf.loc[gdf["VIGUEUR_N"].str.contains("faible", na=False), "SCORE_URGENCE"] += 2

    if "PRIORITEDERENOUVELLEMENT" in gdf.columns:
        gdf.loc[gdf["PRIO_N"].str.contains("moins de 5|< ?5", na=False, regex=True), "SCORE_URGENCE"] += 2

    if "NOTEDIAGNOSTIC" in gdf.columns:
        gdf.loc[gdf["NOTE_N"].str.contains("mauvais|danger|urgent", na=False), "SCORE_URGENCE"] += 2

    gdf.loc[gdf["AGE_ESTIME"] >= 60, "SCORE_URGENCE"] += 1
    gdf.loc[gdf["AGE_ESTIME"] >= 80, "SCORE_URGENCE"] += 1

    # Ajouter coordonnees WGS84 au gdf principal
    gdf["lat"] = gdf_wgs.geometry.y
    gdf["lon"] = gdf_wgs.geometry.x

    return gdf


def create_sector_summary(gdf):
    """Cree le resume par secteur"""
    grp = (gdf
           .groupby("ADR_SECTEUR", dropna=False)
           .agg(
                nb_arbres=("geometry", "size"),
                pct_defaut=("DEFAUT", lambda s: np.nanmean(s == 1) * 100),
                pct_defaut_loc=("A_UN_DEFAUT_LOCALISE", "mean"),
                score_moy=("SCORE_URGENCE", "mean"),
                age_moy=("AGE_ESTIME", "mean"),
                top_travaux=("TRAVAUXPRECONISESDIAG", lambda s: s.value_counts().head(1).index[0] if s.notna().any() else np.nan)
           )
           .sort_values(["score_moy", "pct_defaut"], ascending=False)
          )
    return grp.reset_index()


def create_work_summary(gdf):
    """Cree le resume par type de travaux"""
    tab_trav = (gdf.groupby("TRAVAUXPRECONISESDIAG")
                .agg(
                    nb=("geometry", "size"),
                    score_moy=("SCORE_URGENCE", "mean"),
                    pct_defaut=("DEFAUT", lambda s: np.nanmean(s == 1) * 100)
                )
                .sort_values("nb", ascending=False))
    return tab_trav.reset_index()


def render_kpi_cards(gdf):
    """Affiche les cartes KPI"""
    col1, col2, col3, col4, col5 = st.columns(5)

    total_arbres = len(gdf)
    arbres_defaut = int((gdf["DEFAUT"] == 1).sum())
    pct_defaut = (arbres_defaut / total_arbres * 100) if total_arbres > 0 else 0
    score_moyen = gdf["SCORE_URGENCE"].mean()
    arbres_urgents = int((gdf["SCORE_URGENCE"] >= 4).sum())

    with col1:
        st.metric(
            label="Total Arbres",
            value=f"{total_arbres:,}".replace(",", " ")
        )

    with col2:
        st.metric(
            label="Arbres avec Defaut",
            value=f"{arbres_defaut:,}".replace(",", " "),
            delta=f"{pct_defaut:.1f}%"
        )

    with col3:
        st.metric(
            label="Score Moyen",
            value=f"{score_moyen:.2f}"
        )

    with col4:
        st.metric(
            label="Cas Urgents (score >= 4)",
            value=f"{arbres_urgents:,}".replace(",", " ")
        )

    with col5:
        nb_secteurs = gdf["ADR_SECTEUR"].nunique()
        st.metric(
            label="Secteurs",
            value=str(nb_secteurs)
        )


def render_map(gdf, mode="urgents", n_points=2000, score_min=4):
    """Affiche la carte interactive"""
    m = folium.Map(location=[45.18, 5.72], zoom_start=12, tiles="cartodbpositron")

    # Palette de couleurs selon le score
    def get_color(score):
        if score >= 5:
            return "#dc3545"  # Rouge
        elif score >= 4:
            return "#fd7e14"  # Orange
        elif score >= 3:
            return "#ffc107"  # Jaune
        elif score >= 2:
            return "#28a745"  # Vert
        else:
            return "#17a2b8"  # Bleu

    if mode == "urgents":
        data = gdf[gdf["SCORE_URGENCE"] >= score_min].copy()
    else:
        data = gdf.sort_values("SCORE_URGENCE", ascending=False).head(n_points).copy()

    mc = MarkerCluster().add_to(m)

    for _, r in data.iterrows():
        color = get_color(r.get("SCORE_URGENCE", 0))
        popup_text = f"""
        <b>Secteur:</b> {r.get('ADR_SECTEUR', 'N/A')}<br>
        <b>Score:</b> {r.get('SCORE_URGENCE', 'N/A')}<br>
        <b>Travaux:</b> {r.get('TRAVAUXPRECONISESDIAG', 'N/A')}<br>
        <b>Genre:</b> {r.get('GENRE_BOTA', 'N/A')}<br>
        <b>Age estime:</b> {r.get('AGE_ESTIME', 'N/A')} ans
        """
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(mc)

    # Legende
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; font-size: 12px;">
        <b>Score d'urgence</b><br>
        <i style="background: #dc3545; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> >= 5 (Critique)<br>
        <i style="background: #fd7e14; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 4 (Eleve)<br>
        <i style="background: #ffc107; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 3 (Modere)<br>
        <i style="background: #28a745; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 2 (Faible)<br>
        <i style="background: #17a2b8; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> < 2 (Normal)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def render_sector_analysis(gdf, sector_summary):
    """Affiche l'analyse par secteur"""

    # Graphique score moyen par secteur
    fig_score = px.bar(
        sector_summary.sort_values("score_moy", ascending=True),
        x="score_moy",
        y="ADR_SECTEUR",
        orientation="h",
        title="Score moyen d'urgence par secteur",
        labels={"score_moy": "Score moyen", "ADR_SECTEUR": "Secteur"},
        color="score_moy",
        color_continuous_scale="RdYlGn_r"
    )
    fig_score.update_layout(
        height=400,
        showlegend=False,
        yaxis=dict(type='category')
    )

    # Graphique taux de defaut par secteur
    fig_defaut = px.bar(
        sector_summary.sort_values("pct_defaut", ascending=True),
        x="pct_defaut",
        y="ADR_SECTEUR",
        orientation="h",
        title="Taux de defaut par secteur (%)",
        labels={"pct_defaut": "% avec defaut", "ADR_SECTEUR": "Secteur"},
        color="pct_defaut",
        color_continuous_scale="RdYlGn_r"
    )
    fig_defaut.update_layout(
        height=400,
        showlegend=False,
        yaxis=dict(type='category')
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_score, use_container_width=True)
    with col2:
        st.plotly_chart(fig_defaut, use_container_width=True)

    # Tableau recapitulatif
    st.markdown("### Tableau recapitulatif par secteur")

    display_df = sector_summary.copy()
    display_df.columns = ["Secteur", "Nb Arbres", "% Defaut", "% Defaut Loc.", "Score Moyen", "Age Moyen", "Travaux Principal"]
    display_df["% Defaut"] = display_df["% Defaut"].round(1)
    display_df["% Defaut Loc."] = (display_df["% Defaut Loc."] * 100).round(1)
    display_df["Score Moyen"] = display_df["Score Moyen"].round(2)
    display_df["Age Moyen"] = display_df["Age Moyen"].round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_work_analysis(gdf, work_summary):
    """Affiche l'analyse des travaux"""

    # Top travaux
    fig_travaux = px.bar(
        work_summary.head(10),
        x="nb",
        y="TRAVAUXPRECONISESDIAG",
        orientation="h",
        title="Types de travaux les plus frequents",
        labels={"nb": "Nombre d'arbres", "TRAVAUXPRECONISESDIAG": "Type de travaux"},
        color="score_moy",
        color_continuous_scale="RdYlGn_r"
    )
    fig_travaux.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})

    st.plotly_chart(fig_travaux, use_container_width=True)

    # Comparaison global vs urgents
    st.markdown("### Comparaison Global vs Top 10% Urgents")

    q = gdf["SCORE_URGENCE"].quantile(0.90)
    high = gdf[gdf["SCORE_URGENCE"] >= q]

    global_dist = (gdf["TRAVAUXPRECONISESDIAG"].value_counts(normalize=True) * 100).head(8)
    high_dist = (high["TRAVAUXPRECONISESDIAG"].value_counts(normalize=True) * 100).head(8)

    comp = pd.DataFrame({
        "Global (%)": global_dist,
        "Top 10% Urgents (%)": high_dist
    }).fillna(0).round(1)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Global",
        y=comp.index,
        x=comp["Global (%)"],
        orientation="h",
        marker_color="#667eea"
    ))
    fig_comp.add_trace(go.Bar(
        name="Top 10% Urgents",
        y=comp.index,
        x=comp["Top 10% Urgents (%)"],
        orientation="h",
        marker_color="#dc3545"
    ))
    fig_comp.update_layout(
        title="Distribution des travaux: Global vs Cas Urgents",
        barmode="group",
        height=450,
        xaxis_title="Pourcentage (%)",
        yaxis_title=""
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # Tableau des travaux
    st.markdown("### Detail par type de travaux")

    display_work = work_summary.copy()
    display_work.columns = ["Type de Travaux", "Nombre", "Score Moyen", "% Defaut"]
    display_work["Score Moyen"] = display_work["Score Moyen"].round(2)
    display_work["% Defaut"] = display_work["% Defaut"].round(1)

    st.dataframe(display_work, use_container_width=True, hide_index=True)


def render_priority_trees(gdf, selected_sector=None):
    """Affiche les arbres prioritaires"""

    if selected_sector and selected_sector != "Tous":
        data = gdf[gdf["ADR_SECTEUR"] == int(selected_sector)]
    else:
        data = gdf

    top_trees = (data
                 .sort_values("SCORE_URGENCE", ascending=False)
                 .head(50)[["ADR_SECTEUR", "GENRE_BOTA", "ESPECE", "AGE_ESTIME", "DEFAUT",
                           "NB_LOCALISATIONS_DEFAUT", "TRAVAUXPRECONISESDIAG",
                           "ANNEETRAVAUXPRECONISESDIAG", "SCORE_URGENCE"]])

    display_trees = top_trees.copy()
    display_trees.columns = ["Secteur", "Genre", "Espece", "Age", "Defaut", "Nb Loc. Defaut",
                             "Travaux Preconises", "Annee Travaux", "Score"]

    st.dataframe(
        display_trees,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                help="Score d'urgence",
                min_value=0,
                max_value=6,
                format="%d"
            )
        }
    )


def render_sector_work_pivot(gdf):
    """Affiche le pivot secteur x travaux"""
    pivot = (gdf.pivot_table(
        index="ADR_SECTEUR",
        columns="TRAVAUXPRECONISESDIAG",
        values="geometry",
        aggfunc="size",
        fill_value=0
    ))

    top8 = gdf["TRAVAUXPRECONISESDIAG"].value_counts().head(8).index
    pivot = pivot[top8].copy()

    fig = px.imshow(
        pivot,
        labels=dict(x="Type de Travaux", y="Secteur", color="Nombre"),
        title="Repartition des travaux par secteur",
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)


def render_statistics(gdf):
    """Affiche les statistiques generales"""

    col1, col2 = st.columns(2)

    with col1:
        # Distribution des scores
        fig_dist = px.histogram(
            gdf,
            x="SCORE_URGENCE",
            nbins=7,
            title="Distribution des scores d'urgence",
            labels={"SCORE_URGENCE": "Score d'urgence", "count": "Nombre d'arbres"},
            color_discrete_sequence=["#667eea"]
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Stats descriptives
        st.markdown("### Statistiques du score d'urgence")
        stats = gdf["SCORE_URGENCE"].describe()
        stats_df = pd.DataFrame({
            "Statistique": ["Moyenne", "Ecart-type", "Min", "25%", "Mediane", "75%", "Max"],
            "Valeur": [
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.0f}",
                f"{stats['25%']:.0f}",
                f"{stats['50%']:.0f}",
                f"{stats['75%']:.0f}",
                f"{stats['max']:.0f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        # Distribution par age
        age_bins = [0, 10, 20, 30, 40, 50, 60, 100]
        age_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"]
        gdf_age = gdf.copy()
        gdf_age["AGE_GROUPE"] = pd.cut(gdf_age["AGE_ESTIME"], bins=age_bins, labels=age_labels, right=False)

        age_stats = gdf_age.groupby("AGE_GROUPE", observed=True).agg({
            "SCORE_URGENCE": "mean",
            "geometry": "size"
        }).reset_index()
        age_stats.columns = ["Groupe Age", "Score Moyen", "Nombre"]

        fig_age = px.bar(
            age_stats,
            x="Groupe Age",
            y="Nombre",
            title="Repartition par groupe d'age",
            labels={"Nombre": "Nombre d'arbres", "Groupe Age": "Age (annees)"},
            color="Score Moyen",
            color_continuous_scale="RdYlGn_r"
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

        # Top genres
        st.markdown("### Top 10 genres botaniques")
        top_genres = gdf["GENRE_BOTA"].value_counts().head(10).reset_index()
        top_genres.columns = ["Genre", "Nombre"]
        st.dataframe(top_genres, use_container_width=True, hide_index=True)


# Application principale
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Choisir une section",
            ["Tableau de bord", "Carte interactive", "Analyse par secteur",
             "Analyse des travaux", "Arbres prioritaires", "Statistiques"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Filtres")

        # Chargement des donnees
        with st.spinner("Chargement des donnees..."):
            gdf = load_and_process_data()

        # Filtre par secteur
        secteurs = ["Tous"] + sorted([str(s) for s in gdf["ADR_SECTEUR"].dropna().unique()])
        selected_sector = st.selectbox("Secteur", secteurs)

        # Filtre par score minimum
        score_min = st.slider("Score minimum", 0, 6, 0)

        st.markdown("---")
        st.markdown("### A propos")
        st.markdown("""
        Application de gestion du patrimoine arboricole.

        **Score d'urgence:**
        - 0-2: Normal
        - 3: Modere
        - 4: Eleve
        - 5-6: Critique
        """)

    # Filtrage des donnees
    filtered_gdf = gdf.copy()
    if selected_sector != "Tous":
        filtered_gdf = filtered_gdf[filtered_gdf["ADR_SECTEUR"] == int(selected_sector)]
    if score_min > 0:
        filtered_gdf = filtered_gdf[filtered_gdf["SCORE_URGENCE"] >= score_min]

    # Header
    st.markdown('<p class="main-header">Gestion du Patrimoine Arboricole</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dashboard d\'analyse et de priorisation des interventions</p>', unsafe_allow_html=True)

    # Affichage selon la page selectionnee
    if page == "Tableau de bord":
        render_kpi_cards(filtered_gdf)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            sector_summary = create_sector_summary(filtered_gdf)
            fig_score = px.bar(
                sector_summary.sort_values("score_moy", ascending=True),
                x="score_moy",
                y="ADR_SECTEUR",
                orientation="h",
                title="Score moyen par secteur",
                color="score_moy",
                color_continuous_scale="RdYlGn_r"
            )
            fig_score.update_layout(height=350, showlegend=False, yaxis=dict(type='category'))
            st.plotly_chart(fig_score, use_container_width=True)

        with col2:
            work_summary = create_work_summary(filtered_gdf)
            fig_work = px.pie(
                work_summary.head(6),
                values="nb",
                names="TRAVAUXPRECONISESDIAG",
                title="Repartition des travaux preconises",
                hole=0.4
            )
            fig_work.update_layout(height=350)
            st.plotly_chart(fig_work, use_container_width=True)

        # Apercu carte
        st.markdown("### Apercu cartographique (cas urgents)")
        m = render_map(filtered_gdf, mode="urgents", score_min=4)
        st_folium(m, width=None, height=400, returned_objects=[])

    elif page == "Carte interactive":
        st.markdown("### Carte interactive des arbres")

        col1, col2 = st.columns([1, 3])
        with col1:
            map_mode = st.radio(
                "Mode d'affichage",
                ["Cas urgents (score >= seuil)", "Top N arbres"]
            )

            if "urgents" in map_mode:
                map_score_min = st.slider("Score minimum", 1, 6, 4, key="map_score")
                m = render_map(filtered_gdf, mode="urgents", score_min=map_score_min)
                st.info(f"{len(filtered_gdf[filtered_gdf['SCORE_URGENCE'] >= map_score_min])} arbres affiches")
            else:
                n_points = st.slider("Nombre d'arbres", 100, 5000, 2000, step=100)
                m = render_map(filtered_gdf, mode="top_n", n_points=n_points)
                st.info(f"{min(n_points, len(filtered_gdf))} arbres affiches")

        with col2:
            st_folium(m, width=None, height=600, returned_objects=[])

    elif page == "Analyse par secteur":
        sector_summary = create_sector_summary(filtered_gdf)
        render_sector_analysis(filtered_gdf, sector_summary)

        st.markdown("---")
        st.markdown("### Matrice Secteur x Travaux")
        render_sector_work_pivot(filtered_gdf)

    elif page == "Analyse des travaux":
        work_summary = create_work_summary(filtered_gdf)
        render_work_analysis(filtered_gdf, work_summary)

    elif page == "Arbres prioritaires":
        st.markdown("### Top 50 arbres prioritaires")

        sector_filter = st.selectbox(
            "Filtrer par secteur",
            ["Tous"] + sorted([str(s) for s in gdf["ADR_SECTEUR"].dropna().unique()]),
            key="priority_sector"
        )

        render_priority_trees(filtered_gdf, sector_filter)

    elif page == "Statistiques":
        render_statistics(filtered_gdf)


if __name__ == "__main__":
    main()
