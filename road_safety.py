"""
UK Road Accident Severity Predictor
=====================================
A multi-page Dash app demonstrating predictive modelling on STATS19-style data.
Styled using the GOV.UK Design System colour palette and typographic conventions.

Run:
    pip install -r requirements.txt
    python app.py
Then visit: http://127.0.0.1:8050
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import shap

import dash
from dash import dcc, html, Input, Output, State

# ─────────────────────────────────────────────────────────────────────────────
# GOV.UK DESIGN SYSTEM COLOURS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "black":        "#0b0c0c",
    "white":        "#ffffff",
    "blue":         "#1d70b8",
    "blue_dark":    "#003078",
    "yellow":       "#ffdd00",
    "green":        "#00703c",
    "green_dark":   "#005a30",
    "red":          "#d4351c",
    "orange":       "#f47738",
    "light_grey":   "#f3f2f1",
    "mid_grey":     "#b1b4b6",
    "dark_grey":    "#505a5f",
    "purple":       "#4c2c92",
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC STATS19-STYLE DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 6000

WEATHER_CONDS   = ["Fine", "Raining", "Snowing", "Fog or mist", "High winds"]
SURFACE_CONDS   = ["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood"]
ROAD_TYPES      = ["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"]
LIGHT_CONDS     = ["Daylight", "Darkness – lights lit", "Darkness – no lighting", "Darkness – lights unlit"]
URBAN_RURAL     = ["Urban", "Rural"]
DAYS_OF_WEEK    = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SPEED_LIMITS    = [20, 30, 40, 50, 60, 70]
YEARS           = list(range(2018, 2024))

raw = {
    "weather":            np.random.choice(WEATHER_CONDS, N, p=[0.60, 0.25, 0.05, 0.05, 0.05]),
    "road_surface":       np.random.choice(SURFACE_CONDS, N, p=[0.55, 0.30, 0.07, 0.05, 0.03]),
    "road_type":          np.random.choice(ROAD_TYPES,    N, p=[0.50, 0.25, 0.12, 0.08, 0.05]),
    "light_conditions":   np.random.choice(LIGHT_CONDS,   N, p=[0.60, 0.25, 0.10, 0.05]),
    "urban_rural":        np.random.choice(URBAN_RURAL,   N, p=[0.65, 0.35]),
    "day_of_week":        np.random.choice(DAYS_OF_WEEK,  N),
    "speed_limit":        np.random.choice(SPEED_LIMITS,  N, p=[0.10, 0.40, 0.10, 0.10, 0.20, 0.10]),
    "hour":               np.random.randint(0, 24, N),
    "number_of_vehicles": np.random.randint(1, 6, N),
    "number_of_casualties": np.random.randint(1, 4, N),
    "year":               np.random.choice(YEARS, N),
    "latitude":           np.random.uniform(50.5, 55.8, N),
    "longitude":          np.random.uniform(-3.8, 1.8, N),
}

df = pd.DataFrame(raw)


def assign_severity(row):
    score = 0
    if row["speed_limit"] >= 60:             score += 4
    elif row["speed_limit"] >= 40:           score += 2
    if row["road_surface"] in ["Frost or ice", "Snow"]: score += 3
    elif row["road_surface"] == "Wet or damp":           score += 1
    if row["weather"] in ["Snowing", "Fog or mist"]:     score += 2
    elif row["weather"] == "Raining":                     score += 1
    if row["light_conditions"] == "Darkness – no lighting": score += 3
    elif "Darkness" in row["light_conditions"]:              score += 1
    if row["urban_rural"] == "Rural":        score += 2
    if 0 <= row["hour"] < 5:                 score += 2
    elif row["hour"] in [7, 8, 17, 18]:      score += 1
    score += int(np.random.normal(0, 1.5))
    if score <= 2:   return "Slight"
    elif score <= 6: return "Serious"
    else:            return "Fatal"


df["severity"] = df.apply(assign_severity, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "weather", "road_surface", "road_type", "light_conditions",
    "urban_rural", "day_of_week", "speed_limit", "hour",
    "number_of_vehicles", "number_of_casualties",
]

df_model = df[FEATURES + ["severity"]].copy()

le_map = {}
for col in ["weather", "road_surface", "road_type", "light_conditions", "urban_rural", "day_of_week"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_map[col] = le

le_target = LabelEncoder()
df_model["severity_enc"] = le_target.fit_transform(df_model["severity"])

X = df_model[FEATURES].values
y = df_model["severity_enc"].values
CLASSES = le_target.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# SHAP (computed on a 200-sample subset for speed)
explainer   = shap.TreeExplainer(clf)
shap_sample = pd.DataFrame(X_test[:200], columns=FEATURES)
shap_values = explainer.shap_values(shap_sample)   # list[class] of (200, n_features)

# Pre-compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)
report    = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
feat_imp  = pd.DataFrame({"feature": FEATURES, "importance": clf.feature_importances_}).sort_values("importance")

y_test_bin = label_binarize(y_test, classes=list(range(len(CLASSES))))

# ─────────────────────────────────────────────────────────────────────────────
# GOV.UK CSS (injected via <style>)
# ─────────────────────────────────────────────────────────────────────────────
GOVUK_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: 'Source Sans 3', Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: {C['black']};
    background: {C['light_grey']};
}}

/* ── Header ── */
.gov-header {{
    background: {C['black']};
    border-bottom: 8px solid {C['blue']};
}}
.gov-header__inner {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}}
.gov-header__logo {{
    color: {C['white']};
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    white-space: nowrap;
}}
.gov-header__logo span {{
    color: {C['blue']};
}}
.gov-header__divider {{
    width: 1px;
    height: 36px;
    background: {C['dark_grey']};
}}
.gov-header__service {{
    color: {C['white']};
    font-size: 17px;
    font-weight: 600;
}}

/* ── Phase Banner ── */
.gov-phase {{
    background: {C['blue_dark']};
    padding: 6px 0;
}}
.gov-phase__inner {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: {C['white']};
}}
.gov-phase__tag {{
    background: {C['yellow']};
    color: {C['black']};
    font-weight: 700;
    font-size: 13px;
    padding: 2px 10px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

/* ── Main Wrapper ── */
.gov-main {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px 60px;
}}

/* ── Page Title ── */
.gov-page-title {{ font-size: 40px; font-weight: 700; line-height: 1.15; margin-bottom: 8px; color: {C['black']}; }}
.gov-page-lead   {{ font-size: 20px; color: {C['dark_grey']}; margin-bottom: 28px; max-width: 800px; }}

/* ── Tabs ── */
.gov-tabs {{ margin-bottom: 0; }}
.custom-tab {{
    background: {C['light_grey']};
    border: 1px solid {C['mid_grey']};
    border-bottom: none;
    color: {C['blue']};
    font-size: 15px;
    font-weight: 600;
    padding: 10px 22px;
    cursor: pointer;
    font-family: inherit;
    text-decoration: underline;
    transition: background 0.15s;
}}
.custom-tab:hover  {{ background: {C['white']}; }}
.custom-tab--selected {{
    background: {C['white']};
    border-color: {C['mid_grey']};
    color: {C['black']};
    text-decoration: none;
}}

/* ── Stat Cards ── */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}
.stat-card {{
    background: {C['white']};
    border: 1px solid {C['mid_grey']};
    border-top: 5px solid {C['blue']};
    padding: 20px 24px;
}}
.stat-card--green  {{ border-top-color: {C['green']}; }}
.stat-card--orange {{ border-top-color: {C['orange']}; }}
.stat-card--red    {{ border-top-color: {C['red']}; }}
.stat-num  {{ display: block; font-size: 40px; font-weight: 700; line-height: 1; margin-bottom: 4px; color: {C['blue']}; }}
.stat-num--green  {{ color: {C['green']}; }}
.stat-num--orange {{ color: {C['orange']}; }}
.stat-num--red    {{ color: {C['red']}; }}
.stat-label {{ font-size: 13px; color: {C['dark_grey']}; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }}

/* ── Chart Cards ── */
.chart-card {{
    background: {C['white']};
    border: 1px solid {C['mid_grey']};
    padding: 4px;
}}
.chart-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
.chart-grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
.chart-full   {{ margin-bottom: 16px; }}

/* ── Predict Tab ── */
.predict-layout {{
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 24px;
    align-items: start;
}}
.predict-sidebar {{
    background: {C['white']};
    border: 1px solid {C['mid_grey']};
    border-top: 5px solid {C['blue']};
    padding: 24px;
}}
.predict-sidebar h2 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
.predict-sidebar p  {{ font-size: 14px; color: {C['dark_grey']}; margin-bottom: 20px; }}

.form-group  {{ margin-bottom: 16px; }}
.gov-label   {{ display: block; font-weight: 700; font-size: 15px; margin-bottom: 5px; color: {C['black']}; }}
.gov-hint    {{ display: block; font-size: 13px; color: {C['dark_grey']}; margin-bottom: 5px; }}

.gov-button {{
    background: {C['green']};
    color: {C['white']};
    border: none;
    padding: 12px 22px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    font-family: inherit;
    box-shadow: 0 3px 0 {C['green_dark']};
    width: 100%;
    margin-top: 8px;
    transition: background 0.1s;
    letter-spacing: 0.2px;
}}
.gov-button:hover  {{ background: {C['green_dark']}; }}
.gov-button:active {{ transform: translateY(2px); box-shadow: none; }}

/* ── Result Panel ── */
.result-panel {{
    text-align: center;
    padding: 28px 24px;
    color: {C['white']};
    margin-bottom: 16px;
    font-weight: 700;
}}
.result-panel--slight  {{ background: {C['green']}; }}
.result-panel--serious {{ background: {C['orange']}; }}
.result-panel--fatal   {{ background: {C['red']}; }}
.result-verdict  {{ font-size: 52px; letter-spacing: -1px; }}
.result-label    {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; opacity: 0.85; }}
.result-conf     {{ font-size: 20px; margin-top: 6px; opacity: 0.9; }}

/* ── Warning ── */
.gov-warning {{
    background: #fff8c5;
    border-left: 5px solid {C['yellow']};
    padding: 14px 16px;
    font-size: 14px;
    color: {C['black']};
    margin-top: 16px;
}}

/* ── Section Headers ── */
.section-header {{ font-size: 26px; font-weight: 700; margin-bottom: 6px; color: {C['black']}; }}
.section-sub    {{ font-size: 15px; color: {C['dark_grey']}; margin-bottom: 20px; max-width: 700px; }}

/* ── Inset Box ── */
.inset-box {{
    background: {C['white']};
    border-left: 5px solid {C['mid_grey']};
    padding: 16px 20px;
    margin-bottom: 20px;
    font-size: 15px;
}}

/* ── Footer ── */
.gov-footer {{
    background: {C['light_grey']};
    border-top: 2px solid {C['mid_grey']};
    padding: 28px 24px;
    text-align: center;
    font-size: 13px;
    color: {C['dark_grey']};
    margin-top: 40px;
}}
.gov-footer a {{ color: {C['blue']}; }}

/* ── Responsive ── */
@media (max-width: 900px) {{
    .stat-grid    {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-grid-2 {{ grid-template-columns: 1fr; }}
    .chart-grid-3 {{ grid-template-columns: 1fr; }}
    .predict-layout {{ grid-template-columns: 1fr; }}
}}
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["white"],
    plot_bgcolor=C["white"],
    font=dict(family="Source Sans 3, Arial, sans-serif", size=13, color=C["black"]),
    margin=dict(t=50, b=30, l=20, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["mid_grey"], borderwidth=1),
)

def chart(fig, height=320):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=C["light_grey"], linecolor=C["mid_grey"])
    fig.update_yaxes(showgrid=True, gridcolor=C["light_grey"], linecolor=C["mid_grey"])
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def render_explorer():
    n_total   = len(df)
    n_slight  = (df["severity"] == "Slight").sum()
    n_serious = (df["severity"] == "Serious").sum()
    n_fatal   = (df["severity"] == "Fatal").sum()

    # ── Pie ──────────────────────────────────────────────────────────────────
    pie = go.Figure(go.Pie(
        labels=["Slight", "Serious", "Fatal"],
        values=[n_slight, n_serious, n_fatal],
        marker_colors=[C["green"], C["orange"], C["red"]],
        hole=0.42,
        textinfo="label+percent",
    ))
    pie.update_layout(title_text="Severity Distribution", showlegend=False, **PLOTLY_LAYOUT, height=300)

    # ── Accidents by hour ────────────────────────────────────────────────────
    hourly = df.groupby("hour").size().reset_index(name="count")
    hour_fig = go.Figure(go.Bar(x=hourly["hour"], y=hourly["count"], marker_color=C["blue"],
                                 marker_line_color=C["blue_dark"], marker_line_width=0.5))
    hour_fig.update_layout(title_text="Accidents by Hour of Day",
                            xaxis_title="Hour", yaxis_title="Count", **PLOTLY_LAYOUT, height=300)

    # ── Speed limit vs severity ──────────────────────────────────────────────
    speed_sev = df.groupby(["speed_limit", "severity"]).size().reset_index(name="count")
    speed_fig = px.bar(speed_sev, x="speed_limit", y="count", color="severity",
                       color_discrete_map={"Slight": C["green"], "Serious": C["orange"], "Fatal": C["red"]},
                       barmode="group")
    speed_fig.update_layout(title_text="Severity by Speed Limit (mph)",
                             xaxis_title="Speed Limit", yaxis_title="Count", **PLOTLY_LAYOUT, height=300)

    # ── Weather vs severity ──────────────────────────────────────────────────
    weather_sev = df.groupby(["weather", "severity"]).size().reset_index(name="count")
    weather_fig = px.bar(weather_sev, x="weather", y="count", color="severity",
                         color_discrete_map={"Slight": C["green"], "Serious": C["orange"], "Fatal": C["red"]},
                         barmode="stack")
    weather_fig.update_layout(title_text="Severity by Weather Condition",
                               xaxis_title="", yaxis_title="Count", **PLOTLY_LAYOUT, height=300)

    # ── Day of week heatmap ──────────────────────────────────────────────────
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby(["day_of_week", "severity"]).size().unstack(fill_value=0).reindex(order)
    heat = go.Figure(go.Heatmap(
        z=dow.values,
        x=list(dow.columns),
        y=list(dow.index),
        colorscale=[[0, C["light_grey"]], [0.5, C["blue"]], [1, C["blue_dark"]]],
        text=dow.values, texttemplate="%{text}",
    ))
    heat.update_layout(title_text="Accidents by Day × Severity", **PLOTLY_LAYOUT, height=280,
                       margin=dict(t=50, b=20, l=100, r=20))

    # ── UK Map (sample 800) ──────────────────────────────────────────────────
    samp = df.sample(800, random_state=1)
    colour_map = {"Slight": C["green"], "Serious": C["orange"], "Fatal": C["red"]}
    map_fig = go.Figure()
    for sev, grp in samp.groupby("severity"):
        map_fig.add_trace(go.Scattergeo(
            lat=grp["latitude"], lon=grp["longitude"],
            mode="markers",
            name=sev,
            marker=dict(size=5, color=colour_map[sev], opacity=0.65),
            hovertemplate=f"<b>{sev}</b><br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<extra></extra>"
        ))
    map_fig.update_layout(
        title_text="Accident Hotspot Map (sample of 800)",
        geo=dict(scope="europe", center=dict(lat=53.5, lon=-1.5), projection_scale=7,
                 bgcolor=C["light_grey"], landcolor="#e4ead9", oceancolor="#cce0f0",
                 showocean=True, showlakes=True, lakecolor="#cce0f0",
                 showcountries=True, countrycolor=C["mid_grey"]),
        **PLOTLY_LAYOUT, height=440,
    )

    return html.Div([
        # Stat cards
        html.Div([
            html.Div([html.Span(f"{n_total:,}", className="stat-num"),
                      html.Span("Total Accidents", className="stat-label")], className="stat-card"),
            html.Div([html.Span(f"{n_slight:,}", className="stat-num stat-num--green"),
                      html.Span("Slight", className="stat-label")], className="stat-card stat-card--green"),
            html.Div([html.Span(f"{n_serious:,}", className="stat-num stat-num--orange"),
                      html.Span("Serious", className="stat-label")], className="stat-card stat-card--orange"),
            html.Div([html.Span(f"{n_fatal:,}", className="stat-num stat-num--red"),
                      html.Span("Fatal", className="stat-label")], className="stat-card stat-card--red"),
        ], className="stat-grid"),

        # Row 1
        html.Div([
            html.Div([chart(pie, 320)], className="chart-card"),
            html.Div([chart(hour_fig, 320)], className="chart-card"),
        ], className="chart-grid-2"),

        # Row 2
        html.Div([
            html.Div([chart(speed_fig, 310)], className="chart-card"),
            html.Div([chart(weather_fig, 310)], className="chart-card"),
        ], className="chart-grid-2"),

        # Heatmap
        html.Div([chart(heat, 280)], className="chart-card chart-full"),

        # Map
        html.Div([chart(map_fig, 440)], className="chart-card chart-full"),
    ], style={"padding": "24px", "background": C["white"], "border": f"1px solid {C['mid_grey']}"}),


def render_predict():
    def dropdown(id_, options, value):
        return dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in options],
            value=value,
            clearable=False,
            style={"fontSize": "15px", "fontFamily": "inherit"},
        )

    def speed_dd(id_, options, value):
        return dcc.Dropdown(
            id=id_,
            options=[{"label": str(o) + " mph", "value": o} for o in options],
            value=value,
            clearable=False,
            style={"fontSize": "15px", "fontFamily": "inherit"},
        )

    sidebar = html.Div([
        html.H2("Road Conditions"),
        html.P("Adjust the conditions below and click Predict to see the estimated accident severity."),

        html.Div([html.Label("Weather", className="gov-label"), dropdown("inp-weather", WEATHER_CONDS, "Fine")], className="form-group"),
        html.Div([html.Label("Road Surface", className="gov-label"), dropdown("inp-surface", SURFACE_CONDS, "Dry")], className="form-group"),
        html.Div([html.Label("Road Type", className="gov-label"), dropdown("inp-road-type", ROAD_TYPES, "Single carriageway")], className="form-group"),
        html.Div([html.Label("Lighting", className="gov-label"), dropdown("inp-light", LIGHT_CONDS, "Daylight")], className="form-group"),
        html.Div([html.Label("Area", className="gov-label"), dropdown("inp-urban", URBAN_RURAL, "Urban")], className="form-group"),
        html.Div([html.Label("Day of Week", className="gov-label"), dropdown("inp-day", DAYS_OF_WEEK, "Monday")], className="form-group"),
        html.Div([html.Label("Speed Limit", className="gov-label"), speed_dd("inp-speed", SPEED_LIMITS, 30)], className="form-group"),

        html.Div([
            html.Label("Hour of Day", className="gov-label"),
            html.Span("0 = midnight, 12 = noon", className="gov-hint"),
            dcc.Slider(id="inp-hour", min=0, max=23, step=1, value=9,
                       marks={i: str(i) for i in range(0, 24, 3)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], className="form-group", style={"paddingBottom": "24px"}),

        html.Div([
            html.Label("Number of Vehicles", className="gov-label"),
            dcc.Slider(id="inp-vehicles", min=1, max=5, step=1, value=2,
                       marks={i: str(i) for i in range(1, 6)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], className="form-group", style={"paddingBottom": "18px"}),

        html.Div([
            html.Label("Number of Casualties", className="gov-label"),
            dcc.Slider(id="inp-casualties", min=1, max=4, step=1, value=1,
                       marks={i: str(i) for i in range(1, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], className="form-group", style={"paddingBottom": "18px"}),

        html.Button("🔮  Predict Severity", id="btn-predict", n_clicks=0, className="gov-button"),

    ], className="predict-sidebar")

    results = html.Div([
        html.Div(
            id="prediction-output",
            children=[html.Div([
                html.P("👈  Fill in the conditions and click Predict.",
                       style={"fontSize": "18px", "color": C["dark_grey"], "padding": "40px", "textAlign": "center"})
            ], style={"background": C["white"], "border": f"1px solid {C['mid_grey']}"})]
        )
    ])

    return html.Div([
        html.Div([sidebar, results], className="predict-layout"),
    ], style={"padding": "24px", "background": C["white"], "border": f"1px solid {C['mid_grey']}"}),


def render_shap():
    # Mean |SHAP| across all classes
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    shap_df  = pd.DataFrame({"feature": FEATURES, "mean_shap": mean_abs}).sort_values("mean_shap")

    global_fig = go.Figure(go.Bar(
        x=shap_df["mean_shap"], y=shap_df["feature"], orientation="h",
        marker_color=C["blue"], marker_line_color=C["blue_dark"], marker_line_width=0.6,
    ))
    global_fig.update_layout(
        title_text="Global Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        **PLOTLY_LAYOUT, height=360,
        margin=dict(t=50, b=30, l=160, r=20),
    )

    # Per-class breakdowns
    class_colours = {"Slight": C["green"], "Serious": C["orange"], "Fatal": C["red"]}
    per_class = []
    for i, cls in enumerate(CLASSES):
        cdf = pd.DataFrame({"feature": FEATURES, "shap": np.abs(shap_values[i]).mean(axis=0)}).sort_values("shap")
        fig = go.Figure(go.Bar(x=cdf["shap"], y=cdf["feature"], orientation="h",
                               marker_color=class_colours[cls]))
        fig.update_layout(title_text=f"{cls} — Feature Impact",
                          xaxis_title="Mean |SHAP|",
                          **PLOTLY_LAYOUT, height=300,
                          margin=dict(t=50, b=30, l=160, r=20))
        per_class.append(html.Div([chart(fig, 300)], className="chart-card"))

    return html.Div([
        html.H2("SHAP Explainability", className="section-header"),
        html.P("SHAP (SHapley Additive exPlanations) values reveal exactly how much each feature "
               "contributes to each prediction — making the model transparent and auditable.",
               className="section-sub"),

        html.Div([
            html.Strong("How to read this: "),
            "A higher SHAP value means that feature has a larger impact on pushing the model's output "
            "away from the average prediction. Red bars (on instance charts) increase severity risk; "
            "green bars reduce it."
        ], className="inset-box"),

        html.Div([chart(global_fig, 360)], className="chart-card chart-full"),

        html.H3("Per-Class Breakdown", style={"fontSize": "20px", "fontWeight": "700",
                                              "marginBottom": "12px", "marginTop": "8px"}),
        html.Div(per_class, className="chart-grid-3"),

    ], style={"padding": "24px", "background": C["white"], "border": f"1px solid {C['mid_grey']}"}),


def render_performance():
    # Confusion matrix
    cm_fig = go.Figure(go.Heatmap(
        z=cm, x=list(CLASSES), y=list(CLASSES),
        colorscale=[[0, C["light_grey"]], [0.4, "#a8c8e8"], [1, C["blue_dark"]]],
        text=cm, texttemplate="<b>%{text}</b>", showscale=True,
    ))
    cm_fig.update_layout(
        title_text="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        **PLOTLY_LAYOUT, height=360,
    )

    # ROC curves
    roc_fig = go.Figure()
    roc_colors = [C["green"], C["red"], C["orange"]]
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_score   = auc(fpr, tpr)
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{cls} (AUC={auc_score:.2f})",
            line=dict(color=roc_colors[i], width=2.5),
            mode="lines",
        ))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                  line=dict(dash="dash", color=C["mid_grey"], width=1.5)))
    roc_fig.update_layout(
        title_text="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        **PLOTLY_LAYOUT, height=360,
    )

    # Feature importance
    fi_fig = go.Figure(go.Bar(
        x=feat_imp["importance"], y=feat_imp["feature"], orientation="h",
        marker_color=C["blue"], marker_line_color=C["blue_dark"], marker_line_width=0.5,
    ))
    fi_fig.update_layout(
        title_text="Random Forest — Feature Importance (MDI)",
        xaxis_title="Mean Decrease in Impurity",
        **PLOTLY_LAYOUT, height=360,
        margin=dict(t=50, b=30, l=160, r=20),
    )

    # Per-class scores
    score_cards = []
    for cls in CLASSES:
        r = report[cls]
        col = {"Slight": "green", "Serious": "orange", "Fatal": "red"}[cls]
        score_cards.append(html.Div([
            html.Div(cls, style={"fontSize": "14px", "fontWeight": "700", "textTransform": "uppercase",
                                  "letterSpacing": "0.8px", "marginBottom": "12px", "color": C["dark_grey"]}),
            html.Div([
                html.Div([html.Span(f"{r['precision']:.2f}", className=f"stat-num stat-num--{col}"),
                          html.Span("Precision", className="stat-label")]),
                html.Div([html.Span(f"{r['recall']:.2f}",    className=f"stat-num stat-num--{col}"),
                          html.Span("Recall",    className="stat-label")]),
                html.Div([html.Span(f"{r['f1-score']:.2f}",  className=f"stat-num stat-num--{col}"),
                          html.Span("F1 Score",  className="stat-label")]),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"}),
        ], className=f"stat-card stat-card--{col}", style={"marginBottom": "0"}))

    return html.Div([
        html.H2("Model Performance", className="section-header"),
        html.P("Evaluation of the Random Forest classifier on the held-out 20% test set (stratified split).",
               className="section-sub"),

        # Accuracy headline
        html.Div([
            html.Div([html.Span(f"{accuracy:.1%}", className="stat-num"),
                      html.Span("Overall Accuracy", className="stat-label")], className="stat-card"),
            html.Div([html.Span(f"{len(X_test):,}", className="stat-num"),
                      html.Span("Test Samples", className="stat-label")], className="stat-card"),
            html.Div([html.Span("150", className="stat-num"),
                      html.Span("Trees in Forest", className="stat-label")], className="stat-card"),
            html.Div([html.Span(f"{len(FEATURES)}", className="stat-num"),
                      html.Span("Input Features", className="stat-label")], className="stat-card"),
        ], className="stat-grid"),

        # Per-class score cards
        html.Div(score_cards, style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                                      "gap": "16px", "marginBottom": "20px"}),

        # Charts
        html.Div([
            html.Div([chart(cm_fig,  360)], className="chart-card"),
            html.Div([chart(roc_fig, 360)], className="chart-card"),
        ], className="chart-grid-2"),

        html.Div([chart(fi_fig, 360)], className="chart-card chart-full"),

    ], style={"padding": "24px", "background": C["white"], "border": f"1px solid {C['mid_grey']}"}),


# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Road Accident Severity – GOV.UK"

app.layout = html.Div([
    html.Style(GOVUK_CSS),

    # Header
    html.Header(html.Div([
        html.Span([html.Span("GOV", style={"color": C["yellow"]}), ".UK"], className="gov-header__logo"),
        html.Div(className="gov-header__divider"),
        html.Span("Road Accident Severity Predictor", className="gov-header__service"),
    ], className="gov-header__inner"), className="gov-header"),

    # Phase banner
    html.Div(html.Div([
        html.Span("BETA", className="gov-phase__tag"),
        "This is a predictive modelling demonstration using synthetic STATS19-style data from the Department for Transport.",
    ], className="gov-phase__inner"), className="gov-phase"),

    # Main
    html.Main(html.Div([
        html.H1("UK Road Safety — Accident Severity Analysis", className="gov-page-title"),
        html.P("Explore historical accident patterns and predict injury severity from road and environmental conditions using machine learning.",
               className="gov-page-lead"),

        dcc.Tabs(id="tabs", value="tab-explorer", className="gov-tabs", children=[
            dcc.Tab(label="📊  Data Explorer",     value="tab-explorer",    className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="🤖  Predict Severity",  value="tab-predict",     className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="🔍  SHAP Explainability", value="tab-shap",      className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="📈  Model Performance", value="tab-performance",  className="custom-tab", selected_className="custom-tab--selected"),
        ]),

        html.Div(id="tab-content"),

    ], className="gov-main"), className="gov-main"),

    # Footer
    html.Footer(html.Div([
        html.P("Built with synthetic STATS19 road safety data · Department for Transport · © Crown Copyright"),
        html.P([
            "Data source: ",
            html.A("data.gov.uk/dataset/road-accidents-safety-data",
                   href="https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data",
                   target="_blank"),
        ], style={"marginTop": "6px"}),
    ], style={"maxWidth": "1200px", "margin": "0 auto"}), className="gov-footer"),

], style={"minHeight": "100vh"})


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-explorer":    return render_explorer()
    if tab == "tab-predict":     return render_predict()
    if tab == "tab-shap":        return render_shap()
    if tab == "tab-performance": return render_performance()


@app.callback(
    Output("prediction-output", "children"),
    Input("btn-predict", "n_clicks"),
    State("inp-weather",    "value"),
    State("inp-surface",    "value"),
    State("inp-road-type",  "value"),
    State("inp-light",      "value"),
    State("inp-urban",      "value"),
    State("inp-day",        "value"),
    State("inp-speed",      "value"),
    State("inp-hour",       "value"),
    State("inp-vehicles",   "value"),
    State("inp-casualties", "value"),
    prevent_initial_call=True,
)
def predict(n_clicks, weather, surface, road_type, light, urban, day,
            speed, hour, vehicles, casualties):

    def encode(col, val):
        return int(le_map[col].transform([val])[0])

    row = np.array([[
        encode("weather",          weather),
        encode("road_surface",     surface),
        encode("road_type",        road_type),
        encode("light_conditions", light),
        encode("urban_rural",      urban),
        encode("day_of_week",      day),
        speed, hour, vehicles, casualties,
    ]])

    pred_enc  = clf.predict(row)[0]
    proba     = clf.predict_proba(row)[0]
    severity  = le_target.inverse_transform([pred_enc])[0]
    confidence = proba[pred_enc] * 100

    # SHAP for this instance
    inst_df   = pd.DataFrame(row, columns=FEATURES)
    inst_shap = explainer.shap_values(inst_df)          # list of 3 arrays (1, n_features)
    shap_vals = inst_shap[pred_enc][0]

    shap_inst_df = pd.DataFrame({"feature": FEATURES, "shap": shap_vals}).sort_values("shap")
    bar_colors   = [C["red"] if v > 0 else C["green"] for v in shap_inst_df["shap"]]

    shap_inst_fig = go.Figure(go.Bar(
        x=shap_inst_df["shap"], y=shap_inst_df["feature"],
        orientation="h", marker_color=bar_colors,
    ))
    shap_inst_fig.update_layout(
        title_text="Feature Contributions to this Prediction",
        xaxis_title="SHAP value  (red = increases severity risk, green = reduces it)",
        **PLOTLY_LAYOUT, height=340,
        margin=dict(t=50, b=30, l=160, r=20),
    )

    # Probability gauge bar
    prob_fig = go.Figure()
    for i, cls in enumerate(CLASSES):
        prob_fig.add_trace(go.Bar(
            x=[cls], y=[proba[i] * 100],
            name=cls,
            marker_color=[C["green"], C["red"], C["orange"]][i],
            text=[f"{proba[i]*100:.1f}%"],
            textposition="auto",
        ))
    prob_fig.update_layout(
        title_text="Class Probabilities",
        yaxis_title="Probability (%)", yaxis_range=[0, 100],
        showlegend=False, **PLOTLY_LAYOUT, height=280,
    )

    panel_cls = {"Slight": "result-panel--slight", "Serious": "result-panel--serious", "Fatal": "result-panel--fatal"}[severity]

    return html.Div([
        # Result panel
        html.Div([
            html.Div("Predicted Severity", className="result-label"),
            html.Div(f"⚠  {severity.upper()}", className="result-verdict"),
            html.Div(f"Model confidence: {confidence:.1f}%", className="result-conf"),
        ], className=f"result-panel {panel_cls}"),

        html.Div([chart(prob_fig, 280)], className="chart-card", style={"marginBottom": "12px"}),
        html.Div([chart(shap_inst_fig, 340)], className="chart-card"),

        html.Div([
            "⚠  This is a demonstration model trained on synthetic data. "
            "Do not use for real operational road safety decisions."
        ], className="gov-warning"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n App ready — visit http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)
