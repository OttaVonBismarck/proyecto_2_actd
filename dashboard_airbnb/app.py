import pandas as pd
import numpy as np
from pathlib import Path

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from tensorflow import keras
import tensorflow as tf

BASE_FONT = dict(
    family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color="#484848",
)

def apply_fig_style(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=BASE_FONT,
        hoverlabel=dict(bgcolor="white"),
    )
    return fig

DATA_PATH = Path(__file__).parent / "data" / "airbnb_barcelona.csv"
df = pd.read_csv(DATA_PATH)

BETAS_PATH = Path(__file__).parent / "data" / "betas_regresion.csv"
betas_df = pd.read_csv(BETAS_PATH)

CLASS_MODEL_PATH = Path(__file__).parent / "data" / "recommend_model.keras"
recommend_model = keras.models.load_model(CLASS_MODEL_PATH)

betas_df["abs_beta"] = betas_df["beta"].abs()
betas_df = betas_df.sort_values("abs_beta", ascending=False)

beta_fig = px.bar(
    betas_df,
    x="abs_beta",
    y="variable",
    orientation="h",
    text="abs_beta",
    labels={"abs_beta": "Coeficiente |β|", "variable": "Variable"},
)
beta_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
beta_fig.update_layout(margin=dict(l=150, r=40, t=40, b=40))
BETA_FIG = apply_fig_style(beta_fig)

df["price"] = df["price"].astype(str)
df["price"] = df["price"].str.replace("$", "", regex=False)
df["price"] = df["price"].str.replace(",", "", regex=False)
df["price"] = df["price"].astype(float)

df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})

num_cols = [
    "host_response_rate",
    "host_acceptance_rate",
    "accommodates",
    "bedrooms",
    "beds",
    "review_scores_rating",
    "availability_365",
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

min_price = float(df["price"].min())
max_price = float(df["price"].max())

min_rating = float(df["review_scores_rating"].min())
max_rating = float(df["review_scores_rating"].max())

min_accom = int(df["accommodates"].min())
max_accom = int(df["accommodates"].max())

neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
room_types = sorted(df["room_type"].dropna().unique())

MODEL_PATH = Path(__file__).parent / "data" / "price_model.keras"
price_model = keras.models.load_model(MODEL_PATH)

def infer_rating(neigh, room_type, accommodates):
    dff = df.copy()
    if neigh:
        dff = dff[dff["neighbourhood_cleansed"] == neigh]
    if room_type:
        dff = dff[dff["room_type"] == room_type]
    if accommodates is not None:
        dff = dff[np.abs(dff["accommodates"] - accommodates) <= 1]
    dff = dff.dropna(subset=["review_scores_rating"])
    if len(dff) >= 10:
        return float(dff["review_scores_rating"].mean())
    return float(df["review_scores_rating"].dropna().mean())

def infer_rate(col, neigh, room_type, accommodates):
    dff = df.copy()
    if neigh:
        dff = dff[dff["neighbourhood_cleansed"] == neigh]
    if room_type:
        dff = dff[dff["room_type"] == room_type]
    if accommodates is not None:
        dff = dff[np.abs(dff["accommodates"] - accommodates) <= 1]
    dff = dff.dropna(subset=[col])
    if len(dff) >= 10:
        return float(dff[col].mean())
    return float(df[col].dropna().mean())

def median_col(col):
    return float(df[col].dropna().median())

def mode_bool_flag(col):
    s = df[col].dropna()
    if s.empty:
        return 0
    val = s.mode()[0]
    if isinstance(val, (bool, np.bool_)):
        return 1 if val else 0
    return 1 if str(val).lower() in ["t", "true", "1", "yes"] else 0

def mode_cat(col):
    s = df[col].dropna()
    if s.empty:
        return ""
    return str(s.mode()[0])

def infer_lat_lon(neigh):
    if neigh:
        dff = df[df["neighbourhood_cleansed"] == neigh]
        if not dff.empty:
            return float(dff["latitude"].mean()), float(dff["longitude"].mean())
    return float(df["latitude"].mean()), float(df["longitude"].mean())

def fmt_price(val: float) -> str:
    v = int(val)
    if v >= 1000:
        k = v / 1000
        if k < 10:
            return f"${k:.1f}k"
        else:
            return f"${k:.0f}k"
    else:
        return f"{v}"

min_mark_price = int(min_price)
max_mark_price = int(max_price)
price_marks = {
    min_mark_price: fmt_price(min_mark_price),
    max_mark_price: fmt_price(max_mark_price),
}

rating_marks = {i: str(i) for i in range(int(min_rating), int(max_rating) + 1)}

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    className="main-container",
    children=[
        html.Div(
            className="header",
            children=[
                html.H1("Airbnb Barcelona Dashboard", className="title"),
                html.P(
                    "Explora factores que afectan el precio, la calificación y la ubicación de los alojamientos.",
                    className="subtitle",
                ),
            ],
        ),

        html.Div(
            className="filters-row",
            children=[
                html.Div(
                    className="filter-card",
                    children=[
                        html.Label("Barrio (neighbourhood)"),
                        dcc.Dropdown(
                            id="f-neighbourhood",
                            options=[{"label": n, "value": n} for n in neighbourhoods],
                            value=None,
                            placeholder="Todos",
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    className="filter-card",
                    children=[
                        html.Label("Tipo de habitación"),
                        dcc.Dropdown(
                            id="f-room-type",
                            options=[{"label": r, "value": r} for r in room_types],
                            value=None,
                            placeholder="Todos",
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    className="filter-card",
                    children=[
                        html.Label("Capacidad mínima (accommodates)"),
                        dcc.Slider(
                            id="f-accommodates",
                            min=min_accom,
                            max=max_accom,
                            step=1,
                            value=min_accom,
                            marks={i: str(i) for i in range(min_accom, max_accom + 1) if i % 2 == 0},
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            className="filters-row",
            children=[
                html.Div(
                    className="filter-card",
                    children=[
                        html.Label("Rango de precio (USD)"),
                        dcc.RangeSlider(
                            id="f-price",
                            min=min_price,
                            max=max_price,
                            step=5,
                            value=[min_price, max_price],
                            marks=price_marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(
                    className="filter-card",
                    children=[
                        html.Label("Rango de rating"),
                        dcc.RangeSlider(
                            id="f-rating",
                            min=min_rating,
                            max=max_rating,
                            step=0.1,
                            value=[min_rating, max_rating],
                            marks=rating_marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(
                    className="filter-card",
                    children=[
                        html.Label("Superhost"),
                        dcc.Checklist(
                            id="f-superhost",
                            options=[{"label": "Solo Superhosts", "value": "only"}],
                            value=[],
                            inline=True,
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            className="kpi-row",
            children=[
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div("PRECIO PROMEDIO", className="kpi-label"),
                        html.Div(id="kpi-avg-price", className="kpi-value"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div("RATING PROMEDIO", className="kpi-label"),
                        html.Div(id="kpi-avg-rating", className="kpi-value"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div("NÚMERO DE LISTINGS", className="kpi-label"),
                        html.Div(id="kpi-count", className="kpi-value"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div("INGRESO ESTIMADO PROMEDIO (USD)", className="kpi-label"),
                        html.Div(id="kpi-avg-revenue", className="kpi-value"),
                    ],
                ),
            ],
        ),

        html.Div(
            className="charts-row",
            children=[
                html.Div(
                    className="chart-card",
                    children=[
                        html.H3(
                            "¿Cómo se relaciona el precio con la calificación del Airbnb (review_scores_rating)?",
                            className="chart-title",
                        ),
                        dcc.Graph(id="g-price-rating", style={"height": "380px"}),
                    ],
                ),
                html.Div(
                    className="chart-card",
                    children=[
                        html.H3(
                            "¿Cómo se distribuyen espacialmente las propiedades más caras?",
                            className="chart-title",
                        ),
                        dcc.Graph(id="g-map", style={"height": "380px"}),
                    ],
                ),
            ],
        ),

        html.Div(
            className="charts-row",
            children=[
                html.Div(
                    className="chart-card",
                    children=[
                        html.H3(
                            "¿Ser Superhost influye significativamente en el precio?",
                            className="chart-title",
                        ),
                        dcc.Graph(id="g-superhost-box", style={"height": "380px"}),
                    ],
                ),
                html.Div(
                    className="chart-card",
                    children=[
                        html.H3(
                            "¿Cuáles son los factores que más influyen en el precio de renta de una propiedad en Airbnb?",
                            className="chart-title",
                        ),
                        dcc.Graph(id="g-corr", style={"height": "380px"}),
                    ],
                ),
            ],
        ),

                html.Div(
                    className="prediction-section",
                    children=[
                        html.H2(
                            "Predicciones para un nuevo anuncio",
                            className="title",
                        ),
                html.Div(
                    className="filters-row",
                    children=[
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Label("Barrio (neighbourhood)"),
                                dcc.Dropdown(
                                    id="p-neighbourhood",
                                    options=[{"label": n, "value": n} for n in neighbourhoods],
                                    value=None,
                                    placeholder="Selecciona un barrio",
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Label("Tipo de habitación"),
                                dcc.Dropdown(
                                    id="p-room-type",
                                    options=[{"label": r, "value": r} for r in room_types],
                                    value=None,
                                    placeholder="Selecciona tipo de habitación",
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Label("Capacidad (accommodates)"),
                                dcc.Slider(
                                    id="p-accommodates",
                                    min=min_accom,
                                    max=max_accom,
                                    step=1,
                                    value=min_accom,
                                    marks={i: str(i) for i in range(min_accom, max_accom + 1) if i % 2 == 0},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="filters-row",
                    children=[
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Label("Número de habitaciones (bedrooms)"),
                                dcc.Input(
                                    id="p-bedrooms",
                                    type="number",
                                    min=0,
                                    step=1,
                                    value=1,
                                    className="number-input",
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Label("Número de camas (beds)"),
                                dcc.Input(
                                    id="p-beds",
                                    type="number",
                                    min=1,
                                    step=1,
                                    value=1,
                                    className="number-input",
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Label("Superhost"),
                                dcc.RadioItems(
                                    id="p-superhost",
                                    options=[
                                        {"label": "No", "value": False},
                                        {"label": "Sí", "value": True},
                                    ],
                                    value=False,
                                    inline=True,
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="filters-row",
                    children=[
                        html.Div(className="filter-card"),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.Div("Precio estimado por noche", className="kpi-label"),
                                html.Div(id="predicted-price", className="kpi-value"),
                            ],
                        ),
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.Div(
                                    "Probabilidad de ser recomendado",
                                    className="kpi-label",
                                ),
                                html.Div(
                                    id="predicted-recommend",
                                    className="kpi-value",
                                ),
                            ],
                        ),
                        html.Div(className="filter-card"),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    [
        Output("g-price-rating", "figure"),
        Output("g-map", "figure"),
        Output("g-superhost-box", "figure"),
        Output("g-corr", "figure"),
        Output("kpi-avg-price", "children"),
        Output("kpi-avg-rating", "children"),
        Output("kpi-count", "children"),
        Output("kpi-avg-revenue", "children"),
    ],
    [
        Input("f-neighbourhood", "value"),
        Input("f-room-type", "value"),
        Input("f-accommodates", "value"),
        Input("f-price", "value"),
        Input("f-rating", "value"),
        Input("f-superhost", "value"),
    ],
)
def update_dashboard(
    neighbourhoods_sel,
    room_types_sel,
    accom_min,
    price_range,
    rating_range,
    superhost_flag,
):

    dff = df.copy()

    if neighbourhoods_sel:
        dff = dff[dff["neighbourhood_cleansed"].isin(neighbourhoods_sel)]
    if room_types_sel:
        dff = dff[dff["room_type"].isin(room_types_sel)]
    if accom_min is not None:
        dff = dff[dff["accommodates"] >= accom_min]
    if price_range:
        low, high = price_range
        dff = dff[(dff["price"] >= low) & (dff["price"] <= high)]
    if rating_range and "review_scores_rating" in dff.columns:
        r_low, r_high = rating_range
        dff = dff[
            (dff["review_scores_rating"] >= r_low)
            & (dff["review_scores_rating"] <= r_high)
        ]
    if superhost_flag and "only" in superhost_flag:
        dff = dff[dff["host_is_superhost"] == True]

    if dff.empty:
        return (
            go.Figure(),
            go.Figure(),
            go.Figure(),
            BETA_FIG,
            "—",
            "—",
            "0",
            "—",
        )

    fig_scatter = px.scatter(
        dff,
        x="review_scores_rating",
        y="price",
        color="room_type",
        hover_data=["neighbourhood_cleansed", "accommodates"],
        labels={
            "review_scores_rating": "Rating",
            "price": "Precio (USD)",
            "room_type": "Tipo de habitación",
        },
    )
    fig_scatter.update_layout(legend=dict(orientation="h", y=-0.2))
    fig_scatter = apply_fig_style(fig_scatter)

    fig_map = px.scatter_mapbox(
        dff,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        hover_name="neighbourhood_cleansed",
        color_continuous_scale="tealrose",
        zoom=11,
        height=400,
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map = apply_fig_style(fig_map)

    fig_box = px.box(
        dff,
        x="host_is_superhost",
        y="price",
        color="host_is_superhost",
        labels={
            "host_is_superhost": "Superhost",
            "price": "Precio (USD)",
        },
    )
    fig_box.update_xaxes(ticktext=["No", "Sí"], tickvals=[False, True])
    fig_box.update_layout(showlegend=False)
    fig_box = apply_fig_style(fig_box)

    avg_price = f"${dff['price'].mean():.0f}"
    avg_rating = (
        f"{dff['review_scores_rating'].mean():.2f}"
        if "review_scores_rating" in dff.columns
        else "—"
    )
    count_listings = f"{len(dff):,}".replace(",", ".")
    avg_rev = (
        f"${dff['estimated_revenue_l365d'].mean():.0f}"
        if "estimated_revenue_l365d" in dff.columns
        else "N/A"
    )

    return (
        fig_scatter,
        fig_map,
        fig_box,
        BETA_FIG,
        avg_price,
        avg_rating,
        count_listings,
        avg_rev,
    )

@app.callback(
    [
        Output("predicted-price", "children"),
        Output("predicted-recommend", "children"),
    ],
    [
        Input("p-neighbourhood", "value"),
        Input("p-room-type", "value"),
        Input("p-accommodates", "value"),
        Input("p-bedrooms", "value"),
        Input("p-beds", "value"),
        Input("p-superhost", "value"),
    ],
)
def predict_price(
    neigh,
    room_type,
    accommodates,
    bedrooms,
    beds,
    is_superhost,
):
    if neigh is None or room_type is None or accommodates is None:
        return "Completa barrio, tipo de habitación y capacidad", "—"

    est_rating = infer_rating(neigh, room_type, accommodates)
    resp_rate = infer_rate("host_response_rate", neigh, room_type, accommodates)
    acc_rate = infer_rate("host_acceptance_rate", neigh, room_type, accommodates)

    if bedrooms is None:
        bedrooms = float(df["bedrooms"].dropna().median())
    if beds is None:
        beds = float(df["beds"].dropna().median())
    if is_superhost is None:
        is_superhost = False

    host_listings_count = median_col("host_listings_count")
    host_total_listings_count = median_col("host_total_listings_count")
    minimum_nights = median_col("minimum_nights")
    maximum_nights = median_col("maximum_nights")
    number_of_reviews = median_col("number_of_reviews")
    availability_eoy = median_col("availability_eoy")
    estimated_revenue_l365d = median_col("estimated_revenue_l365d")
    calculated_host_listings_count = median_col("calculated_host_listings_count")
    reviews_per_month = median_col("reviews_per_month")

    host_has_profile_pic_flag = mode_bool_flag("host_has_profile_pic")
    instant_bookable_flag = mode_bool_flag("instant_bookable")
    property_type = mode_cat("property_type")
    lat, lon = infer_lat_lon(neigh)
    review_scores_accuracy = median_col("review_scores_accuracy")

    X_dict = {
        "host_is_superhost": tf.constant([1 if is_superhost else 0], dtype=tf.int64),
        "host_has_profile_pic": tf.constant([host_has_profile_pic_flag], dtype=tf.int64),
        "instant_bookable": tf.constant([instant_bookable_flag], dtype=tf.int64),

        "host_listings_count": tf.constant([host_listings_count], dtype=tf.float32),
        "host_total_listings_count": tf.constant([host_total_listings_count], dtype=tf.float32),
        "accommodates": tf.constant([float(accommodates)], dtype=tf.float32),
        "bedrooms": tf.constant([float(bedrooms)], dtype=tf.float32),
        "beds": tf.constant([float(beds)], dtype=tf.float32),
        "minimum_nights": tf.constant([minimum_nights], dtype=tf.float32),
        "maximum_nights": tf.constant([maximum_nights], dtype=tf.float32),
        "availability_365": tf.constant([float(df["availability_365"].dropna().median())], dtype=tf.float32),
        "number_of_reviews": tf.constant([number_of_reviews], dtype=tf.float32),
        "availability_eoy": tf.constant([availability_eoy], dtype=tf.float32),
        "estimated_occupancy_l365d": tf.constant([float(df["estimated_occupancy_l365d"].dropna().median())], dtype=tf.float32),
        "estimated_revenue_l365d": tf.constant([estimated_revenue_l365d], dtype=tf.float32),
        "calculated_host_listings_count": tf.constant([calculated_host_listings_count], dtype=tf.float32),
        "host_response_rate": tf.constant([float(resp_rate)], dtype=tf.float32),
        "host_acceptance_rate": tf.constant([float(acc_rate)], dtype=tf.float32),
        "latitude": tf.constant([lat], dtype=tf.float32),
        "longitude": tf.constant([lon], dtype=tf.float32),
        "review_scores_rating": tf.constant([float(est_rating)], dtype=tf.float32),
        "review_scores_accuracy": tf.constant([review_scores_accuracy], dtype=tf.float32),
        "reviews_per_month": tf.constant([reviews_per_month], dtype=tf.float32),

        "neighbourhood_cleansed": tf.constant([neigh], dtype=tf.string),
        "property_type": tf.constant([property_type], dtype=tf.string),
        "room_type": tf.constant([room_type], dtype=tf.string),
    }

    pred_price = float(price_model.predict(X_dict, verbose=0)[0][0])
    price_text = f"${pred_price:,.0f}".replace(",", ".")

    prob_rec = float(recommend_model.predict(X_dict, verbose=0)[0][0])
    prob_text = f"{prob_rec * 100:.1f}%"

    return price_text, prob_text

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
