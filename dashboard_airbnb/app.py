import pandas as pd
from pathlib import Path

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# ESTILO BASE PARA GRÁFICAS
# -----------------------------
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


# -----------------------------
# CARGA Y LIMPIEZA DE DATOS
# -----------------------------
DATA_PATH = Path(__file__).parent / "data" / "airbnb_barcelona.csv"
df = pd.read_csv(DATA_PATH)

# limpiar columna price de $ y comas, y convertir a float
df["price"] = df["price"].astype(str)
df["price"] = df["price"].str.replace("$", "", regex=False)
df["price"] = df["price"].str.replace(",", "", regex=False)
df["price"] = df["price"].astype(float)

# convertir booleano superhost
df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})

# asegurar tipo numérico en columnas relevantes
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

# valores para filtros
min_price = float(df["price"].min())
max_price = float(df["price"].max())

min_rating = float(df["review_scores_rating"].min())
max_rating = float(df["review_scores_rating"].max())

min_accom = int(df["accommodates"].min())
max_accom = int(df["accommodates"].max())

neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
room_types = sorted(df["room_type"].dropna().unique())


# marcas para sliders
def fmt_price(val: float) -> str:
    v = int(val)
    if v >= 1000:
        k = v / 1000
        if k < 10:
            return f"${k:.1f}k"
        else:
            return f"${k:.0f}k"
    else:
        return f"${v}"


min_mark_price = int(min_price)
max_mark_price = int(max_price)
price_marks = {
    min_mark_price: fmt_price(min_mark_price),
    max_mark_price: fmt_price(max_mark_price),
}

rating_marks = {
    i: str(i) for i in range(int(min_rating), int(max_rating) + 1)
}


# -----------------------------
# INICIALIZAR APP
# -----------------------------
app = Dash(__name__)
server = app.server


# -----------------------------
# LAYOUT
# -----------------------------
app.layout = html.Div(
    className="main-container",
    children=[
        # HEADER
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

        # FILA 1 FILTROS
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

        # FILA 2 FILTROS
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
                            marks=price_marks,  # SOLO min y max
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

        # KPIs
        html.Div(
            className="kpi-row",
            children=[
                html.Div(className="kpi-card", children=[
                    html.Div("Precio promedio", className="kpi-label"),
                    html.Div(id="kpi-avg-price", className="kpi-value"),
                ]),
                html.Div(className="kpi-card", children=[
                    html.Div("Rating promedio", className="kpi-label"),
                    html.Div(id="kpi-avg-rating", className="kpi-value"),
                ]),
                html.Div(className="kpi-card", children=[
                    html.Div("Número de listings", className="kpi-label"),
                    html.Div(id="kpi-count", className="kpi-value"),
                ]),
                html.Div(className="kpi-card", children=[
                    html.Div("Ingreso estimado promedio (USD)", className="kpi-label"),
                    html.Div(id="kpi-avg-revenue", className="kpi-value"),
                ]),
            ],
        ),

        # GRAFICAS FILA 1
        html.Div(
            className="charts-row",
            children=[
                html.Div(className="chart-card", children=[
                    html.H3("Precio vs Rating", className="chart-title"),
                    dcc.Graph(id="g-price-rating", style={"height": "380px"}),
                ]),
                html.Div(className="chart-card", children=[
                    html.H3("Mapa de propiedades", className="chart-title"),
                    dcc.Graph(id="g-map", style={"height": "380px"}),
                ]),
            ],
        ),

        # GRAFICAS FILA 2
        html.Div(
            className="charts-row",
            children=[
                html.Div(className="chart-card", children=[
                    html.H3("Distribución de precio: Superhost vs No Superhost", className="chart-title"),
                    dcc.Graph(id="g-superhost-box", style={"height": "380px"}),
                ]),
                html.Div(className="chart-card", children=[
                    html.H3("Correlación de variables", className="chart-title"),
                    dcc.Graph(id="g-corr", style={"height": "380px"}),
                ]),
            ],
        ),
    ],
)


# -----------------------------
# CALLBACK
# -----------------------------
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
def update_dashboard(neighbourhoods_sel, room_types_sel, accom_min, price_range, rating_range, superhost_flag):

    dff = df.copy()

    # filtros
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
        dff = dff[(dff["review_scores_rating"] >= r_low) & (dff["review_scores_rating"] <= r_high)]
    if superhost_flag and "only" in superhost_flag:
        dff = dff[dff["host_is_superhost"] == True]

    if dff.empty:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), "—", "—", "0", "—"

    # scatter precio vs rating
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

    # mapa
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

    # boxplot superhost
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

    # heatmap correlación
    corr_cols = ["price", "review_scores_rating", "accommodates", "bedrooms", "beds", "availability_365"]
    corr = dff[corr_cols].corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            reversescale=True,
        )
    )
    fig_corr = apply_fig_style(fig_corr)

    # KPIs
    avg_price = f"${dff['price'].mean():.0f}"
    avg_rating = f"{dff['review_scores_rating'].mean():.2f}"
    count_listings = f"{len(dff):,}".replace(",", ".")
    avg_rev = f"${dff['estimated_revenue_l365d'].mean():.0f}" if "estimated_revenue_l365d" in dff.columns else "N/A"

    return fig_scatter, fig_map, fig_box, fig_corr, avg_price, avg_rating, count_listings, avg_rev


# -----------------------------
# EJECUCIÓN LOCAL
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
