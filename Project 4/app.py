from flask import Flask
import pandas as pd
import geopandas as gpd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import joblib
import os
from sklearn.cluster import KMeans

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("cleaned_covid_data.csv")
print("✅ Dataset loaded successfully!")

# Load the world map from an online GeoJSON source
GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"

try:
    print("Fetching world map...")
    world_map = gpd.read_file(GEOJSON_URL)
    print("✅ World map loaded successfully!")
except Exception as e:
    print("❌ Error loading world map:", e)
    world_map = None

# Merge outbreak data with the world map if loaded successfully
if world_map is not None:
    outbreak_map = world_map.merge(df, how="left", left_on="ADMIN", right_on="Country")
else:
    outbreak_map = None

# Load the trained model
MODEL_PATH = "TrainForest_model.pkl"
model = None

print("Loading model...")
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Error loading model:", e)
        model = None
else:
    print(f"❌ Model file '{MODEL_PATH}' not found!")

# Ensure clustering is performed if missing in the dataset
def generate_clusters():
    if 'cluster' not in df.columns:
        print("⚠️ 'cluster' column missing. Performing K-Means clustering...")
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[['total_cases', 'growth_rate']])
    else:
        print("✅ 'cluster' column exists in dataset.")

generate_clusters()

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server, routes_pathname_prefix="/dashboard/")

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("COVID-19 Data Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        # ✅ Geographical Spread
        dcc.Tab(label='Geographical Spread', children=[
            html.H3("COVID Outbreak Duration - Interactive Map", style={'textAlign': 'center'}),
            dcc.Graph(id="covid-map", figure=px.choropleth(
                outbreak_map, geojson=world_map.geometry, locations=outbreak_map.index,
                color="duration", hover_name="ADMIN", title="Interactive COVID Outbreak Duration Map",
                color_continuous_scale="Reds", labels={"duration": "Outbreak Duration (Days)"})),
        ]),

        # ✅ Feature Importance (Fixed)
        dcc.Tab(label='Feature Importance', children=[
            html.H3("Important Features for Epidemic Duration Prediction"),
            dcc.Graph(id='feature-importance')
        ]),

        # ✅ Epidemic Duration for Each Variant (Fixed & Interactive)
        dcc.Tab(label='Epidemic Duration per Variant', children=[
            html.H3("Select a COVID Variant to View Epidemic Duration"),
            dcc.Dropdown(
                id='variant-dropdown',
                options=[{'label': v, 'value': v} for v in df['variant'].unique()],
                value=df['variant'].unique()[0],
                multi=False
            ),
            dcc.Graph(id='epidemic-duration')
        ]),

        # ✅ Cluster Analysis (Single Optimized Chart)
        dcc.Tab(label='Cluster Analysis', children=[
            html.H3("COVID-19 Variant Clustering"),
            dcc.Graph(id='clustering')
        ]),

        # ✅ Predict Epidemic Duration
        dcc.Tab(label='Predict Epidemic Duration', children=[
            html.H3("Enter Features for Prediction"),
            html.Label("Number of Sequences"),
            dcc.Input(id='num_seqs', type='number', placeholder='Number of Sequences', value=1000),

            html.Label("Total Cases"),
            dcc.Input(id='total_cases', type='number', placeholder='Total Cases', value=50000),

            html.Label("Mortality Case Ratio"),
            dcc.Input(id='mortality_case_ratio', type='number', placeholder='Mortality Case Ratio', value=2.5),

            html.Label("Growth Rate"),
            dcc.Input(id='growth_rate', type='number', placeholder='Growth Rate', value=1.1),

            html.Label("Total Deaths"),
            dcc.Input(id='total_deaths', type='number', placeholder='Total Deaths', value=1000),

            html.Label("Censored (1=Yes, 0=No)"),
            dcc.Input(id='censored', type='number', placeholder='Censored (1=Yes, 0=No)', value=0),

            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output')
        ])
    ])
])

# **Fix: Feature Importance Callback**
@app.callback(
    Output('feature-importance', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_feature_importance(n_clicks):
    importance_scores = pd.DataFrame({
        'Feature': ['num_seqs', 'mortality_case_ratio', 'total_cases', 'total_deaths', 'growth_rate', 'censored'],
        'Importance': [0.15, 0.08, 0.03, 0.04, 0.09, 0.68]  # Example values
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(importance_scores, x='Importance', y='Feature', orientation='h', title="Important Features for Epidemic Duration Prediction")
    return fig

# **Fix: Epidemic Duration per Variant Callback**
@app.callback(
    Output('epidemic-duration', 'figure'),
    Input('variant-dropdown', 'value')
)
def update_epidemic_duration(selected_variant):
    filtered_df = df[df['variant'] == selected_variant]
    fig = px.histogram(filtered_df, x='duration', nbins=30, title=f'Epidemic Duration for {selected_variant}')
    return fig

# **Fix: Clustering Callback (Single Chart)**
@app.callback(
    Output('clustering', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_clustering(n_clicks):
    fig = px.scatter(df, x='total_cases', y='growth_rate', color=df['cluster'].astype(str),
                     title="Clusters of COVID-19 Variants", labels={'cluster': "Cluster"})
    return fig

# **Fix: Prediction Callback**
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('num_seqs', 'value'),
     Input('mortality_case_ratio', 'value'),
     Input('total_cases', 'value'),
     Input('total_deaths', 'value'),
     Input('growth_rate', 'value'),
     Input('censored', 'value')]
)
def predict_duration(n_clicks, num_seqs, mortality_case_ratio, total_cases, total_deaths, growth_rate, censored):
    if n_clicks > 0 and model:
        features_df = pd.DataFrame([[num_seqs, mortality_case_ratio, total_cases, total_deaths, growth_rate, censored]],
                                   columns=["num_seqs", "mortality_case_ratio", "total_cases", "total_deaths", "growth_rate", "censored"])
        prediction = model.predict(features_df)[0]
        return f'Predicted Epidemic Duration: {prediction:.2f} days'
    return ""

if __name__ == "__main__":
    app.run_server(debug=True)