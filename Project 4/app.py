from flask import Flask, render_template
import pandas as pd
import dash
from dash import dcc, html  # Updated import
from dash.dependencies import Input, Output
import plotly.express as px
import joblib

# Load the cleaned dataset
print("Loading dataset...")
df = pd.read_csv("cleaned_covid_data.csv")
print("✅ Dataset loaded successfully!")

# Load the trained Random Forest model
print("Loading model...")
try:
    model = joblib.load("TrainForest_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None  # Prevent crashing if model fails to load

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server, routes_pathname_prefix="/dashboard/")

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("COVID-19 Data Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Feature Importance', children=[
            html.Div([
                dcc.Graph(
                    id='feature-importance',
                    figure=px.bar(
                        df.nlargest(10, 'mortality_rate'),
                        x='mortality_rate',
                        y='variant',
                        orientation='h',
                        title='Top 10 Variants by Mortality Rate'
                    )
                )
            ])
        ]),
        
        dcc.Tab(label='Epidic Duration Distribution', children=[
            html.Div([
                dcc.Graph(
                    id='epidemic-duration',
                    figure=px.histogram(
                        df, x='duration', nbins=20, 
                        title='Distribution of Epidemic Duration'
                    )
                )
            ])
        ]),
        
        dcc.Tab(label='Clustering Analysis', children=[
            html.Div([
                dcc.Graph(
                    id='clustering',
                    figure=px.scatter(
                        df, x='total_cases', y='growth_rate', color='variant',
                        title='Clusters of COVID-19 Variants'
                    )
                )
            ])
        ]),
        
        dcc.Tab(label='Variant Trends', children=[
            html.Div([
                dcc.Dropdown(
                    id='variant-dropdown',
                    options=[{'label': v, 'value': v} for v in df['variant'].unique()],
                    value=df['variant'].unique()[0],
                    multi=False
                ),
                dcc.Graph(id='variant-trend')
            ])
        ]),
        
        dcc.Tab(label='Predict Epidemic Duration', children=[
            html.Div([
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
])

@app.callback(
    Output('variant-trend', 'figure'),
    [Input('variant-dropdown', 'value')]
)
def update_line_chart(selected_variant):
    filtered_df = df[df['variant'] == selected_variant]
    fig = px.line(filtered_df, x='censure_date', y='total_cases', title=f'Total Cases Over Time for {selected_variant}')
    return fig

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
    if n_clicks > 0:
        print(f"Inputs received: {num_seqs}, {mortality_case_ratio}, {total_cases}, {total_deaths}, {growth_rate}, {censored}")
        
        if None in [num_seqs, mortality_case_ratio, total_cases, total_deaths, growth_rate, censored]:
            return "❌ Error: Missing input values. Please fill all fields."
        
        try:
            features_df = pd.DataFrame([[num_seqs, mortality_case_ratio, total_cases, total_deaths, growth_rate, censored]],
                                       columns=["num_seqs", "mortality_case_ratio", "total_cases", "total_deaths", "growth_rate", "censored"])
            
            print("Feature DataFrame:", features_df)
            prediction = model.predict(features_df)[0]
            print(f"✅ Prediction successful: {prediction:.2f} days")
            return f'Predicted Epidemic Duration (Random Forest): {prediction:.2f} days'
        
        except Exception as e:
            print("❌ Error during prediction:", e)
            return f"❌ Error: {e}"
    
    return ""

if __name__ == "__main__":
    app.run_server(debug=True)