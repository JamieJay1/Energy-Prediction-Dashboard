import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load data
df = pd.read_csv('energy_dashboard/data/predictions.csv', parse_dates=['timestamp'])

# Initialize Dash
app = dash.Dash(__name__)
app.title = "Energy Forecast Dashboard"

# Layout
app.layout = html.Div([
    html.H2("🏢 Energy Forecast Dashboard - "),

    dcc.Dropdown(
        id='building-dropdown',
        options=[{'label': b, 'value': b} for b in df['building_id'].unique()],
        value=df['building_id'].unique()[0],
        clearable=False
    ),

    dcc.Checklist(
        id='overlay-options',
        options=[
            {'label': 'Show anomalies', 'value': 'anomaly'},
            {'label': 'Overlay air temperature', 'value': 'temp'}
        ],
        value=['anomaly']
    ),

    dcc.Graph(id='prediction-graph')
])

# Callback
@app.callback(
    dash.dependencies.Output('prediction-graph', 'figure'),
    dash.dependencies.Input('building-dropdown', 'value'),
    dash.dependencies.Input('overlay-options', 'value')
)
def update_graph(selected_building, overlays):
    dff = df[df['building_id'] == selected_building]

    fig = px.line(dff, x='timestamp', y='y_pred', title=f'Predictions for {selected_building}', labels={'y_pred': 'Predicted Usage'})

    fig.add_scatter(x=dff['timestamp'], y=dff['y_true'], mode='lines', name='Actual Usage')

    if 'anomaly' in overlays:
        anomalies = dff[dff['is_anomaly'] == 1]
        fig.add_scatter(x=anomalies['timestamp'], y=anomalies['y_true'], mode='markers', name='Anomalies', marker=dict(color='red', size=7))

    if 'temp' in overlays:
        fig.add_scatter(x=dff['timestamp'], y=dff['airTemperature'], mode='lines', name='Air Temp', yaxis='y2')

        fig.update_layout(
            yaxis2=dict(title='Air Temp (°C)', overlaying='y', side='right', showgrid=False)
        )

    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return fig

# Run
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8050, debug=False)

