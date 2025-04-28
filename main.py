import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
import os

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define constants - removed special teams positions
HIGH_CONTACT_POSITIONS = ['OL', 'DL', 'RB', 'LB', 'TE', 'FB']
HIGH_SPEED_POSITIONS = ['WR', 'DB', 'CB', 'S', 'FS', 'SS']
OTHER_POSITIONS = ['QB']  # Removed LS

# Load the saved model and preprocessor
def load_model():
    try:
        model = joblib.load('best_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except:
        return None, None

# Get temperature category from numerical temperature
def get_temperature_category(temp):
    if temp <= 32:
        return "Freezing"
    elif temp <= 60:
        return "Cold"
    elif temp <= 75:
        return "Moderate"
    elif temp <= 85:
        return "Warm"
    else:
        return "Hot"

# Define dropdown options - removed special teams play types
roster_positions = sorted(HIGH_CONTACT_POSITIONS + HIGH_SPEED_POSITIONS + OTHER_POSITIONS)
stadium_types = ['Indoor', 'Outdoor']
field_types = ['Turf', 'Grass']
weather_conditions = ['Clear', 'Rain', 'Snow', 'Cloudy', 'Windy']
play_types = ['Run', 'Pass']  # Removed Punt, Field Goal, Kickoff

# Check if model exists
model_exists = os.path.exists('best_model.joblib') and os.path.exists('preprocessor.joblib')

# Main app layout
app.layout = html.Div([
    html.Div([
        html.H1("Football Injury Risk Predictor", className="app-header"),
        html.P("Machine learning-based tool for predicting player injury risks", className="header-description"),
    ], className="header"),
    
    # Warning message if model doesn't exist
    html.Div([
        html.Div([
            html.I(className="fas fa-exclamation-triangle warning-icon"),
            html.Span("Warning: Model not found! Please run your training script (train.py) first to generate the model files.")
        ], className="warning-message")
    ], id="model-warning", style={'display': 'none' if model_exists else 'block'}),
    
    # Input form
    html.Div([
        html.Div([
            html.Div([
                html.Label("Player Position:", className="form-label"),
                dcc.Dropdown(
                    id='roster-position',
                    options=[{'label': pos, 'value': pos} for pos in roster_positions],
                    placeholder="Select Position",
                    className="dropdown"
                ),
                
                html.Label("Play Type:", className="form-label"),
                dcc.Dropdown(
                    id='play-type',
                    options=[{'label': pt, 'value': pt} for pt in play_types],
                    placeholder="Select Play Type",
                    className="dropdown"
                ),
                
                html.Label("Temperature (°F):", className="form-label"),
                dcc.Input(
                    id='temperature',
                    type='number',
                    min=-10,
                    max=120,
                    step=1,
                    placeholder="Enter temperature",
                    className="input-field"
                ),
            ], className="form-column"),
            
            html.Div([
                html.Label("Field Type:", className="form-label"),
                dcc.Dropdown(
                    id='field-type',
                    options=[{'label': ft, 'value': ft} for ft in field_types],
                    placeholder="Select Field Type",
                    className="dropdown"
                ),
                
                html.Label("Stadium Type:", className="form-label"),
                dcc.Dropdown(
                    id='stadium-type',
                    options=[{'label': st, 'value': st} for st in stadium_types],
                    placeholder="Select Stadium Type",
                    className="dropdown"
                ),
                
                html.Label("Weather Condition:", className="form-label"),
                dcc.Dropdown(
                    id='weather',
                    options=[{'label': w, 'value': w} for w in weather_conditions],
                    placeholder="Select Weather",
                    className="dropdown"
                ),
            ], className="form-column"),
        ], className="form-row"),
        
        html.Button(
            "Calculate Injury Risk", 
            id="predict-button", 
            className="predict-button",
            disabled=not model_exists
        ),
    ], id="input-form", className="input-form"),
    
    # Results section (initially hidden)
    html.Div([
        html.H2("Injury Risk Assessment", className="section-title"),
        
        # Risk meter
        html.Div([
            html.Div([
                html.Div(id="risk-level-display", className="risk-level"),
                html.Div([
                    dcc.Graph(id="risk-gauge", config={'displayModeBar': False}, className="gauge-chart")
                ], className="gauge-container")
            ], className="risk-meter-container")
        ], className="risk-meter-section"),
        
        # Risk details
        html.Div([
            html.Div([
                html.H3("Risk Details", className="subsection-title"),
                html.Table([
                    html.Tr([html.Td("Base Risk:"), html.Td(id="base-prob")]),
                    html.Tr([html.Td("Adjusted Risk:"), html.Td(id="adjusted-prob")]),
                    html.Tr([html.Td("Adjustment:"), html.Td(id="adjustment-text")]),
                ], className="risk-details-table")
            ], className="risk-details"),
            
            html.Div([
                html.H3("Selected Conditions", className="subsection-title"),
                html.Table(id="conditions-table", className="conditions-table")
            ], className="selected-conditions")
        ], className="risk-details-section"),
        
        # Risk factors
        html.Div([
            html.H3("Risk Factors to Consider", className="subsection-title"),
            html.Ul(id="risk-factors-list", className="risk-factors-list")
        ], className="risk-factors-section"),
        
        # Recommendations
        html.Div([
            html.H3("Recommendations", className="subsection-title"),
            html.Div(id="recommendations", className="recommendations-content")
        ], className="recommendations-section"),
        
        # Back button
        html.Button(
            "New Prediction", 
            id="reset-button", 
            className="reset-button"
        ),
    ], id="results-section", style={'display': 'none'}),
    
    # Footer
    html.Div([
        html.P("© 2025 Football Injury Risk Predictor - ML Model for Trainers"),
    ], className="footer"),
    
    # Store component to save prediction data
    dcc.Store(id='prediction-data'),
], className="container")

# Callback for prediction
@callback(
    [Output('prediction-data', 'data'),
     Output('results-section', 'style'),
     Output('input-form', 'style')],
    [Input('predict-button', 'n_clicks')],
    [State('roster-position', 'value'),
     State('play-type', 'value'),
     State('temperature', 'value'),
     State('field-type', 'value'),
     State('stadium-type', 'value'),
     State('weather', 'value')],
    prevent_initial_call=True
)
def predict_injury_risk(n_clicks, roster_position, play_type, temperature, 
                         field_type, stadium_type, weather):
    # Validate input
    if None in [roster_position, play_type, temperature, field_type, stadium_type, weather]:
        return None, {'display': 'none'}, {'display': 'block'}
    
    # Load model and preprocessor
    model, preprocessor = load_model()
    
    if model is None:
        return None, {'display': 'none'}, {'display': 'block'}
    
    # Prepare input data
    input_data = {}
    
    input_data['Temperature'] = float(temperature)
    temp_cat = get_temperature_category(input_data['Temperature'])
    input_data['TemperatureCategory'] = temp_cat
    
    input_data['FieldType'] = field_type
    input_data['StadiumType'] = stadium_type
    input_data['RosterPosition'] = roster_position
    input_data['Weather'] = weather
    input_data['PlayType'] = play_type
    
    # Create interaction features
    input_data['Turf_Hot'] = 1 if input_data['FieldType'] == 'Turf' and temp_cat == 'Hot' else 0
    input_data['Cold_Outdoor'] = 1 if temp_cat in ['Freezing', 'Cold'] and input_data['StadiumType'] == 'Outdoor' else 0
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    input_processed = preprocessor.transform(input_df)
    
    # Make prediction
    prediction_prob = model.predict_proba(input_processed)[0, 1]
    
    # Position baseline risk adjustments
    position_baseline_adjustments = {
        # Offensive positions
        "QB": 0.6,      # Quarterbacks have lowest injury rates
        "OL": 0.90,     # Offensive linemen have higher injury risk
        "RB": 1.0,      # Running backs as baseline
        "FB": 0.85,      # Fullbacks slightly lower than RBs
        "WR": 0.75,     # Wide receivers
        "TE": 0.85,      # Tight ends 
        
        # Defensive positions
        "DL": 0.90,     # Defensive linemen have higher risk
        "LB": 0.95,      # Linebackers
        "CB": 0.80,     # Cornerbacks
        "S": 0.85,      # Safeties
        "FS": 0.85,     # Free safeties
        "SS": 0.85,     # Strong safeties
        "DB": 0.80,     # Defensive backs generally
    }
    
    # Apply position-specific baseline adjustment
    position = input_data['RosterPosition']
    baseline_adjustment = position_baseline_adjustments.get(position, 1.0)
    prediction_prob = prediction_prob * baseline_adjustment
    
    # Track baseline adjustment text
    baseline_text = f"{(1-baseline_adjustment)*100:.0f}% position baseline reduction"
    
    # Position-specific play type adjustments with updated values
    play_type = input_data['PlayType'].lower()
    
    risk_adjustment = 1.0  # Default, no adjustment
    adjustment_text = None
    
    # More subtle adjustments with 5% increase for high-risk combinations
    if play_type == 'run':
        if position in ['RB']:
            # Running backs on run plays have highest risk adjustment
            risk_adjustment = 1.10  # 10% increase
            adjustment_text = f"+10% run-specific {position} adjustment, with {baseline_text}"
        elif position in ['OL', 'DL', 'LB', 'FB']:
            # Other high-contact positions on run plays
            risk_adjustment = 1.07  # 7% increase
            adjustment_text = f"+7% run-specific {position} adjustment, with {baseline_text}"
        else:
            adjustment_text = baseline_text
    elif play_type == 'pass':
        if position in ['WR', 'DB', 'CB']:
            # High-speed positions on pass plays
            risk_adjustment = 1.05  # 5% increase
            adjustment_text = f"+5% pass-specific {position} adjustment, with {baseline_text}"
        elif position in ['TE', 'S', 'FS', 'SS']:
            # Secondary high-speed positions
            risk_adjustment = 1.03  # 3% increase
            adjustment_text = f"+3% pass-specific {position} adjustment, with {baseline_text}"
        else:
            adjustment_text = baseline_text
    else:
        adjustment_text = baseline_text
    
    # Apply the play-type adjustment to the probability
    adjusted_prob = min(0.99, prediction_prob * risk_adjustment)  # Cap at 99%
    
    # Determine risk level
    if adjusted_prob >= 0.7:
        risk_level = "HIGH"
        risk_color = "red"
    elif adjusted_prob >= 0.4:
        risk_level = "MEDIUM"
        risk_color = "orange"
    else:
        risk_level = "LOW" 
        risk_color = "green"
    
    # Compile risk factors
    risk_factors = []
    
    # Surface-related factors
    if input_data['Turf_Hot'] == 1:
        risk_factors.append("Playing on turf in hot conditions increases injury risk")
    if input_data['FieldType'].lower() == 'turf':
        risk_factors.append("Artificial turf surfaces may increase certain injury risks")
    
    # Weather and temperature factors
    if input_data['Cold_Outdoor'] == 1:
        risk_factors.append("Cold outdoor conditions may affect muscle flexibility and injury risk")
    if temp_cat in ["Hot"]:
        risk_factors.append("Hot temperatures can increase fatigue and dehydration risk")
    if input_data['Weather'].lower() in ['rain', 'snow', 'sleet', 'wet']:
        risk_factors.append("Precipitation can create slippery playing conditions")
    
    # Position and play type specific risk factors
    if play_type == 'run':
        if position in HIGH_CONTACT_POSITIONS:
            risk_factors.append(f"{position} has elevated injury risk on run plays due to high-contact nature")
            if position == 'RB':
                risk_factors.append("Running backs are at particular risk during run plays due to repeated impacts")
            elif position in ['OL', 'DL']:
                risk_factors.append("Linemen face increased injury risk during run blocking/tackling situations")
        else:
            risk_factors.append(f"{position} has standard injury risk on run plays")
    elif play_type == 'pass':
        if position in HIGH_SPEED_POSITIONS:
            risk_factors.append(f"{position} has elevated injury risk on pass plays due to high-speed movements")
            if position == 'WR':
                risk_factors.append("Wide receivers are vulnerable when stretched out for catches or after catch tackles")
            elif position in ['CB', 'DB', 'S', 'FS', 'SS']:
                risk_factors.append("Defensive backs face risk during quick direction changes and open-field tackles")
        else:
            risk_factors.append(f"{position} has standard injury risk on pass plays")
    
    # Recommendation text
    if risk_level == "HIGH":
        recommendation = "High injury risk detected. Consider additional player monitoring, reduced workload, or modification of specific play types for this player in these conditions."
    elif risk_level == "MEDIUM":
        recommendation = "Moderate injury risk detected. Exercise caution and monitor player for signs of fatigue or discomfort during these conditions."
    else:
        recommendation = "Low injury risk detected. Standard monitoring protocols should be sufficient for this player in these conditions."
    
    # Store all prediction data
    prediction_data = {
        'input_data': input_data,
        'base_prob': round(prediction_prob * 100, 1),
        'adjusted_prob': round(adjusted_prob * 100, 1),
        'adjustment_text': adjustment_text,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_factors': risk_factors,
        'recommendation': recommendation
    }
    
    return prediction_data, {'display': 'block'}, {'display': 'none'}

# Callback to update the results display
@callback(
    [Output('risk-level-display', 'children'),
     Output('risk-level-display', 'style'),
     Output('risk-gauge', 'figure'),
     Output('base-prob', 'children'),
     Output('adjusted-prob', 'children'),
     Output('adjustment-text', 'children'),
     Output('conditions-table', 'children'),
     Output('risk-factors-list', 'children'),
     Output('recommendations', 'children'),
     Output('recommendations', 'className')],
    Input('prediction-data', 'data'),
    prevent_initial_call=True
)
def update_results(data):
    if data is None:
        return dash.no_update
    
    # Create risk gauge figure with valid colors
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = data['adjusted_prob'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Injury Risk %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': data['risk_color']},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},  # Fixed color name
                {'range': [70, 100], 'color': "lightcoral"}  # Fixed color name
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': data['adjusted_prob']
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        font={"color": "darkblue", "family": "Arial"}
    )
    
    # Create conditions table
    conditions_table = [
        html.Tr([html.Td("Position:"), html.Td(data['input_data']['RosterPosition'])]),
        html.Tr([html.Td("Play Type:"), html.Td(data['input_data']['PlayType'])]),
        html.Tr([html.Td("Temperature:"), html.Td(f"{data['input_data']['Temperature']}°F ({data['input_data']['TemperatureCategory']})")]),
        html.Tr([html.Td("Field Type:"), html.Td(data['input_data']['FieldType'])]),
        html.Tr([html.Td("Stadium:"), html.Td(data['input_data']['StadiumType'])]),
        html.Tr([html.Td("Weather:"), html.Td(data['input_data']['Weather'])])
    ]
    
    # Create risk factors list
    risk_factors_list = [html.Li(factor) for factor in data['risk_factors']]
    
    # Determine recommendations class
    recommendations_class = f"recommendations-content {data['risk_level'].lower()}-risk"
    
    return (
        f"RISK LEVEL: {data['risk_level']}",                 # risk level display
        {'backgroundColor': data['risk_color']},             # risk level style
        fig,                                                 # gauge figure
        f"{data['base_prob']}%",                             # base probability
        f"{data['adjusted_prob']}%",                         # adjusted probability
        data['adjustment_text'] if data['adjustment_text'] else "None",  # adjustment text
        conditions_table,                                    # conditions table
        risk_factors_list,                                   # risk factors list
        data['recommendation'],                              # recommendations
        recommendations_class                                # recommendations class
    )

# Callback for reset button
@callback(
    [Output('input-form', 'style', allow_duplicate=True),
     Output('results-section', 'style', allow_duplicate=True),
     Output('roster-position', 'value'),
     Output('play-type', 'value'),
     Output('temperature', 'value'),
     Output('field-type', 'value'),
     Output('stadium-type', 'value'),
     Output('weather', 'value')],
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_form(n_clicks):
    return (
        {'display': 'block'},  # show input form
        {'display': 'none'},   # hide results
        None, None, None, None, None, None  # reset all inputs
    )

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Football Injury Risk Predictor</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.1);
                padding: 25px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            .app-header {
                color: #2C3E50;
                margin-bottom: 10px;
            }
            .header-description {
                color: #7F8C8D;
                font-size: 16px;
            }
            .form-row {
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }
            .form-column {
                flex: 1;
                min-width: 300px;
                padding: 0 15px;
            }
            .form-label {
                font-weight: bold;
                margin-bottom: 8px;
                display: block;
                color: #2C3E50;
            }
            .dropdown, .input-field {
                width: 100%;
                margin-bottom: 20px;
            }
            .predict-button, .reset-button {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 12px 20px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                width: 100%;
                margin-top: 10px;
                transition: background-color 0.3s;
            }
            .predict-button:hover, .reset-button:hover {
                background-color: #2980B9;
            }
            .predict-button:disabled {
                background-color: #BDC3C7;
                cursor: not-allowed;
            }
            .reset-button {
                background-color: #7F8C8D;
                margin-top: 30px;
            }
            .reset-button:hover {
                background-color: #95A5A6;
            }
            .section-title {
                color: #2C3E50;
                margin-bottom: 20px;
                text-align: center;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .subsection-title {
                color: #34495E;
                margin-top: 0;
                margin-bottom: 15px;
            }
            .risk-meter-section {
                text-align: center;
                margin-bottom: 30px;
            }
            .risk-meter-container {
                display: inline-block;
                text-align: center;
            }
            .risk-level {
                font-size: 20px;
                font-weight: bold;
                color: white;
                background-color: #E74C3C;
                padding: 10px 20px;
                border-radius: 5px;
                margin-bottom: 15px;
                display: inline-block;
            }
            .gauge-container {
                width: 100%;
                max-width: 400px;
                margin: 0 auto;
            }
            .risk-details-section {
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 30px;
                border: 1px solid #eee;
                border-radius: 5px;
                padding: 15px;
                background-color: #f9f9f9;
            }
            .risk-details, .selected-conditions {
                flex: 1;
                min-width: 300px;
                padding: 0 15px;
            }
            .risk-details-table, .conditions-table {
                width: 100%;
                border-collapse: collapse;
            }
            .risk-details-table td, .conditions-table td {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            .risk-details-table td:first-child, .conditions-table td:first-child {
                font-weight: bold;
                width: 40%;
            }
            .risk-factors-section {
                margin-bottom: 30px;
                border: 1px solid #eee;
                border-radius: 5px;
                padding: 15px;
                background-color: #f9f9f9;
            }
            .risk-factors-list {
                margin: 0;
                padding-left: 20px;
            }
            .risk-factors-list li {
                margin-bottom: 5px;
            }
            .recommendations-section {
                margin-bottom: 20px;
            }
            .recommendations-content {
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            .high-risk {
                background-color: #FADBD8;
                color: #943126;
            }
            .medium-risk {
                background-color: #FCF3CF;
                color: #9A7D0A;
            }
            .low-risk {
                background-color: #D5F5E3;
                color: #1E8449;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                color: #7F8C8D;
                font-size: 14px;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }
            .warning-message {
                background-color: #FEEFB3;
                color: #9F6000;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                display: flex;
                align-items: center;
            }
            .warning-icon {
                margin-right: 10px;
                font-size: 18px;
            }
            @media (max-width: 768px) {
                .form-column, .risk-details, .selected-conditions {
                    flex: 100%;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)