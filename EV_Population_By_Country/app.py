from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and data
model = joblib.load('random_forest_ev_model.pkl')
feature_names = joblib.load('model_features.pkl')
df = pd.read_csv('preprocessed_ev_data.csv')

# Get list of counties
available_counties = sorted(df['County'].unique())

def get_latest_features_for_county(df, county_name):
    df_county = df[df['County'].str.lower() == county_name.lower()]
    if df_county.empty:
        raise ValueError(f"❌ County '{county_name}' not found in training data.")
    latest = df_county.sort_values('Date').iloc[-1]
    return {
        'months_since_start': latest['months_since_start'] + 1,
        'county_encoded': latest['county_encoded'],
        'ev_total_lag1': latest['Electric Vehicle (EV) Total'],
        'ev_total_lag2': latest['ev_total_lag1'],
        'ev_total_lag3': latest['ev_total_lag2'],
        'ev_total_roll_mean_3': latest['ev_total_roll_mean_3'],
        'ev_total_pct_change_1': latest['ev_total_pct_change_1'],
        'ev_total_pct_change_3': latest['ev_total_pct_change_3'],
        'ev_growth_slope': latest['ev_growth_slope']
    }

@app.route("/")
def index():
    return render_template("index.html", counties=available_counties)

@app.route("/predict", methods=["POST"])
def predict():
    county = request.form["county"]
    total_vehicles = int(request.form["total_vehicles"])
    try:
        features = get_latest_features_for_county(df, county)
        X_input = pd.DataFrame([features])
        ev_total_pred = model.predict(X_input)[0]
        ev_percent_pred = (ev_total_pred / total_vehicles) * 100

        return render_template(
            "result.html",
            county=county,
            ev_total_pred=round(ev_total_pred, 2),
            ev_percent_pred=round(ev_percent_pred, 2),
            features=features,
            is_low=ev_percent_pred < 0.01
        )
    except Exception as e:
        return render_template("error.html", message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
