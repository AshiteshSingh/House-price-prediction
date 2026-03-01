import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
import datetime
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
MODEL_PATH = BASE / "custom_hybrid_model.pkl"
KERAS_PATH = BASE / "custom_hybrid_keras.keras"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__)

artifact     = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
model_xgb    = artifact["model_xgb"]
model_lgb    = artifact["model_lgb"]
model_gbr    = artifact["model_gbr"]
model_et     = artifact["model_et"]
meta         = artifact["meta_learner"]
model_dnn    = tf.keras.models.load_model(KERAS_PATH)
log.info("All models loaded successfully.")

CITY_COORDS = {
    "Ahmedabad":  (23.0225, 72.5714),
    "Bengaluru":  (12.9716, 77.5946),
    "Chennai":    (13.0827, 80.2707),
    "Delhi NCR":  (28.6139, 77.2090),
    "Hyderabad":  (17.3850, 78.4867),
    "Kolkata":    (22.5744, 88.3629),
    "Mumbai":     (19.0760, 72.8777),
    "Pune":       (18.5204, 73.8567),
}

CITIES      = sorted(CITY_COORDS.keys())
PROP_TYPES  = ["Apartment", "Builder Floor", "Independent House",
               "Penthouse", "Row House", "Studio", "Villa"]
FURNISHINGS = ["Furnished", "Semi-Furnished", "Unfurnished"]
PARKINGS    = ["Basement", "Covered", "Open", "Stilt"]
FACINGS     = ["North", "South", "East", "West",
               "North-East", "North-West", "South-East", "South-West"]
BUILDING_TYPES = ["Apartment Complex", "Bungalow", "Gated Community",
                  "Independent", "Society", "Township"]


@app.route("/")
def index():
    return render_template("index.html",
                           cities=CITIES,
                           prop_types=PROP_TYPES,
                           furnishings=FURNISHINGS,
                           parkings=PARKINGS,
                           facings=FACINGS,
                           building_types=BUILDING_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        city = data.get("city", "Mumbai")
        lat, lon = CITY_COORDS.get(city, (19.0760, 72.8777))

        super_area  = float(data.get("super_area", 1000))
        buildup     = float(data.get("buildup_area", super_area * 0.85))
        carpet      = float(data.get("carpet_area", super_area * 0.70))
        age         = int(data.get("age", 5))
        year_built  = datetime.datetime.now().year - age

        row = {
            "ListingID":             "WEB-QUERY",
            "City":                  city,
            "Locality":              data.get("locality", "Unknown"),
            "PropertyType":          data.get("property_type", "Apartment"),
            "BHK":                   int(data.get("bhk", 2)),
            "Bathrooms":             float(data.get("bathrooms", 2)),
            "Balconies":             float(data.get("balconies", 1)),
            "Furnishing":            data.get("furnishing", "Semi-Furnished"),
            "SuperBuiltUpArea_sqft": super_area,
            "BuiltUpArea_sqft":      buildup,
            "CarpetArea_sqft":       carpet,
            "Floor":                 int(data.get("floor", 3)),
            "TotalFloors":           int(data.get("total_floors", 10)),
            "Parking":               data.get("parking", "Covered"),
            "BuildingType":          data.get("building_type", "Apartment Complex"),
            "YearBuilt":             float(year_built),
            "AgeYears":              age,
            "Facing":                data.get("facing", "East"),
            "AmenitiesCount":        int(data.get("amenities", 5)),
            "IsRERARegistered":      bool(data.get("rera", True)),
            "RERAID":                data.get("rera_id", "N/A"),
            "Latitude":              lat,
            "Longitude":             lon,
        }

        df_input = pd.DataFrame([row])
        X_p = preprocessor.transform(df_input)

        v_xgb = model_xgb.predict(X_p)
        v_lgb = model_lgb.predict(X_p)
        v_gbr = model_gbr.predict(X_p)
        v_et  = model_et.predict(X_p)
        v_dnn = model_dnn.predict(X_p, verbose=0).flatten()

        pred_log = meta.predict(np.column_stack((v_xgb, v_lgb, v_gbr, v_et, v_dnn)))
        price    = float(np.expm1(pred_log)[0])

        price = max(500_000, min(price, 1_500_000_000))

        return jsonify({
            "price_inr":   round(price, 2),
            "price_lakh":  round(price / 1e5, 2),
            "price_crore": round(price / 1e7, 4),
        })

    except Exception as exc:
        log.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    log.info("Starting server at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
