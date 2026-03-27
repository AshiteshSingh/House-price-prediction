import os, warnings, logging
warnings.filterwarnings("ignore")

from pathlib import Path
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from huggingface_hub import hf_hub_download

BASE = Path(__file__).parent
MODEL_PATH = BASE / "mldl.pkl"
HF_REPO_ID  = "jarvisai1234/house-price-prediction-india"

# Auto-download model from Hugging Face if not present locally
if not MODEL_PATH.exists():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Downloading mldl.pkl from Hugging Face...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="mldl.pkl",
        local_dir=str(BASE),
        local_dir_use_symlinks=False,
    )
    logging.getLogger(__name__).info("Model downloaded to %s", downloaded)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from GitHub Pages

artifact     = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
model_xgb    = artifact["model_xgb"]
model_lgb    = artifact["model_lgb"]
model_gbr    = artifact["model_gbr"]
model_et     = artifact["model_et"]
meta         = artifact["meta_learner"]

import torch.nn as nn

def silu_act(x):
    return x * torch.sigmoid(x)

class TransformerResNetDNN(nn.Module):
    def __init__(self, input_dim, units1=1024, units2=512, units3=256, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.res1 = nn.Linear(units1, units2)
        self.bn_res1 = nn.BatchNorm1d(units2)
        self.drop_res = nn.Dropout(dropout_rate * 0.75)
        
        self.res2 = nn.Linear(units2, units2)
        self.bn_res2 = nn.BatchNorm1d(units2)
        
        self.x_proj = nn.Linear(units1, units2)
        
        self.attn_proj = nn.Linear(input_dim, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(64)
        
        self.merge_fc1 = nn.Linear(units2 + 64, units3)
        self.bn_merge = nn.BatchNorm1d(units3)
        self.drop_merge = nn.Dropout(dropout_rate * 0.5)
        self.merge_fc2 = nn.Linear(units3, 64)
        self.output_fc = nn.Linear(64, 1)

    def forward(self, x):
        h = silu_act(self.bn1(self.fc1(x)))
        h = self.drop1(h)

        res = silu_act(self.bn_res1(self.res1(h)))
        res = self.drop_res(res)
        res = silu_act(self.bn_res2(self.res2(res)))
        h = silu_act(self.x_proj(h) + res)

        seq = self.attn_proj(x).unsqueeze(1)
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out = self.attn_norm(attn_out + seq)
        attn_flat = attn_out.mean(dim=1)

        merged = torch.cat([h, attn_flat], dim=1)
        merged = silu_act(self.bn_merge(self.merge_fc1(merged)))
        merged = self.drop_merge(merged)
        merged = silu_act(self.merge_fc2(merged))
        return self.output_fc(merged).squeeze(-1)

def predict_pytorch_dnn(model, device, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy()

DNN_PATH = BASE / "custom_hybrid_dnn.pt"
dnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CITY_CENTERS = {
    'Mumbai': (19.0760, 72.8777), 'Delhi NCR': (28.6139, 77.2090),
    'Bengaluru': (12.9716, 77.5946), 'Chennai': (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867), 'Pune': (18.5204, 73.8567),
    'Kolkata': (22.5726, 88.3639), 'Ahmedabad': (23.0225, 72.5714)
}

def haversine_dist(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    r_lat1, r_lon1 = math.radians(lat1), math.radians(lon1)
    r_lat2, r_lon2 = math.radians(lat2), math.radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = math.sin(dlat/2)**2 + math.cos(r_lat1)*math.cos(r_lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def engineer_features(df):
    df_feat = df.copy()
    num_cols = ['SuperBuiltUpArea_sqft', 'BHK', 'Bathrooms', 'CarpetArea_sqft', 'Floor', 'TotalFloors', 'AmenitiesCount', 'AgeYears']
    for col in num_cols:
        if col in df_feat.columns: df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
    
    if 'SuperBuiltUpArea_sqft' in df_feat.columns and 'BHK' in df_feat.columns:
        df_feat['Area_per_BHK'] = df_feat['SuperBuiltUpArea_sqft'] / df_feat['BHK'].clip(lower=1)
        df_feat['BHK_Area_Interaction'] = df_feat['BHK'] * df_feat['SuperBuiltUpArea_sqft']
    if 'SuperBuiltUpArea_sqft' in df_feat.columns and 'Bathrooms' in df_feat.columns:
        df_feat['Area_per_Bath'] = df_feat['SuperBuiltUpArea_sqft'] / df_feat['Bathrooms'].clip(lower=1)
    if 'SuperBuiltUpArea_sqft' in df_feat.columns and 'CarpetArea_sqft' in df_feat.columns:
        df_feat['Carpet_to_Super_Ratio'] = df_feat['CarpetArea_sqft'] / df_feat['SuperBuiltUpArea_sqft'].clip(lower=1)
    if 'Floor' in df_feat.columns and 'TotalFloors' in df_feat.columns:
        df_feat['Floor_Ratio'] = df_feat['Floor'] / df_feat['TotalFloors'].clip(lower=1)
        df_feat['Is_Top_Floor'] = (df_feat['Floor'] == df_feat['TotalFloors']).astype(int)
        df_feat['Is_Ground_Floor'] = (df_feat['Floor'] <= 1).astype(int)
    if 'AgeYears' in df_feat.columns:
        bins = [-1, 1, 5, 10, 20, 100]
        labels = ['Brand New', '1-5 Yrs', '5-10 Yrs', '10-20 Yrs', '20+ Yrs']
        df_feat['Age_Binned'] = pd.cut(df_feat['AgeYears'].fillna(5), bins=bins, labels=labels, right=True).astype(str)
    if 'BHK' in df_feat.columns and 'Bathrooms' in df_feat.columns:
        df_feat['Bath_per_BHK'] = df_feat['Bathrooms'] / df_feat['BHK'].clip(lower=1)
        df_feat['Unusual_Bath_Ratio'] = (df_feat['Bath_per_BHK'] > 1.5).astype(int)
    if 'SuperBuiltUpArea_sqft' in df_feat.columns:
        df_feat['Log_Area'] = np.log1p(df_feat['SuperBuiltUpArea_sqft'].clip(lower=1))
    if 'AmenitiesCount' in df_feat.columns and 'SuperBuiltUpArea_sqft' in df_feat.columns and 'BHK' in df_feat.columns:
        df_feat['Luxury_Score'] = (df_feat['AmenitiesCount'] * df_feat['SuperBuiltUpArea_sqft']) / df_feat['BHK'].clip(lower=1)
    if 'Latitude' in df_feat.columns and 'Longitude' in df_feat.columns and 'City' in df_feat.columns:
        dists = []
        for _, row in df_feat.iterrows():
            city = str(row['City'])
            lat, lon = row['Latitude'], row['Longitude']
            if pd.isna(lat) or pd.isna(lon) or city not in CITY_CENTERS: dists.append(-1.0)
            else: dists.append(haversine_dist(lat, lon, CITY_CENTERS[city][0], CITY_CENTERS[city][1]))
        df_feat['City_Center_Distance_km'] = dists
    return df_feat

input_dim = 34

model_dnn = TransformerResNetDNN(input_dim).to(dnn_device)
if DNN_PATH.exists():
    model_dnn.load_state_dict(torch.load(DNN_PATH, map_location=dnn_device, weights_only=True))
    model_dnn.eval()
    log.info("PyTorch DNN loaded on %s", dnn_device)
else:
    log.warning("PyTorch DNN weights not found at %s", DNN_PATH)
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
        df_input = engineer_features(df_input)
        
        X_p = preprocessor.transform(df_input)

        v_xgb = model_xgb.predict(X_p)
        v_lgb = model_lgb.predict(X_p)
        v_gbr = model_gbr.predict(X_p)
        v_et  = model_et.predict(X_p)
        v_dnn = predict_pytorch_dnn(model_dnn, dnn_device, X_p)

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
    log.info("Starting server at http://0.0.0.0:7860")
    app.run(debug=False, host="0.0.0.0", port=7860)
