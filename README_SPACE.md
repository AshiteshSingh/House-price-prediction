---
title: House Price Prediction India
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏠 House Price Prediction - India

A machine learning API that predicts Indian real estate prices using an Ultra Hybrid Ensemble model (LightGBM + XGBoost + GBR + ExtraTrees + PyTorch DNN) with **96.82% R² accuracy**.

## API Usage

**POST** `/predict`

```json
{
  "city": "Mumbai",
  "super_area": 1000,
  "bhk": 2,
  "bathrooms": 2,
  "property_type": "Apartment",
  "furnishing": "Semi-Furnished"
}
```

**Response:**
```json
{
  "price_inr": 12500000,
  "price_lakh": 125.0,
  "price_crore": 1.25
}
```
