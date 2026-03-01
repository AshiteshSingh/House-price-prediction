# 🏠 House Price Prediction

> A smart AI system that predicts house prices across India. Just enter your property details — city, area, BHK, floor, furnishing — and get an instant price estimate in seconds.

---

## 🎯 What Does This Project Do?

This project predicts **how much a house is worth** based on its features like city, size, number of rooms, floor number, age, parking, etc.

It uses **5 different AI models** working together like a team — each one makes its own prediction, and then a final "boss" model combines all their answers to give the most accurate price.

Think of it like asking 5 real estate experts and then taking the smartest average of their opinions.

---

## 📊 How Accurate Is It?

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Overall Accuracy** | **92.13%** | Out of every ₹100, the model is off by only ₹7.87 on average |
| **R² Score** | **0.9682** | The model explains 96.82% of the variation in house prices |
| **MAE** | **₹15.2 Lakhs** | Average error is about ₹15 Lakhs (on houses worth ₹20L–₹10Cr+) |
| **MAPE** | **7.87%** | Average percentage error across all predictions |

### Individual Model Scores

| Model | R² Score |
|-------|----------|
| XGBoost | 0.9424 |
| LightGBM | 0.9705 |
| GradientBoosting | 0.9703 |
| ExtraTrees | 0.9697 |
| Deep Neural Network | 0.9611 |
| **Final Stacked Model** | **0.9682** |

---

## 📈 Performance Graphs

### Actual vs Predicted Prices
<p align="center">
  <img src="hybrid_actual_vs_predicted.png" width="700" alt="Actual vs Predicted Prices"/>
</p>

> Points close to the red dashed line = accurate predictions. Most predictions are tightly packed along this line.

### 3D Residual Analysis
<p align="center">
  <img src="hybrid_3d_residuals.png" width="700" alt="3D Residual Plot"/>
</p>

### Full Performance Dashboard
<p align="center">
  <img src="performance_dashboard.png" width="800" alt="Performance Dashboard"/>
</p>

> Shows 4 views: scatter plot, residuals, error distribution, and per-model R² comparison.

### 3D Performance Surface
<p align="center">
  <img src="performance_3d.png" width="800" alt="3D Performance Surface"/>
</p>

### Additional Plots
<p align="center">
  <img src="actual_vs_predicted.png" width="600" alt="Actual vs Predicted"/>
  <img src="actual_vs_predicted_3d.png" width="600" alt="Actual vs Predicted 3D"/>
</p>

---

## 🧠 How It Works (Simple Explanation)

```
Property Details (City, BHK, Area, Floor, etc.)
                    ↓
         Clean & Prepare the Data
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
 XGBoost        LightGBM      3 More Models
 (Prediction)   (Prediction)   (Predictions)
    ↓               ↓               ↓
    └───────────────┼───────────────┘
                    ↓
         Meta-Learner (XGBoost)
       Combines all 5 predictions
                    ↓
          Final Price Estimate
```

**5 Models Used:**
1. **XGBoost** — Fast and accurate tree-based model
2. **LightGBM** — Another powerful tree model, great with large data
3. **GradientBoosting** — Classic ensemble method
4. **ExtraTrees** — Randomized decision trees for diversity
5. **Transformer DNN** — Deep learning with self-attention (like how ChatGPT processes data)

All 5 predictions are combined by a **Meta-Learner** (stacking) for the final answer.

---

## 🌐 Live Demo

👉 **[Try the Web App on GitHub Pages](https://ashiteshsingh.github.io/House-price-prediction/)**

---

## 📁 Project Structure

```
├── House_Price_Training.ipynb        # Training notebook (run this to train)
├── House_Price_Evaluation.ipynb      # Evaluation & graphs notebook
├── app.py                            # Flask web app (local hosting)
├── trainme.py                        # Training script
├── evaluate.py                       # Evaluation script
├── templates/
│   └── index.html                    # Web UI template
├── static/                           # CSS & JS files
├── docs/                             # GitHub Pages static site
│   └── index.html                    # Live demo page
├── requirements.txt                  # Python dependencies
├── Training_dataset/                 # Training & test CSV datasets
├── performance_dashboard.png         # Performance dashboard
├── performance_3d.png                # 3D performance surface
├── hybrid_actual_vs_predicted.png    # Actual vs predicted scatter
├── hybrid_3d_residuals.png           # 3D residual analysis
└── test_predictions.csv              # Predictions on test data
```

---

## 🚀 How to Use

### Option 1: Run Locally
```bash
git clone https://github.com/AshiteshSingh/House-price-prediction.git
cd House-price-prediction
pip install -r requirements.txt
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Option 2: Train From Scratch
Open `House_Price_Training.ipynb` in Jupyter Notebook and run all cells.

---

## 🛠️ Built With

| | Technology |
|--|-----------|
| 🤖 | XGBoost, LightGBM, Scikit-learn |
| 🧠 | TensorFlow / Keras (Transformer + ResNet) |
| 🔍 | Optuna (hyperparameter tuning) |
| 🌐 | Flask (web app) |
| 📊 | Matplotlib, Seaborn |

---

## 🏙️ Supported Cities

Mumbai • Delhi NCR • Bengaluru • Chennai • Hyderabad • Pune • Kolkata • Ahmedabad

---

<p align="center">
  <b>Made with ❤️ by Ashitesh</b>
</p>
