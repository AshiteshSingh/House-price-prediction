# 🏠 House Price Prediction — Ultra Hybrid ML Pipeline

An industry-level house price prediction system for the Indian real estate market, combining **5 powerful models** into a stacking ensemble with an XGBoost meta-learner.

## 🧠 Model Architecture

| Layer | Models |
|-------|--------|
| **Base Learners** | XGBoost, LightGBM, GradientBoostingRegressor, ExtraTreesRegressor, Transformer+ResNet DNN (Keras) |
| **Meta Learner** | XGBoost (trained on stacked base predictions) |
| **Hyperparameter Tuning** | Optuna (30 trials × 3-Fold CV per model) |

## 📁 Project Structure

```
├── House_Price_Training.ipynb      # Full training pipeline notebook
├── House_Price_Evaluation.ipynb    # Evaluation & visualization notebook
├── app.py                          # Flask web app for live predictions
├── templates/
│   └── index.html                  # Web UI for the predictor
├── static/                         # Static assets
├── requirements.txt                # Python dependencies
├── PanTrainModel_data1/            # Training & test CSV datasets
├── performance_dashboard.png       # 2D evaluation dashboard
├── performance_3d.png              # 3D performance surface plot
├── hybrid_actual_vs_predicted.png  # Scatter plot
├── hybrid_3d_residuals.png         # 3D residual plot
└── test_predictions.csv            # Predictions on test data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Open and run `House_Price_Training.ipynb` in Jupyter Notebook or VS Code.

### 3. Evaluate

Open and run `House_Price_Evaluation.ipynb` to generate performance dashboards.

### 4. Run the Web App

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## 📊 Performance

The hybrid ensemble achieves strong prediction accuracy across 8 major Indian cities (Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad).

## 🛠️ Tech Stack

- **ML/DL**: XGBoost, LightGBM, Scikit-learn, TensorFlow/Keras
- **Tuning**: Optuna
- **Web**: Flask
- **Visualization**: Matplotlib, Seaborn
