import warnings; warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch

from trainme import TransformerResNetDNN, predict_pytorch_dnn, engineer_features

artifact = joblib.load("custom_hybrid_model.pkl")
preprocessor = artifact["preprocessor"]
models = artifact["models_dict"]
meta = artifact["meta_learner"]
eng_feat = artifact.get("engineer_features", engineer_features)

df_raw = (pd.read_csv("Training_dataset/train_part1.csv", on_bad_lines="skip")
        .dropna(axis=1, how="all").dropna(subset=["Price_INR"]))
df = eng_feat(df_raw)

_, X_val, _, y_val = train_test_split(
    df.drop(columns=["Price_INR"]), df["Price_INR"], test_size=0.15, random_state=42)

X_p = preprocessor.transform(X_val)
input_dim = X_p.shape[1]

dnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dnn = TransformerResNetDNN(input_dim).to(dnn_device)
model_dnn.load_state_dict(torch.load("custom_hybrid_dnn.pt", map_location=dnn_device, weights_only=True))
model_dnn.eval()

# Generate predictions for all base learners
val_preds = np.zeros((X_p.shape[0], len(models) + 1))
model_names = list(models.keys()) + ["Keras DNN"]
model_r2s = []

col_idx = 0
for name, m in models.items():
    p = m.predict(X_p)
    val_preds[:, col_idx] = p
    model_r2s.append(r2_score(y_val, np.expm1(p)))
    col_idx += 1

# DNN prediction
p_dnn = predict_pytorch_dnn(model_dnn, dnn_device, X_p)
val_preds[:, col_idx] = p_dnn
model_r2s.append(r2_score(y_val, np.expm1(p_dnn)))

preds = np.expm1(meta.predict(val_preds))
y_arr = np.array(y_val)
errors = y_arr - preds
abs_err = np.abs(errors)
pct_err = abs_err / y_arr * 100

r2 = r2_score(y_arr, preds)
mae = mean_absolute_error(y_arr, preds)
rmse = np.sqrt(mean_squared_error(y_arr, preds))
mape = float(pct_err.mean())

print(f"R2={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")

model_names.append("Meta Stack")
model_r2s.append(r2)

sns.set_theme(style="darkgrid", palette="muted")
fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle(
    f"Ultra Hybrid Model (v2) — Performance Report\nR² = {r2:.4f} | MAE = ₹{mae/1e6:.2f}M | MAPE = {mape:.2f}%",
    fontsize=16, fontweight="bold", y=1.02
)

ax = axes[0, 0]
sc = ax.scatter(y_arr / 1e6, preds / 1e6, c=abs_err / 1e6, cmap="plasma",
                alpha=0.6, edgecolors="none", s=35)
mn = min(y_arr.min(), preds.min()) / 1e6
mx = max(y_arr.max(), preds.max()) / 1e6
ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Fit")
plt.colorbar(sc, ax=ax, label="Abs Error (₹M)")
ax.set_xlabel("Actual Price (₹ Millions)")
ax.set_ylabel("Predicted Price (₹ Millions)")
ax.set_title(f"Actual vs Predicted  |  R² = {r2:.4f}")
ax.legend()

ax = axes[0, 1]
ax.scatter(y_arr / 1e6, errors / 1e6, alpha=0.5, c="steelblue", s=30, edgecolors="none")
ax.axhline(0, color="red", lw=2, linestyle="--", label="Zero Error")
ax.set_xlabel("Actual Price (₹ Millions)")
ax.set_ylabel("Residual = Actual − Predicted (₹M)")
ax.set_title("Residual Plot")
ax.legend()

ax = axes[1, 0]
sns.histplot(pct_err, bins=40, kde=True, ax=ax, color="coral", edgecolor="none")
ax.axvline(mape, color="navy", linestyle="--", lw=2, label=f"MAPE = {mape:.2f}%")
ax.set_xlabel("Percentage Error (%)")
ax.set_ylabel("Count")
ax.set_title("Error Distribution (% Error)")
ax.legend()

ax = axes[1, 1]
colors = sns.color_palette("husl", len(model_names))
bars = ax.barh(model_names, model_r2s, color=colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, model_r2s):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("R² Score")
ax.set_title("Individual Model vs Final Stack R²")
ax.set_xlim(0, 1.05)
ax.axvline(r2, color="red", linestyle="--", lw=1.5, label=f"Final={r2:.4f}")
ax.legend()

fig1.tight_layout()
fig1.savefig("performance_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved performance_dashboard.png")
plt.close(fig1)

fig2 = plt.figure(figsize=(13, 9))
ax3d = fig2.add_subplot(111, projection="3d")

scatter = ax3d.scatter(
    y_arr / 1e6,
    preds / 1e6,
    abs_err / 1e6,
    c=pct_err,
    cmap="inferno",
    alpha=0.7,
    s=30
)
cb = fig2.colorbar(scatter, ax=ax3d, pad=0.1, shrink=0.6)
cb.set_label("% Error", fontsize=11)

lim_min = min(y_arr.min(), preds.min()) / 1e6
lim_max = max(y_arr.max(), preds.max()) / 1e6
xs = np.linspace(lim_min, lim_max, 10)
ys = xs
Xs, Ys = np.meshgrid(xs, ys)
Zs = np.zeros_like(Xs)
ax3d.plot_surface(Xs, Ys, Zs, alpha=0.12, color="cyan")

ax3d.set_xlabel("Actual Price (₹M)", labelpad=10)
ax3d.set_ylabel("Predicted Price (₹M)", labelpad=10)
ax3d.set_zlabel("Abs Error (₹M)", labelpad=10)
ax3d.set_title(
    f"3D Performance Surface\nR²={r2:.4f} | MAPE={mape:.2f}% | Accuracy={100-mape:.2f}%",
    fontsize=13, fontweight="bold"
)
fig2.tight_layout()
fig2.savefig("performance_3d.png", dpi=150, bbox_inches="tight")
print("Saved performance_3d.png")
plt.close(fig2)

print("Done! Open 'performance_dashboard.png' and 'performance_3d.png' to view results.")
