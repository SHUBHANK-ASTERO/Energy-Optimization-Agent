import pandas as pd
from model_utils import train_model, MODEL_PATH
import joblib

df = pd.read_csv("data/energy_data.csv")
model, metrics = train_model(df)
joblib.dump(model, MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")
print(f"MAE: {metrics['mae']:.3f} kWh")
print(f"R2 : {metrics['r2']:.3f}")
