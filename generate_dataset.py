from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "energy_data.csv"

rng = np.random.default_rng(42)
n = 1000
ts = pd.date_range("2026-08-01", periods=n, freq="h")
hour = ts.hour.to_numpy()

temperature = 25 + 7*np.sin((hour - 7)*np.pi/12) + rng.normal(0, 1.4, n)
occupancy = np.where(
    (hour >= 8) & (hour <= 18),
    rng.integers(6, 26, n),
    rng.integers(0, 7, n)
)

lighting = np.where((hour >= 7) & (hour <= 21), 0.15 + occupancy*0.025, 0.04)
ac = np.maximum(temperature - 25, 0) * 0.22 + np.where((hour >= 10) & (hour <= 18), 0.7, 0.12)
computer = np.where((hour >= 8) & (hour <= 18), occupancy*0.055, 0.04)
other = 0.25 + rng.normal(0, 0.08, n)
energy = np.maximum(lighting + ac + computer + other + rng.normal(0, 0.18, n), 0.1)

# Add a few realistic abnormal spikes for anomaly detection.
spike_indices = rng.choice(np.arange(50, n), size=12, replace=False)
energy[spike_indices] += rng.uniform(2.5, 4.5, len(spike_indices))

previous = np.roll(energy, 1)
previous[0] = energy[0]

df = pd.DataFrame({
    "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
    "hour": hour,
    "temperature": temperature.round(2),
    "occupancy": occupancy,
    "lighting_kwh": lighting.round(2),
    "ac_kwh": ac.round(2),
    "computer_kwh": computer.round(2),
    "other_kwh": other.round(2),
    "previous_kwh": previous.round(2),
    "energy_kwh": energy.round(2)
})

OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Created {OUT}")
