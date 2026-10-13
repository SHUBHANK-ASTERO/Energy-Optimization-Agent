# Energy Optimization Agent — Complete AIML Project

## SDG 13 – Climate Action

A complete Flask + Machine Learning prototype that monitors energy consumption, predicts future usage, detects unusual spikes, estimates cost/CO₂, and generates optimization recommendations.

### Main features
- AI/ML energy prediction using Random Forest Regression
- Automatic model training on the local machine
- Energy consumption line chart
- Device-wise energy breakdown
- Anomaly detection
- AI optimization recommendations
- Electricity cost estimation
- CO₂ emission estimation
- What-if prediction simulator
- Responsive professional dashboard

### Tech stack
Python, Flask, Pandas, NumPy, Scikit-learn, Joblib, HTML, CSS, JavaScript, Chart.js.

### Folder structure
```text
Energy_Optimization_Agent_Complete/
├── app.py
├── model_utils.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── energy_data.csv
├── models/
│   └── energy_model.joblib
├── templates/
│   └── index.html
└── static/
    └── style.css
```

### Run in VS Code

Open the folder containing `app.py`.

```powershell
pip install -r requirements.txt
python app.py
```

Open this in Chrome/Edge:
```text
http://127.0.0.1:5000
```

### If you want to regenerate the dataset
Stop Flask with Ctrl+C and run:
```powershell
python generate_dataset.py
python train_model.py
python app.py
```

### Important
The application automatically trains a new model if the model file is missing or cannot be loaded. This helps avoid scikit-learn model-version compatibility problems.

### Project workflow
```text
Energy Data
   ↓
Preprocessing
   ↓
Random Forest Prediction
   ↓
Anomaly Detection
   ↓
Optimization Agent
   ↓
Cost + CO₂ Estimation
   ↓
Energy Saving Recommendations
```

### Academic/project outcome
The system demonstrates how machine learning can support energy efficiency and climate action by forecasting consumption, identifying abnormal usage and recommending energy-saving actions.

### Future scope
- IoT smart-meter integration
- Real-time device switching
- Electricity tariff optimization
- Mobile application
- Cloud deployment
- Advanced time-series models
- Multi-building monitoring
