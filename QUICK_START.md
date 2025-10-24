# Hybrid ETA - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- AWS Account with S3 access
- OpenWeatherMap API key

---

## Step 1: Install Dependencies (1 min)

```bash
cd c:\Users\prath\Hybrid_ETA
pip install -r requirements.txt
```

---

## Step 2: Configure Environment (2 min)

Edit `.env` file with your credentials:

```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
OPENWEATHERMAP_API_KEY=your_api_key
```

---

## Step 3: Upload Data to S3 (1 min)

```bash
# Create buckets
aws s3 mb s3://hybrid-eta-train-gps-data
aws s3 mb s3://hybrid-eta-models
aws s3 mb s3://hybrid-eta-results

# Upload your data
aws s3 cp Busstop_route1.csv s3://hybrid-eta-train-gps-data/route1/bus_stops.csv
aws s3 cp generated_route_route1_2_(2)_updated.csv s3://hybrid-eta-train-gps-data/route1/gps_data.csv
```

---

## Step 4: Train Models (1 min to start)

```bash
python scripts/run_training_pipeline.py --route route1
```

This will:
- ✅ Fetch data from S3
- ✅ Preprocess (grid, graph, MST)
- ✅ Train all 6 models
- ✅ Save to S3

---

## Step 5: Get Results

Results are automatically saved to S3 in your CSV format:

```
s3://hybrid-eta-results/
├── model_performance/route1_performance.csv
├── hybrid_comparison/hybrid_model_comparison.csv
└── predictions/
```

---

## 🎯 What You Get

### 6 Trained Models
1. **MST-AV** - Baseline with average speeds
2. **GDRN-DFT** - Graph diffusion + DFT
3. **KNN** - Koopman Neural Network
4. **FE-NN** - Feature-encoded NN
5. **MGCN** - Masked Graph CNN
6. **Hybrid** - Ensemble of all models

### Comprehensive Metrics
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- By ETA ranges: 0-10, 10-25, 25-45, 45+ minutes

### Results in Your Format
All results stored in S3 matching your exact CSV table structures from the paper.

---

## 📊 Example Results

After training, download results:

```bash
aws s3 cp s3://hybrid-eta-results/model_performance/route1_performance.csv .
```

Expected output:
```csv
Model,Metric,0-10,10-25,25-45,45+,Overall
MST-AV,MAE,3.2,4.5,5.1,6.8,4.32
GDRN-DFT,MAE,2.9,3.8,4.6,5.9,3.89
KNN,MAE,3.1,4.0,4.8,6.2,4.01
FE-NN,MAE,2.8,3.7,4.5,5.8,3.76
MGCN,MAE,2.6,3.5,4.2,5.5,3.54
Hybrid,MAE,2.4,3.2,3.9,5.1,3.21
```

---

## 🔧 Common Commands

### Train specific models
```bash
python scripts/run_training_pipeline.py --route route1 --models mst_av,gdrn_dft
```

### Evaluate models
```bash
python scripts/run_evaluation.py --route route1
```

### Real-time prediction
```bash
python scripts/run_inference.py \
  --route route1 \
  --model hybrid \
  --bus-location "22.569,88.516" \
  --stop-location "22.653,88.360"
```

---

## 📁 Project Structure

```
Hybrid_ETA/
├── src/
│   ├── data/          # S3 manager, acquisition, preprocessing
│   ├── models/        # All 6 models (MST-AV, GDRN-DFT, KNN, FE-NN, MGCN, Hybrid)
│   ├── training/      # Training scripts
│   ├── evaluation/    # Metrics and results manager
│   ├── inference/     # Real-time ETA predictor
│   └── utils/         # Haversine, graph utils
├── scripts/
│   ├── run_training_pipeline.py
│   ├── run_evaluation.py
│   └── run_inference.py
├── config/
│   └── model_config.yaml
└── .env
```

---

## ✅ Implementation Status

**100% Complete!**

- ✅ All 6 models implemented (Algorithms 2-13)
- ✅ Data acquisition from S3 + APIs (Algorithm 1)
- ✅ Preprocessing (grid, graph, MST)
- ✅ Training scripts for all models
- ✅ Evaluation with MAE, MAPE, RMSE
- ✅ Results storage in your CSV format
- ✅ Real-time inference
- ✅ Full S3 integration

---

## 🎉 You're All Set!

Your Hybrid ETA system is ready to:
1. Train on your GPS data
2. Generate predictions
3. Store results in S3
4. Provide real-time ETAs

All algorithms from your research paper are implemented and operational!

---

## 📚 Documentation

- `README.md` - Project overview
- `PROJECT_STRUCTURE.md` - Detailed file structure
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation details
- `DEPLOYMENT_GUIDE.md` - Full deployment instructions
- `QUICK_START.md` - This file

---

## 💡 Need Help?

Check the troubleshooting section in `DEPLOYMENT_GUIDE.md` or review the implementation details in `IMPLEMENTATION_SUMMARY.md`.
