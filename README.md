# Hybrid ETA Prediction System

A comprehensive bus arrival time prediction system implementing multiple models (MST-AV, GDRN-DFT, KNN, FE-NN, MGCN) and a hybrid ensemble approach.

## Project Structure

```
Hybrid_ETA/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py          # Data acquisition from S3 and APIs
│   │   ├── preprocessing.py        # Data preprocessing and graph construction
│   │   └── s3_manager.py          # S3 operations manager
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mst_av.py              # MST-AV model
│   │   ├── gdrn_dft.py            # GDRN-DFT model
│   │   ├── knn.py                 # Koopman Neural Network
│   │   ├── fenn.py                # Feature-Encoded Neural Network
│   │   ├── mgcn.py                # Masked Graph Convolutional Network
│   │   ├── hybrid.py              # Hybrid ensemble model
│   │   └── benchmarks.py          # Benchmark models (DCRNN, STGCN, etc.)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_mst_av.py
│   │   ├── train_gdrn_dft.py
│   │   ├── train_knn.py
│   │   ├── train_fenn.py
│   │   ├── train_mgcn.py
│   │   ├── train_hybrid.py
│   │   └── train_benchmarks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # MAE, MAPE, RMSE calculations
│   │   ├── evaluator.py           # Model evaluation pipeline
│   │   └── results_manager.py     # Results storage to S3
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── eta_predictor.py       # Real-time ETA prediction
│   │   └── model_loader.py        # Load models from S3
│   └── utils/
│       ├── __init__.py
│       ├── graph_utils.py         # Graph construction utilities
│       ├── weather_api.py         # OpenWeatherMap API integration
│       ├── haversine.py           # Distance calculations
│       └── config.py              # Configuration management
├── config/
│   └── model_config.yaml          # Model hyperparameters
├── scripts/
│   ├── run_training_pipeline.py   # Full training pipeline
│   ├── run_evaluation.py          # Evaluation pipeline
│   └── run_inference.py           # Inference script
├── tests/
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_s3_operations.py
├── requirements.txt
├── .env
└── README.md
```

## S3 Bucket Structure

```
S3 Buckets:
├── hybrid-eta-train-gps-data/      # Training GPS data
├── hybrid-eta-test-gps-data/       # Test GPS data
├── hybrid-eta-models/              # Trained model weights
├── hybrid-eta-dft-coefficients/    # DFT coefficients
├── hybrid-eta-weather-cache/       # Cached weather data
└── hybrid-eta-results/             # Evaluation results (NEW)
    ├── model_performance/
    │   ├── route1_performance.csv
    │   ├── route2_performance.csv
    │   └── route3_performance.csv
    ├── hybrid_comparison/
    │   └── hybrid_model_comparison.csv
    ├── baseline_comparison/
    │   └── baseline_model_comparison.csv
    ├── ablation_studies/
    │   ├── gdrn_dft_hyperparameters.csv
    │   ├── knn_ablation.csv
    │   ├── fenn_ablation.csv
    │   └── mgcn_ablation.csv
    └── predictions/
        ├── {timestamp}_predictions.csv
        └── {timestamp}_metrics.json
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
OPENWEATHERMAP_API_KEY=your_api_key
```

3. Run training pipeline:
```bash
python scripts/run_training_pipeline.py --route route1
```

4. Run evaluation:
```bash
python scripts/run_evaluation.py --route route1 --model all
```

5. Run inference:
```bash
python scripts/run_inference.py --bus-location "22.569,88.516" --stop-location "22.653,88.360"
```

## Models Implemented

1. **MST-AV**: Minimum Spanning Tree with Average Velocities
2. **GDRN-DFT**: Graph Diffusion RNN with Discrete Fourier Transform
3. **KNN**: Koopman Neural Network
4. **FE-NN**: Feature-Encoded Neural Network
5. **MGCN**: Masked Graph Convolutional Network
6. **HYB(n)**: Hybrid ensemble combining top-n models
7. **Benchmarks**: DCRNN, STGCN, GWNet, T-GCN, MTGNN, STFGNN, ST-ResNet, ST-GConv

## Results Storage

All results are automatically stored in S3 bucket `hybrid-eta-results/` with the following format:
- Performance metrics: CSV files with MAE, MAPE, RMSE
- Predictions: Timestamped prediction files
- Model artifacts: Saved in `hybrid-eta-models/`
