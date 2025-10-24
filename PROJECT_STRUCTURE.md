# Hybrid ETA Project Structure

## Complete File Structure

```
Hybrid_ETA/
├── .env                                    # Environment variables (AWS credentials, API keys)
├── README.md                               # Project documentation
├── PROJECT_STRUCTURE.md                    # This file
├── requirements.txt                        # Python dependencies
│
├── config/
│   └── model_config.yaml                   # Model hyperparameters and configuration
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── s3_manager.py                   # S3 operations (COMPLETE)
│   │   ├── acquisition.py                  # Data acquisition from S3 and APIs (COMPLETE)
│   │   └── preprocessing.py                # Graph construction and preprocessing (COMPLETE)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mst_av.py                       # MST-AV model (TO IMPLEMENT)
│   │   ├── gdrn_dft.py                     # GDRN-DFT model (TO IMPLEMENT)
│   │   ├── knn.py                          # Koopman Neural Network (TO IMPLEMENT)
│   │   ├── fenn.py                         # Feature-Encoded NN (TO IMPLEMENT)
│   │   ├── mgcn.py                         # Masked GCN (TO IMPLEMENT)
│   │   ├── hybrid.py                       # Hybrid ensemble (TO IMPLEMENT)
│   │   └── benchmarks.py                   # Benchmark models (TO IMPLEMENT)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_mst_av.py                 # MST-AV training (TO IMPLEMENT)
│   │   ├── train_gdrn_dft.py               # GDRN-DFT training (TO IMPLEMENT)
│   │   ├── train_knn.py                    # KNN training (TO IMPLEMENT)
│   │   ├── train_fenn.py                   # FE-NN training (TO IMPLEMENT)
│   │   ├── train_mgcn.py                   # MGCN training (TO IMPLEMENT)
│   │   ├── train_hybrid.py                 # Hybrid training (TO IMPLEMENT)
│   │   └── train_benchmarks.py             # Benchmark training (TO IMPLEMENT)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                      # MAE, MAPE, RMSE (COMPLETE)
│   │   ├── evaluator.py                    # Evaluation pipeline (TO IMPLEMENT)
│   │   └── results_manager.py              # Results storage to S3 (COMPLETE)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── eta_predictor.py                # Real-time ETA prediction (TO IMPLEMENT)
│   │   └── model_loader.py                 # Load models from S3 (TO IMPLEMENT)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── haversine.py                    # Distance calculations (COMPLETE)
│       ├── graph_utils.py                  # Graph utilities (TO IMPLEMENT)
│       ├── weather_api.py                  # Weather API integration (TO IMPLEMENT)
│       └── config.py                       # Config loader (TO IMPLEMENT)
│
├── scripts/
│   ├── run_training_pipeline.py            # Main training script (COMPLETE)
│   ├── run_evaluation.py                   # Evaluation script (TO IMPLEMENT)
│   └── run_inference.py                    # Inference script (TO IMPLEMENT)
│
└── tests/
    ├── test_models.py                      # Model tests (TO IMPLEMENT)
    ├── test_preprocessing.py               # Preprocessing tests (TO IMPLEMENT)
    └── test_s3_operations.py               # S3 tests (TO IMPLEMENT)
```

## S3 Bucket Organization

### Input Buckets (Data Sources)
```
hybrid-eta-train-gps-data/
├── route1/
│   ├── gps_data.csv
│   ├── bus_stops.csv
│   └── preprocessed/
│       └── graph_data.pkl
├── route2/
└── route3/

hybrid-eta-test-gps-data/
├── route1/
├── route2/
└── route3/
```

### Model Storage
```
hybrid-eta-models/
├── route1/
│   ├── mst_av/
│   │   ├── model.pth
│   │   └── metadata.json
│   ├── gdrn_dft/
│   ├── knn/
│   ├── fenn/
│   ├── mgcn/
│   └── hybrid/
├── route2/
└── route3/
```

### Results Storage (Output)
```
hybrid-eta-results/
├── model_performance/
│   ├── route1_performance.csv
│   ├── route2_performance.csv
│   └── route3_performance.csv
│
├── hybrid_comparison/
│   └── hybrid_model_comparison.csv
│
├── baseline_comparison/
│   └── baseline_model_comparison.csv
│
├── ablation_studies/
│   ├── gdrn_dft_hyperparameters.csv
│   ├── knn_ablation.csv
│   ├── fenn_ablation.csv
│   └── mgcn_ablation.csv
│
└── predictions/
    ├── 20250125_013000_route1_predictions.csv
    ├── 20250125_013000_route1_metrics.json
    └── 20250125_013000_route1_summary.json
```

## Key Features Implemented

### ✅ Complete Modules

1. **S3Manager** (`src/data/s3_manager.py`)
   - Load/save GPS data
   - Load/save models
   - Load/save DFT coefficients
   - Cache weather data
   - Save/load results in CSV format
   - Save metrics as JSON

2. **DataAcquisition** (`src/data/acquisition.py`)
   - Fetch GPS traces from S3
   - Fetch bus stops from S3
   - Fetch weather data from OpenWeatherMap API
   - Interpolate weather data
   - Validate data constraints

3. **DataPreprocessor** (`src/data/preprocessing.py`)
   - Compute speeds using Haversine distance
   - Build 50m x 50m grid
   - Compute grid cell medians
   - Build graph from grid cells
   - Extract MST (Kruskal's algorithm)
   - Compute shortest paths (Dijkstra's algorithm)
   - Build adjacency matrix

4. **Metrics** (`src/evaluation/metrics.py`)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)
   - RMSE (Root Mean Squared Error)
   - Categorize by ETA ranges
   - Compare multiple models

5. **ResultsManager** (`src/evaluation/results_manager.py`)
   - Save model performance (Table 1 format)
   - Save route performance (Table 2, 3 format)
   - Save hybrid comparison
   - Save baseline comparison
   - Save ablation studies
   - Save predictions with timestamps
   - Create summary reports

6. **Haversine** (`src/utils/haversine.py`)
   - Distance calculations between GPS coordinates
   - Vectorized operations for arrays

### 🔨 To Be Implemented

The following modules need implementation based on the algorithms:

1. **Model Implementations** (7 files)
   - MST-AV (Algorithm 2, 3)
   - GDRN-DFT (Algorithm 4, 5)
   - KNN (Algorithm 6, 7)
   - FE-NN (Algorithm 8, 9)
   - MGCN (Algorithm 10, 11)
   - Hybrid (Algorithm 12, 13)
   - Benchmarks (Algorithm 14)

2. **Training Scripts** (7 files)
   - One for each model

3. **Inference Module** (2 files)
   - Real-time ETA predictor
   - Model loader

4. **Evaluation Pipeline** (1 file)
   - Complete evaluation orchestration

## Usage Examples

### 1. Train All Models
```bash
python scripts/run_training_pipeline.py --route route1
```

### 2. Train Specific Models
```bash
python scripts/run_training_pipeline.py --route route1 --models mst_av,gdrn_dft,knn
```

### 3. Skip Preprocessing (Load from S3)
```bash
python scripts/run_training_pipeline.py --route route1 --skip-preprocessing
```

### 4. Evaluate Models (TO IMPLEMENT)
```bash
python scripts/run_evaluation.py --route route1 --model all
```

### 5. Run Inference (TO IMPLEMENT)
```bash
python scripts/run_inference.py --bus-location "22.569,88.516" --stop-location "22.653,88.360"
```

## Data Flow

```
1. S3 (GPS Data) → DataAcquisition → Raw GPS DataFrame
                                    ↓
2. OpenWeatherMap API → DataAcquisition → GPS + Weather DataFrame
                                    ↓
3. DataPreprocessor → Graph, MST, Adjacency Matrix
                                    ↓
4. Training Modules → Trained Models → S3 (Models Bucket)
                                    ↓
5. Evaluation → Metrics → ResultsManager → S3 (Results Bucket)
```

## Environment Variables Required

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_SESSION_TOKEN=your_token  # Optional
AWS_DEFAULT_REGION=us-east-1

# S3 Buckets
S3_TRAIN_GPS_BUCKET=hybrid-eta-train-gps-data
S3_TEST_GPS_BUCKET=hybrid-eta-test-gps-data
S3_MODEL_BUCKET=hybrid-eta-models
S3_DFT_COEFFICIENTS_BUCKET=hybrid-eta-dft-coefficients
S3_WEATHER_CACHE_BUCKET=hybrid-eta-weather-cache
S3_RESULTS_BUCKET=hybrid-eta-results

# OpenWeatherMap API
OPENWEATHERMAP_API_KEY=your_api_key
OPENWEATHERMAP_BASE_URL=https://history.openweathermap.org/data/2.5/history/city
```

## Next Steps

1. Implement the 7 model classes based on the algorithms
2. Implement the 7 training scripts
3. Implement the evaluation pipeline
4. Implement the inference module
5. Add unit tests
6. Add integration tests
7. Deploy to production

## Notes

- All data operations go through S3 (no local storage except temporary)
- Results are automatically formatted to match your CSV structures
- Models are versioned and stored with metadata
- Weather data is cached to reduce API calls
- Graph preprocessing is saved to avoid recomputation
