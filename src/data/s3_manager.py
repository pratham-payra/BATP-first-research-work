"""
S3 Manager for Hybrid ETA System
Handles all S3 operations for data retrieval, storage, and model management
"""

import boto3
import os
import json
import pandas as pd
from io import StringIO, BytesIO
from typing import Optional, List, Dict, Any
from pathlib import Path
import pickle
import torch
from dotenv import load_dotenv

load_dotenv()


class S3Manager:
    """Manages all S3 operations for the Hybrid ETA system"""
    
    def __init__(self):
        """Initialize S3 client and bucket names from environment variables"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        # Bucket names from environment
        self.train_gps_bucket = os.getenv('S3_TRAIN_GPS_BUCKET', 'hybrid-eta-train-gps-data')
        self.test_gps_bucket = os.getenv('S3_TEST_GPS_BUCKET', 'hybrid-eta-test-gps-data')
        self.model_bucket = os.getenv('S3_MODEL_BUCKET', 'hybrid-eta-models')
        self.dft_bucket = os.getenv('S3_DFT_COEFFICIENTS_BUCKET', 'hybrid-eta-dft-coefficients')
        self.weather_bucket = os.getenv('S3_WEATHER_CACHE_BUCKET', 'hybrid-eta-weather-cache')
        self.results_bucket = os.getenv('S3_RESULTS_BUCKET', 'hybrid-eta-results')
    
    # ==================== GPS Data Operations ====================
    
    def load_gps_data(self, route: str, data_type: str = 'train') -> pd.DataFrame:
        """
        Load GPS data from S3
        
        Args:
            route: Route identifier (e.g., 'route1', 'route2')
            data_type: 'train' or 'test'
        
        Returns:
            DataFrame with GPS data
        """
        bucket = self.train_gps_bucket if data_type == 'train' else self.test_gps_bucket
        key = f"{route}/gps_data.csv"
        
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            print(f"Loaded GPS data from s3://{bucket}/{key}")
            return df
        except Exception as e:
            print(f"Error loading GPS data: {e}")
            raise
    
    def save_gps_data(self, df: pd.DataFrame, route: str, data_type: str = 'train'):
        """Save GPS data to S3"""
        bucket = self.train_gps_bucket if data_type == 'train' else self.test_gps_bucket
        key = f"{route}/gps_data.csv"
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print(f"Saved GPS data to s3://{bucket}/{key}")
    
    def load_bus_stops(self, route: str) -> pd.DataFrame:
        """Load bus stop data from S3"""
        bucket = self.train_gps_bucket
        key = f"{route}/bus_stops.csv"
        
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            print(f"Loaded bus stops from s3://{bucket}/{key}")
            return df
        except Exception as e:
            print(f"Error loading bus stops: {e}")
            raise
    
    # ==================== Model Operations ====================
    
    def save_model(self, model: Any, model_name: str, route: str, metadata: Optional[Dict] = None):
        """
        Save trained model to S3
        
        Args:
            model: Model object (PyTorch model, sklearn model, or custom)
            model_name: Name of the model (e.g., 'mst_av', 'gdrn_dft')
            route: Route identifier
            metadata: Optional metadata dictionary
        """
        key = f"{route}/{model_name}/model.pth"
        
        # Save PyTorch models
        if hasattr(model, 'state_dict'):
            buffer = BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            self.s3_client.put_object(Bucket=self.model_bucket, Key=key, Body=buffer.getvalue())
        else:
            # Save other models with pickle
            buffer = BytesIO()
            pickle.dump(model, buffer)
            buffer.seek(0)
            self.s3_client.put_object(Bucket=self.model_bucket, Key=key, Body=buffer.getvalue())
        
        # Save metadata
        if metadata:
            metadata_key = f"{route}/{model_name}/metadata.json"
            self.s3_client.put_object(
                Bucket=self.model_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
        
        print(f"Saved model to s3://{self.model_bucket}/{key}")
    
    def load_model(self, model_class: Any, model_name: str, route: str) -> Any:
        """
        Load trained model from S3
        
        Args:
            model_class: Model class to instantiate
            model_name: Name of the model
            route: Route identifier
        
        Returns:
            Loaded model
        """
        key = f"{route}/{model_name}/model.pth"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.model_bucket, Key=key)
            buffer = BytesIO(obj['Body'].read())
            
            # Try loading as PyTorch model
            try:
                model = model_class()
                model.load_state_dict(torch.load(buffer))
                model.eval()
            except:
                # Load as pickle
                buffer.seek(0)
                model = pickle.load(buffer)
            
            print(f"Loaded model from s3://{self.model_bucket}/{key}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_model_metadata(self, model_name: str, route: str) -> Dict:
        """Load model metadata from S3"""
        key = f"{route}/{model_name}/metadata.json"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.model_bucket, Key=key)
            metadata = json.loads(obj['Body'].read())
            return metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
    
    # ==================== DFT Coefficients Operations ====================
    
    def save_dft_coefficients(self, coefficients: Dict, route: str):
        """Save DFT coefficients to S3"""
        key = f"{route}/dft_coefficients.pkl"
        
        buffer = BytesIO()
        pickle.dump(coefficients, buffer)
        buffer.seek(0)
        
        self.s3_client.put_object(
            Bucket=self.dft_bucket,
            Key=key,
            Body=buffer.getvalue()
        )
        print(f"Saved DFT coefficients to s3://{self.dft_bucket}/{key}")
    
    def load_dft_coefficients(self, route: str) -> Dict:
        """Load DFT coefficients from S3"""
        key = f"{route}/dft_coefficients.pkl"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.dft_bucket, Key=key)
            coefficients = pickle.load(BytesIO(obj['Body'].read()))
            print(f"Loaded DFT coefficients from s3://{self.dft_bucket}/{key}")
            return coefficients
        except Exception as e:
            print(f"Error loading DFT coefficients: {e}")
            raise
    
    # ==================== Weather Cache Operations ====================
    
    def cache_weather_data(self, weather_data: pd.DataFrame, route: str, date: str):
        """Cache weather data to S3"""
        key = f"{route}/{date}/weather.csv"
        
        csv_buffer = StringIO()
        weather_data.to_csv(csv_buffer, index=False)
        
        self.s3_client.put_object(
            Bucket=self.weather_bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print(f"Cached weather data to s3://{self.weather_bucket}/{key}")
    
    def load_cached_weather(self, route: str, date: str) -> Optional[pd.DataFrame]:
        """Load cached weather data from S3"""
        key = f"{route}/{date}/weather.csv"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.weather_bucket, Key=key)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            print(f"Loaded cached weather from s3://{self.weather_bucket}/{key}")
            return df
        except:
            return None
    
    # ==================== Results Operations ====================
    
    def save_results(self, results_df: pd.DataFrame, result_type: str, filename: str):
        """
        Save evaluation results to S3
        
        Args:
            results_df: DataFrame with results
            result_type: Type of results ('model_performance', 'hybrid_comparison', 
                        'baseline_comparison', 'ablation_studies', 'predictions')
            filename: Name of the file (e.g., 'route1_performance.csv')
        """
        key = f"{result_type}/{filename}"
        
        csv_buffer = StringIO()
        results_df.to_csv(csv_buffer, index=False)
        
        self.s3_client.put_object(
            Bucket=self.results_bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print(f"Saved results to s3://{self.results_bucket}/{key}")
    
    def load_results(self, result_type: str, filename: str) -> pd.DataFrame:
        """Load results from S3"""
        key = f"{result_type}/{filename}"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.results_bucket, Key=key)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            print(f"Loaded results from s3://{self.results_bucket}/{key}")
            return df
        except Exception as e:
            print(f"Error loading results: {e}")
            raise
    
    def save_metrics_json(self, metrics: Dict, result_type: str, filename: str):
        """Save metrics as JSON to S3"""
        key = f"{result_type}/{filename}"
        
        self.s3_client.put_object(
            Bucket=self.results_bucket,
            Key=key,
            Body=json.dumps(metrics, indent=2)
        )
        print(f"Saved metrics to s3://{self.results_bucket}/{key}")
    
    def list_files(self, bucket_name: str, prefix: str = '') -> List[str]:
        """List all files in a bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def save_preprocessed_data(self, data: Dict, route: str, data_name: str):
        """Save preprocessed data (graphs, adjacency matrices, etc.) to S3"""
        key = f"{route}/preprocessed/{data_name}.pkl"
        
        buffer = BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        
        self.s3_client.put_object(
            Bucket=self.train_gps_bucket,
            Key=key,
            Body=buffer.getvalue()
        )
        print(f"Saved preprocessed data to s3://{self.train_gps_bucket}/{key}")
    
    def load_preprocessed_data(self, route: str, data_name: str) -> Dict:
        """Load preprocessed data from S3"""
        key = f"{route}/preprocessed/{data_name}.pkl"
        
        try:
            obj = self.s3_client.get_object(Bucket=self.train_gps_bucket, Key=key)
            data = pickle.load(BytesIO(obj['Body'].read()))
            print(f"Loaded preprocessed data from s3://{self.train_gps_bucket}/{key}")
            return data
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            raise


# Singleton instance
_s3_manager = None

def get_s3_manager() -> S3Manager:
    """Get singleton S3Manager instance"""
    global _s3_manager
    if _s3_manager is None:
        _s3_manager = S3Manager()
    return _s3_manager
