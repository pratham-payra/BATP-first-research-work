"""
Inference Script
Real-time ETA prediction using trained models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
from src.inference.eta_predictor import ETAPredictor


def main():
    parser = argparse.ArgumentParser(description='Run Real-time ETA Prediction')
    parser.add_argument('--route', type=str, default='route1',
                       help='Route identifier (default: route1)')
    parser.add_argument('--model', type=str, default='hybrid',
                       help='Model to use (mst_av, gdrn_dft, knn, fenn, mgcn, hybrid, dcrnn, stgcn, gwnet, tgcn, mtgnn, stfgnn, st_resnet, st_gconv)')
    parser.add_argument('--bus-location', type=str, required=True,
                       help='Current bus location as "lat,lon"')
    parser.add_argument('--stop-location', type=str, required=True,
                       help='Target stop location as "lat,lon"')
    parser.add_argument('--speed', type=float, default=20.0,
                       help='Current bus speed in km/h (default: 20.0)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID ETA REAL-TIME PREDICTION")
    print("=" * 80)
    
    # Parse locations
    try:
        bus_lat, bus_lon = map(float, args.bus_location.split(','))
        stop_lat, stop_lon = map(float, args.stop_location.split(','))
    except:
        print("Error: Invalid location format. Use 'lat,lon' format.")
        return
    
    bus_location = (bus_lat, bus_lon)
    stop_location = (stop_lat, stop_lon)
    
    print(f"\nRoute: {args.route}")
    print(f"Model: {args.model}")
    print(f"Bus Location: {bus_location}")
    print(f"Stop Location: {stop_location}")
    print(f"Current Speed: {args.speed} km/h")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Initialize predictor
    try:
        predictor = ETAPredictor(args.route, args.model)
    except Exception as e:
        print(f"\nError initializing predictor: {e}")
        print("\nMake sure the model has been trained and saved to S3.")
        return
    
    # Make prediction
    try:
        result = predictor.predict_eta(
            bus_location=bus_location,
            stop_location=stop_location,
            current_speed=args.speed
        )
        
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print(f"Estimated Time of Arrival: {result['eta_minutes']:.2f} minutes")
        print(f"Distance to Stop: {result['distance_km']:.2f} km")
        print(f"Average Speed Required: {(result['distance_km'] / (result['eta_minutes'] / 60.0)):.2f} km/h")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
