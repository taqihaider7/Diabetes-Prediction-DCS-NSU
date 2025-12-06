"""
Test script for Diabetes Prediction FastAPI
Tests various endpoints and functionality
"""

import requests
import json
from typing import Dict, Any
import time

# API Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 10

# Test data - Single prediction
SINGLE_PREDICTION_DATA = {
    "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 0.0, "N5": 0.0,
    "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
    "N15": 0.0, "Pregnancies": 0.64, "Glucose": 0.86, "BloodPressure": 0.03,
    "SkinThickness": 0.67, "Insulin": 0.31, "BMI": 0.17,
    "DiabetesPedigreeFunction": 0.47, "Age": 1.43, "N0": 0.44,
    "N8": 0.14, "N13": 0.56, "N12": 1.20, "N14": 0.02
}

BATCH_PREDICTION_DATA = {
    "predictions": [
        {
            "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 0.0, "N5": 0.0,
            "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
            "N15": 0.0, "Pregnancies": 0.64, "Glucose": 0.86, "BloodPressure": 60,
            "SkinThickness": 0.67, "Insulin": 0.31, "BMI": 0.17,
            "DiabetesPedigreeFunction": 0.47, "Age": 1.43, "N0": 0.44,
            "N8": 0.14, "N13": 0.56, "N12": 1.20, "N14": 0.02
        },
        {
            "N1": 0.0, "N2": 1.0, "N3": 0.0, "N4": 1.0, "N5": 0.0,
            "N6": 0.0, "N7": 1.0, "N9": 1.0, "N10": 1.0, "N11": 1.0,
            "N15": 1.0, "Pregnancies": 0.84, "Glucose": 1.20, "BloodPressure": 100,
            "SkinThickness": 0.01, "Insulin": 0.44, "BMI": 0.85,
            "DiabetesPedigreeFunction": 0.37, "Age": 0.19, "N0": 0.46,
            "N8": 0.93, "N13": 0.54, "N12": 0.38, "N14": 0.01
        },
        {
            "N1": 1.0, "N2": 1.0, "N3": 1.0, "N4": 0.0, "N5": 0.0,
            "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
            "N15": 1.0, "Pregnancies": 1.23, "Glucose": 2.01, "BloodPressure": 69,
            "SkinThickness": 0.33, "Insulin": 0.31, "BMI": 1.33,
            "DiabetesPedigreeFunction": 0.60, "Age": 0.11, "N0": 0.51,
            "N8": 1.73, "N13": 0.42, "N12": 0.44, "N14": 0.57
        }
    ]
}


class TestColors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_section(title: str):
    """Print a section header"""
    print(f"\n{TestColors.BOLD}{TestColors.BLUE}{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}{TestColors.END}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{TestColors.GREEN}✓ {message}{TestColors.END}")


def print_error(message: str):
    """Print error message"""
    print(f"{TestColors.RED}✗ {message}{TestColors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"{TestColors.YELLOW}ℹ {message}{TestColors.END}")


def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent))


def test_root_endpoint():
    """Test root endpoint"""
    print_section("Test 1: Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        print_success("Root endpoint returned successfully")
        print_json(data)
        
        # Verify structure
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        print_success("All required fields present")
        
        return True
    except Exception as e:
        print_error(f"Root endpoint test failed: {str(e)}")
        return False


def test_health_check():
    """Test health check endpoint"""
    print_section("Test 2: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        print_success("Health check endpoint returned successfully")
        print_json(data)
        
        # Verify structure
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "unhealthy"]
        
        if data["model_loaded"]:
            print_success("Model is loaded and ready")
        else:
            print_error("Model is not loaded")
            return False
        
        return True
    except Exception as e:
        print_error(f"Health check test failed: {str(e)}")
        return False


def test_features_endpoint():
    """Test features endpoint"""
    print_section("Test 3: Features Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/features", timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        print_success("Features endpoint returned successfully")
        print(f"\nTotal features: {data['count']}")
        print(f"Features: {data['features']}")
        
        assert "features" in data
        assert "count" in data
        assert data["count"] == 24
        print_success(f"Expected 24 features, got {data['count']}")
        
        return True
    except Exception as e:
        print_error(f"Features endpoint test failed: {str(e)}")
        return False


def test_model_info():
    """Test model information endpoint"""
    print_section("Test 4: Model Information")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info", timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        print_success("Model info endpoint returned successfully")
        print_json(data)
        
        assert "model_info" in data
        assert "features" in data
        print_success("Model information retrieved successfully")
        
        return True
    except Exception as e:
        print_error(f"Model info test failed: {str(e)}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    print_section("Test 5: Single Prediction")
    
    try:
        print_info(f"Sending prediction request...")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=SINGLE_PREDICTION_DATA,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        data = response.json()
        print_success("Single prediction returned successfully")
        print_json(data)
        
        # Verify structure
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "timestamp" in data
        assert "model_info" in data
        
        prediction = data["prediction"]
        probability = data["probability"]
        
        if prediction == 0:
            print_success(f"Prediction: No Diabetes (probability: {probability:.4f})")
        else:
            print_success(f"Prediction: Diabetes (probability: {probability:.4f})")
        
        return True
    except Exception as e:
        print_error(f"Single prediction test failed: {str(e)}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_section("Test 6: Batch Prediction")
    
    try:
        print_info(f"Sending batch prediction request with {len(BATCH_PREDICTION_DATA['predictions'])} records...")
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=BATCH_PREDICTION_DATA,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        data = response.json()
        print_success("Batch prediction returned successfully")
        print_json(data)
        
        # Verify structure
        assert "total_predictions" in data
        assert "successful_predictions" in data
        assert "failed_predictions" in data
        assert "predictions" in data
        
        total = data["total_predictions"]
        successful = data["successful_predictions"]
        failed = data["failed_predictions"]
        
        print_success(f"Total: {total}, Successful: {successful}, Failed: {failed}")
        
        if failed == 0:
            print_success("All batch predictions successful")
        else:
            print_error(f"{failed} predictions failed")
        
        return True
    except Exception as e:
        print_error(f"Batch prediction test failed: {str(e)}")
        return False


def test_invalid_input():
    """Test invalid input handling"""
    print_section("Test 7: Invalid Input Handling")
    
    try:
        print_info("Testing invalid input...")
        
        # Missing required field
        invalid_data = SINGLE_PREDICTION_DATA.copy()
        del invalid_data["N1"]
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:
            print_success("Invalid input properly rejected with status 422")
            print_json(response.json())
            return True
        else:
            print_error(f"Expected status 422, got {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Invalid input test failed: {str(e)}")
        return False


def test_performance():
    """Test API performance"""
    print_section("Test 8: Performance Test")
    
    try:
        print_info("Running 10 sequential predictions for performance analysis...")
        
        times = []
        
        for i in range(10):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json=SINGLE_PREDICTION_DATA,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Prediction {i+1}: {elapsed*1000:.2f}ms")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print_success(f"Performance Statistics:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        
        return True
    except Exception as e:
        print_error(f"Performance test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print(f"\n{TestColors.BOLD}{TestColors.BLUE}")
    print("="*80)
    print("  DIABETES PREDICTION API - TEST SUITE")
    print("="*80)
    print(f"{TestColors.END}")
    
    print_info(f"Testing API at: {BASE_URL}")
    print_info(f"Timeout: {TIMEOUT}s\n")
    
    tests = [
        test_root_endpoint,
        test_health_check,
        test_features_endpoint,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
        test_invalid_input,
        test_performance
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_error(f"Test {test_func.__name__} failed with exception: {str(e)}")
            results.append(False)
        
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(results)
    total = len(results)
    failed = total - passed
    
    print(f"Total Tests: {total}")
    print_success(f"Passed: {passed}")
    
    if failed > 0:
        print_error(f"Failed: {failed}")
    
    print()
    
    if failed == 0:
        print(f"{TestColors.GREEN}{TestColors.BOLD}✓ ALL TESTS PASSED!{TestColors.END}")
        return True
    else:
        print(f"{TestColors.RED}{TestColors.BOLD}✗ SOME TESTS FAILED!{TestColors.END}")
        return False


if __name__ == "__main__":
    print("\nStarting API tests...")
    print("Make sure the API is running: uvicorn main:app --reload --log-level info\n")
    
    success = run_all_tests()
    
    exit(0 if success else 1)
