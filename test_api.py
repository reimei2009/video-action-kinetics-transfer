"""
Test script for Video Action Recognition API
"""

import requests
import json
from pathlib import Path
import time

# API endpoint
API_URL = "http://127.0.0.1:8000/api/v1/predict"
HEALTH_URL = "http://127.0.0.1:8000/api/v1/health"

# Test video
VIDEO_PATH = "4088191-hd_1920_1080_25fps.mp4"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("🏥 Testing Health Endpoint...")
    print("="*60)
    
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        data = response.json()
        
        print(f"✅ Status: {data['status']}")
        print(f"✅ Model: {data['model_name']}")
        print(f"✅ Device: {data['device']}")
        print(f"✅ Uptime: {data.get('uptime_seconds', 0):.1f}s")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_predict():
    """Test predict endpoint"""
    print("\n" + "="*60)
    print("🎬 Testing Predict Endpoint...")
    print("="*60)
    
    # Check video exists
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"❌ Video not found: {VIDEO_PATH}")
        return False
    
    print(f"📹 Video: {video_path.name}")
    print(f"📦 Size: {video_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Prepare request
    files = {
        'file': (video_path.name, open(video_path, 'rb'), 'video/mp4')
    }
    params = {'top_k': 5}
    
    print("\n⏳ Uploading and processing...")
    start_time = time.time()
    
    try:
        response = requests.post(
            API_URL, 
            files=files, 
            params=params, 
            timeout=60
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ SUCCESS! (Request time: {request_time:.2f}s)")
            print("\n" + "="*60)
            print("🎯 TOP PREDICTIONS:")
            print("="*60)
            
            for pred in data['predictions']:
                rank = pred['rank']
                label = pred['label']
                confidence = pred['confidence'] * 100
                
                # Progress bar
                bar_length = int(confidence / 2)  # 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                print(f"#{rank}. {label:<20} {confidence:>6.2f}% │{bar}│")
            
            print("\n" + "="*60)
            print("📊 METADATA:")
            print("="*60)
            print(f"Model: {data['model_name']}")
            print(f"Processing Time: {data['processing_time']:.3f}s")
            
            if 'video_metadata' in data:
                meta = data['video_metadata']
                print(f"Video Duration: {meta.get('duration', 'N/A')}s")
                print(f"Video FPS: {meta.get('fps', 'N/A')}")
                print(f"Video Size: {meta.get('size_mb', 'N/A')} MB")
            
            print("="*60)
            return True
        else:
            print(f"\n❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timeout (>{60}s)")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("🧪 Video Action Recognition API - Test Suite")
    print("="*60)
    
    # Test 1: Health check
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Is the server running?")
        print("   Run: python run_api.py")
        return
    
    # Test 2: Prediction
    predict_ok = test_predict()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"Health Check: {'✅ PASSED' if health_ok else '❌ FAILED'}")
    print(f"Prediction:   {'✅ PASSED' if predict_ok else '❌ FAILED'}")
    print("="*60)
    
    if health_ok and predict_ok:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ SOME TESTS FAILED")


if __name__ == "__main__":
    main()
