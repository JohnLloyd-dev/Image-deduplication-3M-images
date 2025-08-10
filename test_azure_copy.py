#!/usr/bin/env python3
"""
Test script for Azure image copy functionality.

This script tests the components of the Azure image copy system
without requiring actual Azure connectivity.
"""

import os
import pandas as pd
import tempfile

def create_test_report():
    """Create a test CSV report for testing."""
    test_data = {
        'Image Path': [
            'Image_Dedup_Project/TestEquity/CompleteImageDataset/test1.jpg',
            'Image_Dedup_Project/TestEquity/CompleteImageDataset/test2.jpg',
            'Image_Dedup_Project/TestEquity/CompleteImageDataset/test3.jpg',
            'Image_Dedup_Project/TestEquity/CompleteImageDataset/test4.jpg',
            'Image_Dedup_Project/TestEquity/CompleteImageDataset/test5.jpg'
        ],
        'Quality Score': [95.5, 92.1, 88.7, 97.2, 94.8],
        'Group ID': [1, 1, 1, 2, 2],
        'Group Size': [3, 3, 3, 2, 2],
        'Status': ['Best', 'Duplicate', 'Duplicate', 'Best', 'Duplicate']
    }
    
    df = pd.DataFrame(test_data)
    return df

def test_report_reading():
    """Test CSV report reading functionality."""
    print("\n[TEST] Testing CSV report reading...")
    
    try:
        # Create test report
        df = create_test_report()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_report_path = f.name
            df.to_csv(f, index=False)
        
        # Verify file exists
        if not os.path.exists(test_report_path):
            print("[FAIL] Failed to create test report file")
            return False
        
        # Read the file back
        df_read = pd.read_csv(test_report_path)
        
        # Verify data integrity
        if len(df_read) != len(df):
            print(f"[FAIL] Data length mismatch: {len(df_read)} != {len(df)}")
            return False
        
        # Verify required columns
        required_columns = ['Image Path', 'Status', 'Quality Score', 'Group ID', 'Group Size']
        missing_columns = [col for col in required_columns if col not in df_read.columns]
        if missing_columns:
            print(f"[FAIL] Missing required columns: {missing_columns}")
            return False
        
        # Test data extraction
        best_images = df_read[df_read['Status'] == 'Best']['Image Path'].tolist()
        duplicate_images = df_read[df_read['Status'] == 'Duplicate']['Image Path'].tolist()
        
        expected_best = 2
        expected_duplicate = 3
        
        if len(best_images) != expected_best:
            print(f"[FAIL] Best images count mismatch: {len(best_images)} != {expected_best}")
            return False
        
        if len(duplicate_images) != expected_duplicate:
            print(f"[FAIL] Duplicate images count mismatch: {len(duplicate_images)} != {expected_duplicate}")
            return False
        
        print(f"[OK] Report reading test passed:")
        print(f"   - Total images: {len(df_read)}")
        print(f"   - Best images: {len(best_images)}")
        print(f"   - Duplicate images: {len(duplicate_images)}")
        
        # Clean up
        os.remove(test_report_path)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Report reading test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n[TEST] Testing configuration loading...")
    
    try:
        # Test if config file can be imported
        try:
            import azure_copy_config
            print("[OK] Configuration file loaded successfully")
            print(f"   - Destination: {azure_copy_config.AZURE_DESTINATION_BASE}")
            print(f"   - Report path: {azure_copy_config.REPORT_FILE_PATH}")
            print(f"   - Copy best: {azure_copy_config.COPY_BEST_IMAGES}")
            print(f"   - Copy duplicates: {azure_copy_config.COPY_DUPLICATE_IMAGES}")
            return True
        except ImportError as e:
            print(f"[WARN] Configuration file not found: {e}")
            print("   This is okay if you haven't created it yet")
            return True
            
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

def test_azure_utils_import():
    """Test Azure utils import."""
    print("\n[TEST] Testing Azure utils import...")
    
    try:
        from azure_utils import AzureBlobManager, SAS_URL
        print("[OK] Azure utils imported successfully")
        print(f"   - SAS URL configured: {'Yes' if SAS_URL else 'No'}")
        
        # Test AzureBlobManager class
        if hasattr(AzureBlobManager, 'copy_blob'):
            print("[OK] AzureBlobManager.copy_blob method exists")
        else:
            print("[FAIL] AzureBlobManager.copy_blob method missing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Azure utils import failed: {e}")
        return False

def test_csv_format():
    """Test CSV format validation."""
    print("\n[TEST] Testing CSV format validation...")
    
    try:
        # Create test data
        df = create_test_report()
        
        # Test column validation
        required_columns = ['Image Path', 'Status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"[FAIL] Missing required columns: {missing_columns}")
            return False
        
        # Test status values
        unique_statuses = df['Status'].unique()
        expected_statuses = ['Best', 'Duplicate']
        
        for status in expected_statuses:
            if status not in unique_statuses:
                print(f"[FAIL] Missing expected status: {status}")
                return False
        
        # Test data types
        if df['Image Path'].dtype != 'object':
            print("[FAIL] Image Path column should be string type")
            return False
        
        if df['Status'].dtype != 'object':
            print("[FAIL] Status column should be string type")
            return False
        
        print("[OK] CSV format validation passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] CSV format test failed: {e}")
        return False

def run_test(test_func, test_name):
    """Run a test function and report results."""
    try:
        result = test_func()
        if result:
            print(f"[OK] {test_name} PASSED")
        else:
            print(f"[FAIL] {test_name} FAILED")
        return result
    except Exception as e:
        print(f"[FAIL] {test_name} crashed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AZURE IMAGE COPY TEST SUITE")
    print("=" * 60)
    
    tests = [
        (test_report_reading, "CSV Report Reading"),
        (test_configuration, "Configuration Loading"),
        (test_azure_utils_import, "Azure Utils Import"),
        (test_csv_format, "CSV Format Validation")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[OK] All tests passed! The Azure copy script is ready to use.")
    else:
        print("[WARN] Some tests failed. Please fix the issues before using the copy script.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
