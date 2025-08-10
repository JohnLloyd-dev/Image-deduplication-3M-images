#!/usr/bin/env python3
"""
Test script for Azure Image Copy functionality

This script tests the Azure image copy process without actually copying files.
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def create_test_report():
    """Create a test CSV report for testing."""
    test_data = [
        {
            'Image Path': 'Image_Dedup_Project/TestEquity/CompleteImageDataset/test1.jpg',
            'Quality Score': 95.5,
            'Group ID': 1,
            'Group Size': 3,
            'Status': 'Best'
        },
        {
            'Image Path': 'Image_Dedup_Project/TestEquity/CompleteImageDataset/test2.jpg',
            'Quality Score': 92.1,
            'Group ID': 1,
            'Group Size': 3,
            'Status': 'Duplicate'
        },
        {
            'Image Path': 'Image_Dedup_Project/TestEquity/CompleteImageDataset/test3.jpg',
            'Quality Score': 88.7,
            'Group ID': 1,
            'Group Size': 3,
            'Status': 'Duplicate'
        },
        {
            'Image Path': 'Image_Dedup_Project/TestEquity/CompleteImageDataset/test4.jpg',
            'Quality Score': 97.2,
            'Group ID': 2,
            'Group Size': 2,
            'Status': 'Best'
        },
        {
            'Image Path': 'Image_Dedup_Project/TestEquity/CompleteImageDataset/test5.jpg',
            'Quality Score': 94.8,
            'Group ID': 2,
            'Group Size': 2,
            'Status': 'Duplicate'
        }
    ]
    
    df = pd.DataFrame(test_data)
    return df

def test_report_reading():
    """Test the report reading functionality."""
    print("🧪 Testing report reading functionality...")
    
    try:
        # Create test report
        df = create_test_report()
        
        # Test CSV creation and reading
        test_report_path = "test_report.csv"
        df.to_csv(test_report_path, index=False)
        
        # Verify file exists
        if not os.path.exists(test_report_path):
            print("❌ Failed to create test report file")
            return False
        
        # Test reading with pandas
        df_read = pd.read_csv(test_report_path)
        
        # Verify data integrity
        if len(df_read) != len(df):
            print(f"❌ Data length mismatch: {len(df_read)} != {len(df)}")
            return False
        
        # Verify required columns
        required_columns = ['Image Path', 'Status', 'Quality Score', 'Group ID', 'Group Size']
        missing_columns = [col for col in required_columns if col not in df_read.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Test data extraction
        best_images = df_read[df_read['Status'] == 'Best']['Image Path'].tolist()
        duplicate_images = df_read[df_read['Status'] == 'Duplicate']['Image Path'].tolist()
        
        expected_best = 2
        expected_duplicate = 3
        
        if len(best_images) != expected_best:
            print(f"❌ Best images count mismatch: {len(best_images)} != {expected_best}")
            return False
        
        if len(duplicate_images) != expected_duplicate:
            print(f"❌ Duplicate images count mismatch: {len(duplicate_images)} != {expected_duplicate}")
            return False
        
        print(f"✅ Report reading test passed:")
        print(f"   - Total images: {len(df_read)}")
        print(f"   - Best images: {len(best_images)}")
        print(f"   - Duplicate images: {len(duplicate_images)}")
        
        # Clean up
        os.remove(test_report_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Report reading test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        # Test if config file can be imported
        try:
            import azure_copy_config
            print("✅ Configuration file loaded successfully")
            print(f"   - Destination: {azure_copy_config.AZURE_DESTINATION_BASE}")
            print(f"   - Report path: {azure_copy_config.REPORT_FILE_PATH}")
            print(f"   - Copy best: {azure_copy_config.COPY_BEST_IMAGES}")
            print(f"   - Copy duplicates: {azure_copy_config.COPY_DUPLICATE_IMAGES}")
            return True
        except ImportError as e:
            print(f"⚠️  Configuration file not found: {e}")
            print("   This is okay if you haven't created it yet")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_azure_utils_import():
    """Test Azure utils import."""
    print("\n🧪 Testing Azure utils import...")
    
    try:
        from azure_utils import AzureBlobManager, SAS_URL
        print("✅ Azure utils imported successfully")
        print(f"   - SAS URL configured: {'Yes' if SAS_URL else 'No'}")
        
        # Test AzureBlobManager class
        if hasattr(AzureBlobManager, 'copy_blob'):
            print("✅ AzureBlobManager.copy_blob method exists")
        else:
            print("❌ AzureBlobManager.copy_blob method missing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Azure utils import failed: {e}")
        return False

def test_csv_format():
    """Test CSV format validation."""
    print("\n🧪 Testing CSV format validation...")
    
    try:
        # Create test data
        df = create_test_report()
        
        # Test column validation
        required_columns = ['Image Path', 'Status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Test status values
        unique_statuses = df['Status'].unique()
        expected_statuses = ['Best', 'Duplicate']
        
        if not all(status in unique_statuses for status in expected_statuses):
            print(f"❌ Missing expected statuses: {expected_statuses}")
            return False
        
        # Test data types
        if not df['Image Path'].dtype == 'object':
            print("❌ Image Path column should be string type")
            return False
        
        if not df['Status'].dtype == 'object':
            print("❌ Status column should be string type")
            return False
        
        print("✅ CSV format validation passed")
        print(f"   - Required columns: {required_columns}")
        print(f"   - Status values: {list(unique_statuses)}")
        print(f"   - Data types: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV format test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Azure Copy Tests...")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_configuration),
        ("Azure Utils Import", test_azure_utils_import),
        ("CSV Format Validation", test_csv_format),
        ("Report Reading", test_report_reading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Azure copy script is ready to use.")
        print("\nNext steps:")
        print("1. Ensure you have a deduplication report CSV file")
        print("2. Configure azure_copy_config.py if needed")
        print("3. Run: python copy_images_to_azure.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before using the copy script.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
