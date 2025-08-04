
import sys

# Configure logging at the very top, before any other imports


from pipeline import run_pipeline

def main():
    results = run_pipeline()
    print(f"Pipeline complete. Processed {len(results)} images.")
if __name__ == '__main__':
    main() 