import pandas as pd
from pathlib import Path

def extract_sample_data(input_path, output_path, n_samples=50):
    """Extract first n consecutive rows from dataset for inspection"""
    df = pd.read_csv(input_path)
    sample = df.head(n_samples)
    sample.to_csv(output_path, index=False)
    print(f"Saved first {n_samples} consecutive samples to {output_path}")

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    input_file = data_dir / "wtbdata_hourly.csv"
    output_file = data_dir / "first_50_samples.csv"
    
    extract_sample_data(input_file, output_file)