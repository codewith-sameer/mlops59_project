import pandas as pd


def process_data(input_path, output_path):
    # Example data processing
    df = pd.read_csv(input_path)
    # Perform processing (e.g., cleaning, feature engineering)
    df.to_csv(output_path, index=False)


# Process training data
process_data("data/train.csv", "data/processed_train.csv")

# Process test data
process_data("data/test.csv", "data/processed_test.csv")
