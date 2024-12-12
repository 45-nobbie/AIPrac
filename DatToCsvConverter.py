import pandas as pd
import os

def convert_dat_to_csv(dat_file_path, csv_file_path):
    try:
        # Open the .dat file and inspect lines for preprocessing
        with open(dat_file_path, 'r', encoding='latin1') as file:
            lines = file.readlines()

        # Attempt to split the file into columns based on the most common delimiter
        delimiters = [',', '\t', ';', '|', ' ']
        for delimiter in delimiters:
            try:
                data = pd.DataFrame([line.strip().split(delimiter) for line in lines])
                if data.shape[1] > 1:  # Ensure there is more than one column
                    break
            except Exception:
                continue

        # Save the processed data to a CSV file
        data.to_csv(csv_file_path, index=False, header=False)
        print(f"Conversion successful! CSV file saved at: {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # Specify the .dat file path and desired .csv file path
    dat_file = "anamoly.dat"  # Replace with your .dat file path
    csv_file = "example.csv"  # Replace with desired .csv file path

    # Ensure the .dat file exists
    if os.path.exists(dat_file):
        convert_dat_to_csv(dat_file, csv_file)
    else:
        print(f"File not found: {dat_file}")