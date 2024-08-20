import pandas as pd
import argparse
import os

def convert_excel_to_csv(excel_path):
    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Construct CSV file name by appending '_csv' before the file extension
    base_name, ext = os.path.splitext(excel_path)
    csv_file = f"{base_name}_csv.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file

def main():
    parser = argparse.ArgumentParser(description="Convert Excel file to CSV file")
    parser.add_argument("excel_path", type=str, help="Path to the Excel file")
    args = parser.parse_args()
    # Convert the Excel to CSV
    output_file = convert_excel_to_csv(args.excel_path)
    print(f"Converted Excel file to CSV file at: {output_file}")

if __name__ == "__main__":
    main()
