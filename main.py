from preprocess import PreProcessor
from text_processer import TextProcesser
from pathlib import Path
import pandas as pd
import os
import json

BASE_PATH_DICT = {
    '2024': Path(r"C:\Users\User\Desktop\10-X_C_2024\2024"),
    '2025': Path(r"C:\Users\User\Desktop\10-X_C_2025\2025"),
}
OVERRIDE_PROCESSED_DATA_FILE = True  # Set to True to override existing processed data files, False to skip processing if the file exists
FILTERED_DF_OUTPUT = "filtered_file_details_{year}.csv"
PROCESSED_DATA_OUTPUT = "processed_data_{year}.json"

def get_one_year_data(year:str) -> dict[str, list[str]]:
    print(f"Getting year data for {year}")
    filtered_df_output: str = FILTERED_DF_OUTPUT.format(year=year)
    processed_data_output: str = PROCESSED_DATA_OUTPUT.format(year=year)
    base_path: Path = BASE_PATH_DICT[year]

    preprocessor = PreProcessor(base_path=base_path, year=year, output_file=filtered_df_output)
    filtered_files_df = preprocessor.run()
    if filtered_files_df is None:
        try:
            filtered_files_df = pd.read_csv(filtered_df_output)
        except FileNotFoundError:
            print(f"Filtered file details CSV '{filtered_df_output}' not found. Please run the preprocessing step first.")
            return
    
    if os.path.exists(processed_data_output) and not OVERRIDE_PROCESSED_DATA_FILE:
        print(f"Processed data file '{processed_data_output}' already exists. Skipping processing.")
        with open(processed_data_output, "r") as f:
            processed_data = json.load(f)
    else:
        text_processor = TextProcesser(filtered_files_df=filtered_files_df, base_path=base_path, output_file=processed_data_output)
        processed_data = text_processor.process_all_files()

    print(f"Processed data for year {year} has been saved to '{processed_data_output}'")
    return processed_data

def main():
    data_2024: dict[str, list[str]] = get_one_year_data(
        year='2024'
    )
    #data_2025: dict[str, list[str]] = get_one_year_data(
    #    year='2025'
    #)
    print(len(data_2024))



if __name__ == "__main__":
    main()
