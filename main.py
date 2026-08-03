from preprocess import PreProcessor
from text_processer import TextProcesser
from pathlib import Path
import pandas as pd
import os
import re
from gensim.parsing.preprocessing import STOPWORDS
import json


BASE_PATH = Path(r"C:\Users\User\Desktop\10-X_C_2024\2024")
FILTERED_DF_OUTPUT = "filtered_file_details.csv"
PROCESSED_DATA_OUTPUT = "processed_data.json"
OVERRIDE_PROCESSED_DATA_FILE = True

def main():
    preprocessor = PreProcessor(base_path=BASE_PATH, output_file=FILTERED_DF_OUTPUT)
    filtered_files_df = preprocessor.run()
    if filtered_files_df is None:
        try:
            filtered_files_df = pd.read_csv(FILTERED_DF_OUTPUT)
        except FileNotFoundError:
            print(f"Filtered file details CSV '{FILTERED_DF_OUTPUT}' not found. Please run the preprocessing step first.")
            return
    
    if os.path.exists(PROCESSED_DATA_OUTPUT) and not OVERRIDE_PROCESSED_DATA_FILE:
        print(f"Processed data file '{PROCESSED_DATA_OUTPUT}' already exists. Skipping processing.")
        with open(PROCESSED_DATA_OUTPUT, "r") as f:
            processed_data = json.load(f)
    else:
        text_processor = TextProcesser(filtered_files_df=filtered_files_df, base_path=BASE_PATH, output_file=PROCESSED_DATA_OUTPUT)
        processed_data = text_processor.process_all_files()
    
    print(len(processed_data))

if __name__ == "__main__":
    main()
