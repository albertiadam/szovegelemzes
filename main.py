"""
Main file to run the thesis LDA analysis.
It processes the data for the years 2024 and 2025, runs LDA, and generates a comparison of dominant topics between the two years.

Runs post-processing to generate a contingency table and statistical summary of the topic changes.
The script allows for user input to customize the processing, including whether to override existing processed data files,
    whether to run LDA separately for each year, whether to use bigrams, and the number of topics for LDA.

Generated outputs:
    - Filtered file details for each year: filtered_file_details_{year}.csv
    - Processed data for each year (containing company name as key and list of words as value): processed_data_{year}.json
    - LDA comparison results (company_name, prev_year_dominant_topic, curr_year_dominant_topic) : LDA_comparison_results.csv
    - Contingency table: contingency_table.csv
    - Statistical summary of topic changes (printed to console)
    - Skipped companies in step (TextProcessor) for each year: skipped_file_details_{year}.csv
    - Grouped skipped companies by industry and reason: grouped_skipped_companies.csv

Hardcoded values across codebase:
    - Base paths for the 2024 and 2025 data folders
    - Output file names for filtered file details and processed data
    - Output file name for LDA comparison results
    - Output file name for contingency table
    - LDA Model parameters (passes=15, iterations=50)

Additional parameters might need override which can be found at the start of main.py:
    - BASE_PATH_DICT: folders containing the 10-X data
    - FILTERED_DF_OUTPUT: output file name for filtered file details
    - PROCESSED_DATA_OUTPUT: output file name for processed data
    - LDA_MODEL_PASSES: number of passes for LDA model training
    - LDA_MODEL_ITERATIONS: number of iterations for LDA model training
"""
from preprocess import PreProcessor
from text_processor import TextProcessor
from lda_processor import LDAProcessor
from post_processor import PostProcessor
from pathlib import Path
import pandas as pd
import os
import json

BASE_PATH_DICT = {
    '2024': Path(r"C:\Users\User\Desktop\10-X_C_2024\2024"),
    '2025': Path(r"C:\Users\User\Desktop\10-X_C_2025\2025"),
}
FILTERED_DF_OUTPUT = "process_outputs/filtered_file_details_{year}.csv"
PROCESSED_DATA_OUTPUT = "process_outputs/processed_data_{year}.json"
SKIPPED_FILE_OUTPUT = "process_outputs/skipped_file_details_{year}.csv"
LDA_MODEL_PASSES = 15
LDA_MODEL_ITERATIONS = 50

def get_inputs():
    global OVERRIDE_PROCESSED_DATA_FILE, RUN_LDA_SEPARATELY, USE_BIGRAMS, NUM_TOPICS

    # Set to True to override existing processed data files, False to skip processing if the file exists
    OVERRIDE_PROCESSED_DATA_FILE = input("Override existing processed data files? (y/n): ").strip().lower() == 'y'
    RUN_LDA_SEPARATELY = input("Run LDA separately for each year? (y/n): ").strip().lower() == 'y'
    USE_BIGRAMS = input("Use bigrams? (y/n): ").strip().lower() == 'y'
    NUM_TOPICS = int(input("Enter the number of topics for LDA: ").strip())


def get_one_year_data(year:str) -> dict[str, list[str]]:
    print(f"Getting year data for {year}")
    filtered_df_output: str = FILTERED_DF_OUTPUT.format(year=year)
    processed_data_output: str = PROCESSED_DATA_OUTPUT.format(year=year)
    base_path: Path = BASE_PATH_DICT[year]

    preprocessor = PreProcessor(base_path=base_path, year=year, output_file=filtered_df_output)

    if not OVERRIDE_PROCESSED_DATA_FILE and os.path.exists(processed_data_output):
        print(f"Processed data file '{processed_data_output}' already exists. Skipping processing.")
        with open(processed_data_output, "r") as f:
            processed_data = json.load(f)
    else:
        print(f"Running preprocessing for {year}")
        filtered_files_df = preprocessor.run()
        print(f"Running text processing for {year}")
        text_processor = TextProcessor(
            filtered_files_df=filtered_files_df,
            base_path=base_path,
            output_file=processed_data_output,
            skipped_output_file=SKIPPED_FILE_OUTPUT.format(year=year),
        )
        processed_data = text_processor.process_all_files()
        print(f"Processed data for year {year} has been saved to '{processed_data_output}'")

    return processed_data

def main() -> None:
    get_inputs()
    prev_data: dict[str, list[str]] = get_one_year_data(
        year='2024'
    )
    curr_data: dict[str, list[str]] = get_one_year_data(
        year='2025'
    )

    print(f"Previous year data: {len(prev_data)} companies")
    print(f"Current year data: {len(curr_data)} companies")

    LDA_processor = LDAProcessor(
        curr_data=curr_data,
        prev_data=prev_data,
        num_topics=NUM_TOPICS,
        run_separately=RUN_LDA_SEPARATELY,
        use_bigrams=USE_BIGRAMS,
        num_of_passes=LDA_MODEL_PASSES,
        num_of_iterations=LDA_MODEL_ITERATIONS
    )
    comparison_df = LDA_processor.create_comparison()

    comparison_df.to_csv("process_outputs/LDA_comparison_results.csv", index=False)

    post_processor = PostProcessor(
        lda_result_string_path="process_outputs/LDA_comparison_results.csv",
        years=['2024', '2025'],
        skipped_data_path=SKIPPED_FILE_OUTPUT
    )
    post_processor.get_summary()

if __name__ == "__main__":
    main()
