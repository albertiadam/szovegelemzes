"""
Main file to run the thesis LDA analysis.
It processes the 10-K filings for 2024 and 2025, extracts the "Item 1C - Cybersecurity" section,
runs LDA on it, and compares each company's dominant topic between the two years.

Pipeline:
    1. PreProcessor (preprocess.py): selects the 10-K filings for each year
        - keeps files with "K" in the file name
        - keeps filings whose period of report is on or after Dec 15 of the previous year
          (the date from which Item 1C is required)
        - drops amendments (10-K/A, 10-KT/A)
        - drops asset-backed entities, trusts and funds by company name keyword
        - keeps one filing per company (the latest period of report, then submit date, then filing sequence)
    2. TextProcessor (text_processor.py): extracts the Item 1C section from each filing,
        cleans it, removes stopwords and lemmatizes it with spaCy (en_core_web_sm).
        Filings where the section cannot be used are skipped with a reason:
        missing_item_1c_cybersecurity, missing_section_end, item_1c_too_short,
        item_1c_too_long, item_1c_wrong_start, empty_after_processing
    3. LDAProcessor (lda_processor.py): trains LDA on both years combined, or one model per year,
        optionally on bigrams, and assigns a dominant topic to every company present in both years
    4. PostProcessor (post_processor.py): builds the topic transition (contingency) table,
        runs a chi-square test with Cramer's V, and summarizes the skipped companies

User inputs (asked at start):
    - Override existing processed data files? If no and processed_data_{year}.json exists,
      steps 1-2 are skipped for that year and the saved file is loaded instead
    - Run LDA separately for each year?
    - Use bigrams?
    - Number of topics for LDA

Generated outputs (all in process_outputs/, which must exist before running):
    - Filtered file details for each year: filtered_file_details_{year}.csv
    - Companies excluded by ABS / trust / fund keyword for each year: abs_companies_{year}.csv
    - Processed data for each year (company name -> list of words): processed_data_{year}.json
    - Skipped companies in step (TextProcessor) for each year, with reason: skipped_file_details_{year}.csv
    - LDA comparison results (company, previous_dominant_topic, current_dominant_topic): LDA_comparison_results.csv
    - Interactive pyLDAvis topic visualizations: prev_lda_visualization.html, curr_lda_visualization.html
    - Contingency table of topic transitions (with totals): contingency_table.csv
    - Skipped companies grouped by industry and reason: grouped_skipped_companies.csv
    - Skipped company counts by reason and by industry: reason_counts.csv, industry_counts.csv
    - Chart of skipped companies by reason and top 10 industries: skipped_companies_counts.png
    - Printed to console: number of unchanged companies, chi-square statistic, p-value,
      number of observations and Cramer's V

Settings at the start of main.py:
    - BASE_PATH_DICT: folders containing the 10-X data for each year
    - FILTERED_DF_OUTPUT: output file name for filtered file details
    - PROCESSED_DATA_OUTPUT: output file name for processed data
    - SKIPPED_FILE_OUTPUT: output file name for skipped file details
    - LDA_MODEL_PASSES: number of passes for LDA model training (15)
    - LDA_MODEL_ITERATIONS: number of iterations for LDA model training (50)

Hardcoded values across codebase:
    - Years compared: 2024 and 2025 (main.py)
    - Output paths for the LDA comparison, visualizations, contingency table and
      skipped-company summaries (main.py, lda_processor.py, post_processor.py)
    - ABS / trust / fund name keywords (preprocess.py)
    - Item 1C length limits: at least 20 and at most 5,000 words (text_processor.py)
    - Dictionary filtering: words in fewer than 20 documents or more than 50% of documents
      are dropped (lda_processor.py)
    - LDA random_state=42 (lda_processor.py)

utils.py is not used by this script. It holds helpers for choosing the number of topics
(elbow and coherence graphs) and for topic stability statistics.
"""
import json
import os
import pandas as pd
from lda_processor import LDAProcessor
from pathlib import Path
from post_processor import PostProcessor
from preprocess import PreProcessor
from text_processor import TextProcessor

BASE_PATH_DICT:dict[str, Path] = {
    '2024': Path(r"C:\Users\User\Desktop\10-X_C_2024\2024"),
    '2025': Path(r"C:\Users\User\Desktop\10-X_C_2025\2025"),
}
FILTERED_DF_OUTPUT:str = "process_outputs/filtered_file_details_{year}.csv"
PROCESSED_DATA_OUTPUT:str = "process_outputs/processed_data_{year}.json"
SKIPPED_FILE_OUTPUT:str = "process_outputs/skipped_file_details_{year}.csv"
LDA_MODEL_PASSES:int = 15
LDA_MODEL_ITERATIONS:int = 50

def get_inputs() -> None:
    """
    Ask the user for the run settings and store them in global variables:
    OVERRIDE_PROCESSED_DATA_FILE, RUN_LDA_SEPARATELY, USE_BIGRAMS and NUM_TOPICS.
    """
    global OVERRIDE_PROCESSED_DATA_FILE, RUN_LDA_SEPARATELY, USE_BIGRAMS, NUM_TOPICS

    # Set to True to override existing processed data files, False to skip processing if the file exists
    OVERRIDE_PROCESSED_DATA_FILE = input("Override existing processed data files? (y/n): ").strip().lower() == 'y'
    RUN_LDA_SEPARATELY = input("Run LDA separately for each year? (y/n): ").strip().lower() == 'y'
    USE_BIGRAMS = input("Use bigrams? (y/n): ").strip().lower() == 'y'
    NUM_TOPICS = int(input("Enter the number of topics for LDA: ").strip())


def get_one_year_data(year: str) -> dict[str, list[str]]:
    """
    Get the processed data of one year. Loads the saved processed data file if it exists and
    OVERRIDE_PROCESSED_DATA_FILE is False, otherwise runs the PreProcessor and TextProcessor.

    :param year: The year to process, a key of BASE_PATH_DICT.
    :return: A dictionary mapping company names to lists of processed words.
    """
    print(f"[MAIN] --- Getting year data for {year} ---")
    filtered_df_output: str = FILTERED_DF_OUTPUT.format(year=year)
    processed_data_output: str = PROCESSED_DATA_OUTPUT.format(year=year)
    base_path: Path = BASE_PATH_DICT[year]

    preprocessor = PreProcessor(base_path=base_path, year=year, output_file=filtered_df_output)

    if not OVERRIDE_PROCESSED_DATA_FILE and os.path.exists(processed_data_output):
        print(f"[MAIN] Processed data file '{processed_data_output}' already exists. Skipping processing.")
        with open(processed_data_output, "r") as f:
            processed_data = json.load(f)
    else:
        filtered_files_df = preprocessor.run()
        text_processor = TextProcessor(
            filtered_files_df=filtered_files_df,
            base_path=base_path,
            output_file=processed_data_output,
            skipped_output_file=SKIPPED_FILE_OUTPUT.format(year=year),
        )
        processed_data = text_processor.process_all_files()

    return processed_data

def main() -> None:
    """
    Run the whole analysis: ask for the settings, get the data of both years, run LDA,
    save the comparison results and run the post-processing.
    """
    get_inputs()
    prev_data: dict[str, list[str]] = get_one_year_data(
        year='2024'
    )
    curr_data: dict[str, list[str]] = get_one_year_data(
        year='2025'
    )

    print(f"[MAIN] Previous year data: {len(prev_data)} companies")
    print(f"[MAIN] Current year data: {len(curr_data)} companies")

    lda_processor = LDAProcessor(
        curr_data=curr_data,
        prev_data=prev_data,
        num_topics=NUM_TOPICS,
        run_separately=RUN_LDA_SEPARATELY,
        use_bigrams=USE_BIGRAMS,
        num_of_passes=LDA_MODEL_PASSES,
        num_of_iterations=LDA_MODEL_ITERATIONS
    )
    comparison_df = lda_processor.create_comparison()

    comparison_df.to_csv("process_outputs/LDA_comparison_results.csv", index=False)

    post_processor = PostProcessor(
        lda_result_string_path="process_outputs/LDA_comparison_results.csv",
        years=['2024', '2025'],
        skipped_data_path=SKIPPED_FILE_OUTPUT
    )
    post_processor.get_summary()

if __name__ == "__main__":
    main()
