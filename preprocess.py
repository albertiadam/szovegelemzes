"""
Preprocessing of the EDGAR 10-X filings.

Selects which filings of a year go into the analysis, based on the header of each file.
Expected folder layout: base_path/<quarter folder, e.g. QTR1>/<filing .txt files>

Filtering steps:
 - keep files with "K" in the file name (10-K filings and their variants)
 - keep filings whose period of report is on or after Dec 15 of the previous year,
   the date from which Item 1C - Cybersecurity is required
 - drop amendments (10-K/A, 10-KT/A)
 - drop asset-backed entities, trusts and funds by company name keyword (_ABS_KEYWORDS)
 - keep one filing per company: the latest period of report, then the latest submit date,
   then the highest filing sequence

Generated outputs:
    - Filtered file details (output_file, e.g. filtered_file_details_{year}.csv):
      folder, file_name, period_of_report, company_name, accession_number, industry, sic,
      submit_date, form, filing_sequence
    - Companies excluded by ABS / trust / fund keyword: process_outputs/abs_companies_{year}.csv
"""
from pathlib import Path
import os
import re
import pandas as pd
import concurrent.futures
from typing import ClassVar


class PreProcessor:
    """ PreProcessor class to filter and preprocess files based on specific criteria. """
    _ABS_KEYWORDS: ClassVar[list[str]] = ['trust', 'receivables', 'securitization', 'mortgage','fund']

    def __init__(self, base_path: Path, year: str, output_file: str = "filtered_file_details.csv") -> None:
        """
        Initialize the PreProcessor with the base path, year, and output file name.
        :param base_path: The base path where the files are located.
        :param year: The year to filter the files.
        :param output_file: The name of the output CSV file to save filtered file details.
        """
        self.base_path = base_path
        self.output_file = output_file
        self.year = year
        self.cyber_security_needed_date = int(f"{int(year) - 1}1215")

    def preprocess_file(self, file_folder: list[str]) -> list[str | int | None] | None:
        """
        Preprocess a single file to extract relevant details.
        Only the first 10,000 characters (the filing header) are read.
        :param file_folder: A list containing the file name and folder name.
        :return: A list containing the extracted details
            [folder, file_name, period_of_report, company_name, accession_number, industry, sic]
            or None if the file does not meet criteria or cannot be read.
        """
        file, folder = file_folder
        try:
            with open(self.base_path / folder / file,"r", encoding="utf-8") as f:

                # Read only the first 10,000 characters to find the period of report and other details
                text = f.read(10_000).replace("\n","").replace("\t","")
                period_idx = text.find("CONFORMED PERIOD OF REPORT")
                if period_idx == -1:
                    return None
                period_of_report = int(text[
                    period_idx + 27
                    :
                    period_idx + 35
                ])
                if period_of_report >= self.cyber_security_needed_date:
                    company_name = text[
                        text.find("COMPANY CONFORMED NAME")
                        :
                        text.find("CENTRAL INDEX KEY")
                    ].split(":")[-1]
                    accession_number = "-".join(file[file.find("edgar_data_") + 11:].split("-")[:2])
                    industry_match = re.search(
                        r"STANDARD INDUSTRIAL CLASSIFICATION\s*:\s*(.*?)\s*\[(\d{4})\]",
                        text,
                        re.IGNORECASE,
                    )
                    industry = industry_match.group(1).strip() if industry_match else None
                    sic = industry_match.group(2) if industry_match else None
                
                    return [folder, file, period_of_report, company_name, accession_number, industry, sic]
        except Exception as e:
            print(f"[PREPROCESS] Error processing file {file}: {e}")
            return None
            
    def get_files(self) -> list[list[str]]:
        """
        Get a list of files in the base path that match specific criteria.
        :return: A list of lists, where each inner list contains the file name and folder name.
        """
        folders = os.listdir(self.base_path)
        files = []
        for folder in folders:
            folder_path = self.base_path / folder
            if os.path.isdir(folder_path):
                files.extend([[f,folder] for f in os.listdir(folder_path) if os.path.isfile(folder_path / f)])

        # Filter files to include only those with "K" in the file name
        files = [[file,folder] for file,folder in files if "K" in file]
        return files

    def filter_files(self, file_details_list: list[list[str | int | None]]) -> pd.DataFrame:
        """
        Filter the list of file details based on specific criteria:
        drops amendments and ABS / trust / fund companies, and keeps one filing per company.
        Saves the companies dropped by ABS keyword to process_outputs/abs_companies_{year}.csv.
        :param file_details_list: List of file details to filter (output of preprocess_file).
        :return: Filtered DataFrame containing the relevant file details.
        """
        filtered_df = pd.DataFrame(file_details_list)
        filtered_df.columns = [
            "folder",
            "file_name",
            "period_of_report",
            "company_name",
            "accession_number",
            "industry",
            "sic",
        ]
        filtered_df['submit_date'] = filtered_df['file_name'].str[:8].astype(int)

        # Amendments (10-K/A, 10-KT/A) are excluded: Rule 12b-15 requires only the amended
        # items to be restated, so they may lack Item 1C. Original filings are complete.
        # Limitation: a company is dropped if only its amendment contains Item 1C.
        filtered_df["form"] = filtered_df["file_name"].str.split("_").str[1]
        is_amendment = filtered_df["form"].str.endswith("-A")
        print(f"[PRE_PROCESS] Excluded {is_amendment.sum()} amendments out of {len(filtered_df)}.")
        filtered_df = filtered_df[~is_amendment]

        # Drop asset-backed entities, trusts and funds, which are not operating companies.
        # Matching is by name keyword, so a real company with such a word in its name
        # (e.g. "Northern Trust Corp") is also dropped.
        
        contains_abs_keywords = filtered_df['company_name'].str.lower().str.contains("|".join(self._ABS_KEYWORDS))
        abs_keyword_df = filtered_df[contains_abs_keywords]
        print(f"[PRE_PROCESS] Excluded {contains_abs_keywords.sum()} companies with ABS keywords out of {len(filtered_df)}.")
        abs_keyword_df.to_csv(f"process_outputs/abs_companies_{self.year}.csv", index=False)
        filtered_df = filtered_df[~contains_abs_keywords]
        
        filtered_df['filing_sequence'] = filtered_df['file_name'].apply(lambda x: int(x.split("-")[-1].split(".")[0]))
        
        # One filing per company: the latest period of report, then the latest submit date,
        # then the highest sequence number.
        filtered_df = filtered_df.sort_values(
            by=['company_name', 'period_of_report', 'submit_date', 'filing_sequence'],
            ascending=[True, True, True, True]
        )
        filtered_df = filtered_df.drop_duplicates(subset=['company_name'], keep='last')
        return filtered_df

    def run(self) -> pd.DataFrame | None:
        """
        Run the preprocessing steps: get files, preprocess them, and filter the results.
        Uses ThreadPoolExecutor to parallelize the preprocessing of files.
        Saves the filtered file details to output_file.

        :return: Filtered DataFrame containing the relevant file details or None if no files were found.
        """
        files = self.get_files()
        if files is None:
            print("[PRE_PROCESS] No files found, stopping preprocessing.")
            return None
        print(f"[PRE_PROCESS] Found {len(files)} for year {self.year} matching criteria 'K' in file name")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.preprocess_file, files))
            file_details_list = [result for result in results if result is not None]
        print(f"[PRE_PROCESS] {len(file_details_list)} files passed the preprocessing criteria.")
        filtered_df = self.filter_files(file_details_list)
        filtered_df.to_csv(self.output_file, index=False)
        print(f"[PRE_PROCESS] {len(filtered_df)} files passed the filtering criteria.\n\n")
        return filtered_df

