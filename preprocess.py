from pathlib import Path
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor



class PreProcessor:
    _ABS_KEYWORDS = ['trust', 'receivables', 'securitization', 'mortgage','fund']

    def __init__(self, base_path: Path, year:str, output_file:str="filtered_file_details.csv"):
        self.base_path = base_path
        self.output_file = output_file
        self.cyber_security_needed_date = int(f"{year}1218")

    def preprocess_file(self, file_folder: list[str]) -> list[str]:
        file, folder = file_folder
        try:
            with open(self.base_path / folder / file,"r") as f:
                text = f.read(10_000).replace("\n","").replace("\t","")
                period_idx = text.find("CONFORMED PERIOD OF REPORT")
                if period_idx == -1:
                    return None
                period_of_report = int(text[
                    period_idx + 27
                    :
                    period_idx + 35
                ])
                if period_of_report > self.cyber_security_needed_date:
                    company_name = text[
                        text.find("COMPANY CONFORMED NAME")
                        :
                        text.find("CENTRAL INDEX KEY")
                    ].split(":")[-1]
                    accession_number = "-".join(file[file.find("edgar_data_") + 11:].split("-")[:2])
                
                    return [folder,file,period_of_report,company_name,accession_number]
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return None
            
    def get_files(self) -> list[list[str]]:
        folders = os.listdir(self.base_path)
        print(f"Found {len(folders)} folders in base path.")
        files = []
        for folder in folders:
            folder_path = self.base_path / folder
            if os.path.isdir(folder_path):
                files.extend([[f,folder] for f in os.listdir(folder_path) if os.path.isfile(folder_path / f)])
        files = [[file,folder] for file,folder in files if "K" in file]
        print(f"Found {len(files)} files matching criteria.")
        return files

    def filter_files(self, file_details_list: list[list[str]]) -> pd.DataFrame:
        filtered_df = pd.DataFrame(file_details_list)
        filtered_df.columns = ["folder", "file_name", "period_of_report", "company_name", "accession_number"]
        filtered_df['submit_date'] = filtered_df['file_name'].str[:8].astype(int)
        filtered_df = filtered_df[~filtered_df['company_name'].str.lower().str.contains("|".join(self._ABS_KEYWORDS))]
        filtered_df['filing_sequence'] = filtered_df['file_name'].apply(lambda x: int(x.split("-")[-1].split(".")[0]))
        filtered_df = filtered_df.sort_values(
            by=['company_name', 'period_of_report', 'submit_date', 'filing_sequence'],
            ascending=[True, True, True, True]
        )
        filtered_df = filtered_df.drop_duplicates(subset=['company_name', 'period_of_report'], keep='last')
        return filtered_df
    
    def run(self) -> pd.DataFrame | None:
        files = self.get_files()
        if files is None:
            print("No files found, stopping preprocessing.")
            return
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.preprocess_file, files))
            file_details_list = [result for result in results if result is not None]
        filtered_df = self.filter_files(file_details_list)
        filtered_df.to_csv(self.output_file, index=False)
        return filtered_df

