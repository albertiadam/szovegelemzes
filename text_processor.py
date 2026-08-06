import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
import spacy

WORKER = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def _process_single_file(file_path: Path) -> list[str]:
    cleaned_words = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    start_idx = text.find("Item 1C. Cybersecurity")
    if start_idx == -1:
        return cleaned_words

    start_idx += len("Item 1C. Cybersecurity")
    end_idx = text[start_idx:].find("Item 2. Properties.")
    if end_idx == -1:
        return cleaned_words
    else:
        end_idx += start_idx

    item_1c_text = text[start_idx:end_idx]
    
    for word in item_1c_text.split():
        cleaned = TextProcessor.clean_word(word)
        if cleaned is not None and cleaned not in STOPWORDS:
            cleaned_words.append(cleaned)
    doc = WORKER(" ".join(cleaned_words))
    return [token.lemma_ for token in doc if len(token.lemma_) > 1 and token.lemma_ not in STOPWORDS]


class TextProcessor:
    def __init__(self, filtered_files_df: pd.DataFrame, base_path: str, output_file:str):
        self.filtered_files_df = filtered_files_df
        self.base_path = base_path
        self.output_file = output_file

    @staticmethod
    def clean_word(word):
        CHARS = ['[', ']', '(', ')', '{', '}', "'", '"', '.', ',', ':', '\\', '-', '_', ';']
        pattern = r'<([a-zA-Z]+)[^>]*>.*?</\1>|<[^>]+>|[\n\t\r]'
        cleaned = re.sub(pattern, '', word, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\(.\)', '', cleaned)
        cleaned = re.sub(r'\b\w+(?:\.\w+)+\b', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\d+', '', cleaned)
        for char in CHARS:
            cleaned = cleaned.replace(char, ' ')
        if len(cleaned) in (0, 1) or cleaned in STOPWORDS:
            return None
        return cleaned.strip().lower()

    def process_all_files(self, workers: int | None = None) -> dict[str, list[str]]:
        if self.filtered_files_df.empty:
            return {}

        file_paths = []
        company_names = []
        for _, row in self.filtered_files_df.iterrows():
            file_paths.append(Path(self.base_path) / Path(row['folder']) / Path(row['file_name']))
            company_names.append(row['company_name'])

        if workers is None:
            workers = min(os.cpu_count() or 1, len(file_paths))

        if workers <= 1:
            return {
                company_name: _process_single_file(file_path)
                for company_name, file_path in zip(company_names, file_paths)
            }

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = []
            for index, result in enumerate(executor.map(_process_single_file, file_paths)):
                results.append(result)
                if index + 1 == len(file_paths) or (index + 1) % max(1, len(file_paths) // 10) == 0:
                    print(f"Processed {index + 1}/{len(file_paths)} files...")

        final_result = {
            company_name: result
            for company_name, result in zip(company_names, results) if len(result) > 0
        }

        with open(self.output_file, "w") as f:
            json.dump(final_result,f)

        return final_result
    
