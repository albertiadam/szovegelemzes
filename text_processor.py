import os
import re
import html
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
import spacy

WORKER = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def _process_single_file_with_status(
    file_path: Path,
) -> tuple[list[str], str]:
    cleaned_words = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    normalized_text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    start_match = re.search(
        r"\bitem\s*1c\b[^\w]{0,40}cybersecurity\b",
        normalized_text,
        re.IGNORECASE,
    )
    if start_match is None:
        return cleaned_words, "missing_item_1c_cybersecurity"

    start_idx = start_match.end()
    end_match = re.search(
        r"\bitem\s*2\b[^\w]{0,40}properties\b",
        normalized_text[start_idx:],
        re.IGNORECASE,
    )
    if end_match is None:
        return cleaned_words, "missing_item_2_properties"

    item_1c_text = normalized_text[start_idx:start_idx + end_match.start()]
    
    for word in item_1c_text.split():
        cleaned = TextProcessor.clean_word(word)
        if cleaned is not None and cleaned not in STOPWORDS:
            cleaned_words.append(cleaned)
    doc = WORKER(" ".join(cleaned_words))
    tokens = [token.lemma_ for token in doc if len(token.lemma_) > 1 and token.lemma_ not in STOPWORDS]
    if not tokens:
        return tokens, "empty_after_processing"
    return tokens, "processed"


class TextProcessor:
    def __init__(
        self,
        filtered_files_df: pd.DataFrame,
        base_path: str,
        output_file: str,
        skipped_output_file: str | None = None,
    ):
        self.filtered_files_df = filtered_files_df
        self.base_path = base_path
        self.output_file = output_file
        self.skipped_output_file = skipped_output_file

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
            processing_results = [
                _process_single_file_with_status(file_path)
                for file_path in file_paths
            ]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                processing_results = []
                for index, result in enumerate(executor.map(_process_single_file_with_status, file_paths)):
                    processing_results.append(result)
                    if index + 1 == len(file_paths) or (index + 1) % max(1, len(file_paths) // 10) == 0:
                        print(f"Processed {index + 1}/{len(file_paths)} files...")

        results = [result[0] for result in processing_results]
        statuses = [result[1] for result in processing_results]

        if self.skipped_output_file is not None:
            skipped_rows = []
            for index, status in enumerate(statuses):
                if status != "processed":
                    row = self.filtered_files_df.iloc[index].to_dict()
                    row["reason"] = status
                    skipped_rows.append(row)
            pd.DataFrame(skipped_rows).to_csv(self.skipped_output_file, index=False)

        final_result = {
            company_name: result
            for company_name, result in zip(company_names, results) if len(result) > 0
        }

        with open(self.output_file, "w") as f:
            json.dump(final_result,f)

        return final_result
    
