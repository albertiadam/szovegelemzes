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
MIN_ITEM_1C_WORDS = 20

def _process_single_file_with_status(
    file_path: Path,
) -> tuple[list[str], str]:
    """
    Process a single file to extract and clean words from the "Item 1C - Cybersecurity" section.

    :param file_path: Path to the file to be processed.
    :return: A tuple containing a list of cleaned words and a status string indicating the result of processing.
            "missing_item_1c_cybersecurity" if the "Item 1C - Cybersecurity" section is not found,
            "missing_section_end" if the section end is not found,
            "item_1c_too_short" if the "Item 1C - Cybersecurity" section is found but too short,
            "empty_after_processing" if no valid tokens remain after processing,
            "processed" if the file was successfully processed and contains valid tokens.
    """
    cleaned_words = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    normalized_text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    
    # Search for the "Item 1C - Cybersecurity" section
    # Tolerates words split by the parser ("I tem", "C yber Security") and "Item IC".
    # Skips cross-references ("see Part I, Item 1C. Cybersecurity of this Report")
    # and table of contents entries ("Item 1C. Cybersecurity 20 Item 2. Properties").
    start_match = re.search(
        r"(?<![,\"“])(?<!,\s)(?<!\bsee\s)(?<!\bin\s)(?<!\bto\s)(?<!\band\s)(?<!under\s)(?<!within\s)"
        r"(?<!item\s)(?<!item)"
        r"\b(?:i\s?t\s?e\s?m\s*)?[1I]\s*C\b[^\w]{0,40}"
        r"c\s?y\s?b\s?e\s?r\W{0,3}s\s?e\s?c\s?u\s?r\s?i\s?t\s?y\b"
        r"(?![^\w]*(?:\d{1,3}[^\w]+)?(?:i\s?t\s?e\s?m\s*)?(?:[2-4]\b|1\s*c\b))"
        r"(?![^\w]*(?:of\s+th|of\s+part|in\s+th|below|above|for\b|herein))"
        r"[^\w]*",
        normalized_text,
        re.IGNORECASE,
    )
    if start_match is None:
        return cleaned_words, "missing_item_1c_cybersecurity"

    start_idx = start_match.end()

    # Search for the "Item 2 - Properties" section to determine the end of the "Item 1C - Cybersecurity" section
    # Also accepts "Description of Properties" and split words ("P roperties"). Falls back
    # to Item 3 / Item 4 for filings that merge Item 2 into Item 1 ("Business and Properties").
    end_match = re.search(
        r"(?:\bi\s?t\s?e\s?m\s*|\b(?=[2-4]\s*\.))[2-4]\b[^\w]{0,40}"
        r"(?:description\s+of\s+)?(?:p\s?r\s?o\s?p\s?e\s?r\s?t|l\s?e\s?g\s?a\s?l\s+p|m\s?i\s?n\s?e\s+s\s?a\s?f\s?e\s?t\s?y)",
        normalized_text[start_idx:],
        re.IGNORECASE,
    )
    if end_match is None:
        return cleaned_words, "missing_section_end"

    item_1c_text = normalized_text[start_idx:start_idx + end_match.start()]

    if len(item_1c_text.split()) < MIN_ITEM_1C_WORDS:
        return cleaned_words, "item_1c_too_short"

    # Clean and tokenize the extracted text
    for word in item_1c_text.split():
        cleaned = TextProcessor.clean_word(word)
        if cleaned is not None and cleaned not in STOPWORDS:
            cleaned_words.append(cleaned)
    
    # Create a spaCy Doc object for lemmatization and further processing
    doc = WORKER(" ".join(cleaned_words))

    # Lemmatize and filter tokens
    tokens = [token.lemma_ for token in doc if len(token.lemma_) > 1 and token.lemma_ not in STOPWORDS]
    if not tokens:
        return tokens, "empty_after_processing"
    return tokens, "processed"


class TextProcessor:
    """ TextProcessor class to process and clean text files, extracting relevant tokens. """
    def __init__(
        self,
        filtered_files_df: pd.DataFrame,
        base_path: Path,
        output_file: str,
        skipped_output_file: str | None = None,
    ):
        """
        Initialize the TextProcessor with the filtered files DataFrame, base path, and output file names.

        :param filtered_files_df: DataFrame containing the filtered files to process.
        :param base_path: The base path where the files are located.
        :param output_file: The name of the output file to save processed tokens.
        :param skipped_output_file: The name of the file to save skipped files and their reasons.
        """
        self.filtered_files_df = filtered_files_df
        self.base_path = base_path
        self.output_file = output_file
        self.skipped_output_file = skipped_output_file

    @staticmethod
    def clean_word(word):
        """
        Clean a single word by removing unwanted characters and patterns.

        :param word: The word to be cleaned.
        :return: The cleaned word or None if it should be skipped.
        """
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
        """
        Process all files in the filtered DataFrame using multiple processes.

        :param workers: Number of worker processes to use. If None, it will use the number of CPU cores.
        :return: A dictionary mapping company names to lists of processed tokens.
        """
        if self.filtered_files_df.empty:
            return {}

        file_paths = []
        company_names = []
        # Prepare file paths and company names from the filtered DataFrame
        for _, row in self.filtered_files_df.iterrows():
            file_paths.append(Path(self.base_path) / Path(row['folder']) / Path(row['file_name']))
            company_names.append(row['company_name'])

        # Determine the number of worker processes to use
        if workers is None:
            workers = min(os.cpu_count() or 1, len(file_paths))


        
        if workers <= 1:
            processing_results = [
                _process_single_file_with_status(file_path)
                for file_path in file_paths
            ]
        else:  # Use ProcessPoolExecutor to process files in parallel
            with ProcessPoolExecutor(max_workers=workers) as executor:
                processing_results = []
                for index, result in enumerate(executor.map(_process_single_file_with_status, file_paths)):
                    processing_results.append(result)
                    if index + 1 == len(file_paths) or (index + 1) % max(1, len(file_paths) // 10) == 0:
                        print(f"[TEXT_PROCESS] Processed {index + 1}/{len(file_paths)} files...")

        results = [result[0] for result in processing_results]
        statuses = [result[1] for result in processing_results]

        # Save skipped files and their reasons to a CSV file if specified
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
    
