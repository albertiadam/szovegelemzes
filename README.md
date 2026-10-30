# Topic modelling of 10-K cybersecurity disclosures

Text analysis project for a thesis.

Since fiscal years ending on or after December 15, 2023, the SEC requires US public companies to describe their
cybersecurity risk management, strategy and governance in a new section of the annual report (10-K):
**Item 1C – Cybersecurity**. This project extracts that section from the 10-K filings of 2024 and 2025,
finds the main themes in them with LDA (Latent Dirichlet Allocation) topic modelling, and measures
how much each company's main theme changed between the two years.

## How it works

The analysis runs in four steps, each in its own module:

| Step | Module | What it does |
|---|---|---|
| 1. Preprocessing | [preprocess.py](preprocess.py) | Reads the header of every filing and selects the 10-Ks to analyse |
| 2. Text processing | [text_processor.py](text_processor.py) | Extracts the Item 1C section and turns it into a list of lemmatized words |
| 3. Topic modelling | [lda_processor.py](lda_processor.py) | Trains LDA and assigns each company a dominant topic in each year |
| 4. Post-processing | [post_processor.py](post_processor.py) | Compares the topics of the two years and summarizes the skipped companies |

[main.py](main.py) runs the four steps in order. [utils.py](utils.py) is run separately, to choose the number of topics.

### 1. Preprocessing – selecting the filings

- keeps files with "K" in the file name (10-K filings and their variants)
- keeps filings whose period of report is on or after December 15 of the previous year, so that Item 1C is required
- drops amendments (10-K/A, 10-KT/A), since they only restate the changed items and may lack Item 1C
- drops asset-backed entities, trusts and funds by company name keyword (`trust`, `receivables`, `securitization`,
  `mortgage`, `fund`), since they are not operating companies
- keeps one filing per company: the latest period of report, then the latest submit date, then the highest filing sequence

### 2. Text processing – extracting Item 1C

For every selected filing:

1. HTML tags and entities are removed and whitespace is collapsed.
2. The Item 1C heading is located with a regular expression that tolerates split words ("C yber Security") and
   skips cross-references ("see Item 1C. Cybersecurity") and table of contents entries.
3. The end of the section is the next heading: Item 2 – Properties, or Item 1D / Item 3 / Item 4 /
   "Executive Officers of the Registrant" when Item 2 is missing or comes later.
4. The section is cleaned, stopwords are removed (gensim stopword list) and the words are lemmatized
   with spaCy (`en_core_web_sm`).

A filing is skipped, and the reason is recorded, when:

| Reason | Meaning |
|---|---|
| `missing_item_1c_cybersecurity` | no Item 1C heading found |
| `missing_section_end` | no heading found after Item 1C |
| `item_1c_too_short` | fewer than 20 words |
| `item_1c_too_long` | more than 5,000 words, meaning the end heading was missed |
| `item_1c_wrong_start` | the section contains the Item 1B heading, meaning the start was a cross-reference |
| `empty_after_processing` | no words left after cleaning |

### 3. Topic modelling

- **Dictionary:** words that appear in fewer than 20 documents or in more than 50% of documents are dropped.
- **Model:** gensim `LdaModel` with 15 passes, 50 iterations and `random_state=42`.
- **Combined or separate:** LDA is trained either on both years together (one shared set of topics), or on each year
  separately.
- **Bigrams:** optionally, pairs of neighbouring words ("incident response", ...) are used instead of single words.
- **Dominant topic:** the topic with the highest probability in a company's Item 1C. It is assigned for every company
  that has usable text in both years.

### 4. Post-processing

- **Transition table:** a contingency table of 2024 topic (rows) against 2025 topic (columns).
- **Chi-square test of independence:** tests whether a company's 2025 topic is related to its 2024 topic.
- **Cramér's V:** the strength of that relationship, from 0 (unrelated) to 1 (fully determined).
- **Skipped companies:** counted by reason and by industry (SIC description), saved as tables and shown in two charts.
  Every skipped filing is counted, so a company skipped in both years counts once in each year.
  In the charts the three rare reasons (wrong start, too long, missing section end) are merged into "other".
  The chart titles, axis labels and reasons are in Hungarian (e.g. "other" is "Egyéb"), the industry names stay in English.
  - *All skipped filings:* count per reason, and the 10 industries with the most skipped filings, each bar stacked by reason.
  - *By year:* the share of all filings skipped for each reason, and the share of each industry's filings skipped,
    2024 and 2025 side by side. Shares are used instead of counts, because the number of filings differs between
    the years and between the industries.

## Data

The input is the **EDGAR 10-X filings** for 2024 and 2025 as plain-text files, one file per filing, in the format
of the Loughran–McDonald "10-X_C" files (Notre Dame Software Repository for Accounting and Finance).
The data is not included in this repository.

Expected folder layout:

```
10-X_C_2024/2024/
├── QTR1/
├── QTR2/
│   ├── 20240417_10-K_edgar_data_1877461_0001477932-24-002182.txt
│   └── ...
├── QTR3/
└── QTR4/
```

File names follow the pattern `{submit date}_{form}_edgar_data_{CIK}_{accession number}.txt`.
The company name, period of report and industry (SIC code) are read from the header of each file.

## Setup

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

The spaCy model `en_core_web_sm` is installed by `requirements.txt` as well.

## Running the analysis

1. Set the data folders in `BASE_PATH_DICT` at the top of [main.py](main.py).
2. Create the output folder: `mkdir process_outputs`
3. Run:

   ```bash
   python main.py
   ```

4. Answer the questions:

   | Question | Effect |
   |---|---|
   | Override existing processed data files? (y/n) | `n` reuses `processed_data_{year}.json` from an earlier run and skips steps 1–2, which are the slowest |
   | Run LDA separately for each year? (y/n) | `y` trains one model per year, `n` one model on both years combined |
   | Use bigrams? (y/n) | `y` uses pairs of neighbouring words instead of single words |
   | Enter the number of topics for LDA | number of topics (k) |

The number of passes and iterations can be changed with `LDA_MODEL_PASSES` and `LDA_MODEL_ITERATIONS` in
[main.py](main.py). The other settings are listed in the docstring at the top of each module.

### Choosing the number of topics and matching the topics

```bash
python utils.py
```

This trains LDA for k = 2 to 10 on the processed data of both years and draws two graphs:

- **Coherence graph** (`coherence_graph.png`): u_mass and c_v topic coherence for each k. Higher c_v and u_mass
  closer to 0 are better.
- **Elbow graph** (`elbow_graph.png`): how concentrated the companies' topic probabilities are for each k
  (1 = every company belongs fully to one topic, 0 = evenly spread). The bend of the curve suggests a suitable k.

Then it matches the topics of the four saved LDA runs in [output/](output/) (listed in `RUNS` in [utils.py](utils.py)):

1. **Topic overlap:** in every run, each 2024 topic is paired with the 2025 topic that receives most of its companies.
   Their overlap is the Jaccard index of the two sets of companies (companies in both / companies in either, 0 to 1),
   so it is low both when companies leave the topic and when new companies join it. The table of every run is
   printed to the console.
2. **Stable topics:** the 3 pairs with the highest overlap in the reference run (`Bigram, évenkénti`).
3. **Topic mapping table** (`topic_mapping.csv`, `topic_mapping.png`): each stable topic of the reference run (rows,
   "2024 topic → 2025 topic") is matched to the topic with the highest overlap in every other run, on the same
   companies, separately for 2024 and 2025 (columns). A cell shows the matching topic and the overlap, e.g. `6 (0,82)`,
   or `-` if the best overlap is below 0.3.

Topic numbers start from 1, as in the contingency tables. The graph and table labels are in Hungarian.

`main.py` has to be run once before, so that `process_outputs/processed_data_{year}.json` exists, and the
`LDA_comparison_results.csv` of the four runs have to be in their [output/](output/) folders.
[utils.py](utils.py) also contains `topic_stats`, which calculates the share of companies whose topic did not change
and the share expected by chance.

## Outputs

Every run writes to `process_outputs/` (not tracked by git):

| File | Content |
|---|---|
| `filtered_file_details_{year}.csv` | filings selected in step 1 |
| `abs_companies_{year}.csv` | filings dropped as asset-backed entities, trusts or funds |
| `processed_data_{year}.json` | company name → list of lemmatized words of Item 1C |
| `skipped_file_details_{year}.csv` | filings skipped in step 2, with the reason |
| `LDA_comparison_results.csv` | company, `previous_dominant_topic`, `current_dominant_topic` |
| `prev_lda_visualization.html`, `curr_lda_visualization.html` | interactive pyLDAvis topic visualizations (open in a browser) |
| `contingency_table.csv` | 2024 × 2025 topic transition table, with totals |
| `grouped_skipped_companies.csv` | skipped companies by industry and reason |
| `reason_counts.csv`, `industry_counts.csv` | skipped companies by reason and by industry |
| `skipped_companies_counts.png` | chart of all skipped filings by reason, and the top 10 industries with each bar stacked by reason |
| `skipped_companies_rates_by_year.png` | chart of the share of filings skipped by reason and in the top 10 industries, 2024 vs 2025 |
| `coherence_graph.png`, `elbow_graph.png` | graphs from `utils.py` |
| `topic_mapping.csv`, `topic_mapping.png` | topic mapping table of the four saved LDA runs, from `utils.py` |

The chi-square statistic, p-value, Cramér's V and the skipped company counts by reason and by industry
are printed to the console.

### Saved results

The [output/](output/) folder holds the results used in the thesis:

| Folder | Content |
|---|---|
| `data (for all)` | filtering and skipped company results, the skipped company charts, the graphs used to choose k (`coherence_graph.png`, `elbow_graph.png`), and the topic mapping table of the four runs (`topic_mapping.csv`, `topic_mapping.png`). These are the same for every LDA setting |
| `no bigram, together` | single words, one model for both years |
| `no bigram, separate` | single words, one model per year |
| `bigram, together` | bigrams, one model for both years |
| `bigram, separate` | bigrams, one model per year |

Each LDA folder contains the comparison results, the contingency table, the pyLDAvis visualizations and
`contingency.txt` with the printed statistics. `kontingencia_tablak.xlsx` holds the four contingency tables
in one 2×2 grid, for the thesis.

## Limitations

- **Amendments:** a company is dropped if only its amendment (10-K/A) contains Item 1C.
- **Name keywords:** filtering by company name keyword also drops operating companies with such a word in their name
  (e.g. "Northern Trust Corp").
- **Section extraction:** Item 1C is found with regular expressions. Filings with unusual headings are skipped,
  and the skipped companies are summarized by reason and industry to show what was left out.
- **Separate models:** when LDA runs separately for each year, topic numbers are not matched between the years.
  Topic 0 in 2024 is not necessarily the same theme as topic 0 in 2025, so the transition table and Cramér's V
  are only meaningful for the combined model, or after matching the topics. The topic mapping in
  [utils.py](utils.py) matches topics by their companies, not by their words, so a match means the same companies,
  which usually but not necessarily means the same theme.
- **Fixed years:** the years 2024 and 2025 are set in `main()` in [main.py](main.py).
