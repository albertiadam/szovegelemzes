"""
The following functions are not used in the main.py file but are useful for analyzing results.
Run this file directly to create the coherence and elbow graphs for 2024 and 2025,
and the topic mapping table of the four saved LDA runs.

Use-cases:
 - Determine the optimal number of topics for LDA using the elbow method
 - Determine the optimal number of topics for LDA using topic coherence (u_mass, optionally c_v)
 - Calculate topic stability statistics from the LDA comparison results
 - Find the stable topics of a run and match them to the topics of the other runs (topic mapping)

Topic mapping (run_topic_stability):
    1. topic_overlap_table: in every run, each 2024 topic is paired with the 2025 topic that receives most of
       its companies, and the overlap of the two topics is calculated. The overlap is the Jaccard index of the
       two sets of companies: companies in both topics / companies in either topic, from 0 to 1.
       It is low both when companies leave the topic and when new companies join it.
    2. The num_stable pairs with the highest overlap in the reference run are its stable topics.
    3. create_topic_mapping_table: each stable topic is matched to the topic of every other run that has the
       highest overlap with it, on the same companies in the same year, separately for 2024 and 2025.
       A match below min_overlap (0.3) is shown as "-".
    Topic numbers start from 1 in the printed tables and in the mapping table, as in the contingency tables,
    while LDA_comparison_results.csv stores them from 0.

Input:
    - Processed data for each year from main.py: process_outputs/processed_data_{year}.json
      (coherence and elbow graphs)
    - LDA comparison results of the four saved runs: output/{run folder}/LDA_comparison_results.csv,
      the folders are listed in RUNS (topic mapping)

Generated outputs (in process_outputs/):
    - Coherence graph: coherence_graph.png
    - Elbow graph: elbow_graph.png (elbow_graph_bigrams.png when bigrams are used)
    - Topic mapping table: topic_mapping.csv and topic_mapping.png
      rows: stable topics of the reference run (2024 topic → 2025 topic),
      columns: every other run and year, cells: matching topic (overlap) or "-"
    - Printed to console: the topic overlap table of every run
    - The graph and table labels are in Hungarian

Settings at the start of the file:
    - YEAR_COLUMNS: the result column of each year
    - RUNS: name and folder of each saved LDA run, the names are used as column labels of the mapping table

Settings that differ from main.py:
    - The elbow graph models use random_state=10, the coherence graph models use random_state=42
    - Every company-year is a separate document, so both years are always modelled combined
    - The topic mapping assumes 6 topics in every run (num_topics=6) and uses
      "Bigram, évenkénti" as the reference run with 3 stable topics (set in the __main__ block)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaModel, CoherenceModel
from gensim import corpora
import json
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from concurrent.futures import ProcessPoolExecutor

BagOfWords = list[tuple[int, int]]
YEAR_COLUMNS: dict[str, str] = {"2024": "previous_dominant_topic", "2025": "current_dominant_topic"}
RUNS: dict[str, str] = {
    "Unigram, közös": "output/no bigram, together",
    "Bigram, közös": "output/bigram, together",
    "Unigram, évenkénti": "output/no bigram, separate",
    "Bigram, évenkénti": "output/bigram, separate",
}


def get_data(years: list[int]) -> dict[str, list[str]]:
    """
    Load the processed data of the given years into one dictionary.

    :param years: Years to load from process_outputs/processed_data_{year}.json.
    :return: A dictionary mapping "{company}_{year}" to the company's list of words.
    """
    output = {}
    for year in years:
        with open(f"process_outputs/processed_data_{year}.json", "r") as f:
            for company, words in json.load(f).items():
                output[f"{company}_{year}"] = words
    return output

def _normalized_std_for_k(args: tuple[int, list[BagOfWords], corpora.Dictionary, int, int]) -> float:
    """
    Train an LDA model with k topics and measure how concentrated the documents' topic probabilities are.
    Runs in a worker process, so all inputs come in one tuple.

    The standard deviation of each document's topic probabilities is averaged, then divided by its
    largest possible value, sqrt(k - 1) / k, so different k values can be compared:
    1 means every document belongs fully to one topic, 0 means every document is spread evenly.

    :param args: A tuple of (k, corpus, dictionary, passes, iterations).
    :return: The normalized mean standard deviation.
    """
    k, corpus, dictionary, passes, iterations = args
    model = LdaModel(
        corpus=corpus, num_topics=k, id2word=dictionary,
        passes=passes, iterations=iterations,
        random_state=10, eval_every=None,
    )
    gamma, _ = model.inference(corpus)
    probs = gamma / gamma.sum(axis=1, keepdims=True)
    mean_std = np.mean(np.std(probs, axis=1))
    return mean_std / (np.sqrt(k - 1) / k)


def create_elbow_graph(
    data: dict[str, list[str]],
    max_topics: int = 10,
    bigrams: bool = False,
    passes: int = 15,
    iterations: int = 50,
) -> tuple[list[int], list[float]]:
    """
    Create the elbow graph: the normalized standard deviation of topic probabilities
    (see _normalized_std_for_k) for each number of topics from 2 to max_topics.
    The models are trained in parallel. The graph is saved to process_outputs/ and shown.

    :param data: A dictionary mapping documents to lists of words (output of get_data).
    :param max_topics: The largest number of topics to try.
    :param bigrams: True to use pairs of neighbouring words instead of single words.
    :param passes: Number of passes through the corpus during training.
    :param iterations: Maximum number of iterations per document during training.
    :return: A tuple containing the numbers of topics tried and their normalized standard deviations.
    """
    k_values = list(range(2, max_topics + 1))
    if bigrams:
        data = {
            company: [f"{doc[i]} {doc[i + 1]}" for i in range(len(doc) - 1)]
            for company, doc in data.items()
        }
    dictionary = corpora.Dictionary(data.values())
    dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=None)
    corpus = [dictionary.doc2bow(doc) for doc in data.values()]

    jobs = [(k, corpus, dictionary, passes, iterations) for k in k_values]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(k_values))) as executor:
        stds = list(executor.map(_normalized_std_for_k, jobs))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, stds, marker="o", label="Topikvalószínűségek normalizált szórása")
    plt.title("Könyökmódszer az optimális topikszám meghatározásához")
    plt.xlabel("Topikok száma (k)")
    plt.ylabel("Normalizált szórás")
    plt.xticks(k_values)
    plt.grid(True)
    suffix = "_bigrams" if bigrams else ""
    plt.savefig(f"process_outputs/elbow_graph{suffix}.png", bbox_inches="tight")
    plt.show()

    return k_values, stds

def _score_k(
    args: tuple[int, list[list[str]], list[BagOfWords], corpora.Dictionary, int, int, bool]
) -> tuple[int, float, float | None]:
    """
    Train an LDA model with k topics and calculate its topic coherence.
    Runs in a worker process, so all inputs come in one tuple.

    :param args: A tuple of (k, texts, corpus, dictionary, passes, iterations, use_cv).
    :return: A tuple of (k, u_mass coherence, c_v coherence or None if use_cv is False).
    """
    k, texts, corpus, dictionary, passes, iterations, use_cv = args
    model = LdaModel(
        corpus=corpus, num_topics=k, id2word=dictionary,
        passes=passes, iterations=iterations,
        random_state=42, eval_every=None,
    )
    u_mass = CoherenceModel(
        model=model, corpus=corpus, dictionary=dictionary, coherence="u_mass"
    ).get_coherence()
    c_v = None
    if use_cv:
        c_v = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary,
            coherence="c_v", processes=1,
        ).get_coherence()
    return k, u_mass, c_v


def create_coherence_graph(
    data: dict[str, list[str]],
    max_topics: int = 10,
    use_cv: bool = False,
    passes: int = 15,
    iterations: int = 50,
) -> list[tuple[int, float, float | None]]:
    """
    Create the coherence graph: u_mass coherence (and optionally c_v on a second axis)
    for each number of topics from 2 to max_topics.
    The models are trained in parallel. The graph is saved to process_outputs/ and shown.

    :param data: A dictionary mapping documents to lists of words (output of get_data).
    :param max_topics: The largest number of topics to try.
    :param use_cv: True to also calculate c_v coherence (much slower).
    :param passes: Number of passes through the corpus during training.
    :param iterations: Maximum number of iterations per document during training.
    :return: A list of (k, u_mass, c_v) tuples, c_v is None if use_cv is False.
    """
    k_values = list(range(2, max_topics + 1))
    texts = list(data.values())
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=None)
    corpus = [dictionary.doc2bow(t) for t in texts]

    jobs = [(k, texts, corpus, dictionary, passes, iterations, use_cv) for k in k_values]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(k_values))) as executor:
        results = list(executor.map(_score_k, jobs))

    ks = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, [r[1] for r in results], marker="o", label="u_mass")
    ax.set_xlabel("Topikok száma (k)")
    ax.set_ylabel("u_mass (minél közelebb van a 0-hoz, annál jobb)")
    ax.set_xticks(ks)
    ax.grid(True)
    if use_cv:
        ax2 = ax.twinx()
        ax2.plot(ks, [r[2] for r in results], marker="s", color="tab:orange", label="c_v")
        ax2.set_ylabel("c_v (minél nagyobb, annál jobb)")
    plt.title("Topikkoherencia a topikok számának függvényében")
    fig.savefig("process_outputs/coherence_graph.png", bbox_inches="tight")
    plt.show()
    return results

def topic_stats(results_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate how stable the companies' dominant topics are between the two years.

    :param results_df: LDA comparison results (company, previous_dominant_topic, current_dominant_topic).
    :return: A dictionary containing
        - n: number of companies with a dominant topic in both years
        - cramers_v: strength of the relationship between the two years' topics (0 to 1)
        - stability: share of companies whose dominant topic did not change
        - chance: share expected to stay the same by chance if the topics of the two years were unrelated
    """
    df = results_df.drop_duplicates(subset="company").dropna()
    prev = df["previous_dominant_topic"].astype(int)
    curr = df["current_dominant_topic"].astype(int)

    table = pd.crosstab(prev, curr)
    chi2, *_ = chi2_contingency(table)
    cramers_v = (chi2 / (table.values.sum() * (min(table.shape) - 1))) ** 0.5

    stability = (prev == curr).mean()
    chance = (prev.value_counts(normalize=True) * curr.value_counts(normalize=True)).sum()

    return {"n": len(df), "cramers_v": cramers_v, "stability": stability, "chance": chance}

def _clean(results_df: pd.DataFrame) -> pd.DataFrame:
    """ One row per company with integer topics, indexed by company. """
    return results_df.drop_duplicates(subset="company").dropna().set_index("company").astype(int)
 
 
def _overlap(labels_a: pd.Series, labels_b: pd.Series, num_topics: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-tabulate two topic labellings of the same companies and calculate the overlap of every topic pair.
 
    :param labels_a: Topic of every company in the first labelling (rows of the result).
    :param labels_b: Topic of every company in the second labelling (columns of the result).
    :param num_topics: Number of topics in both labellings.
    :return: A tuple of (company counts, overlap), both num_topics x num_topics.
    """
    counts = pd.crosstab(labels_a, labels_b).reindex(
        index=range(num_topics), columns=range(num_topics), fill_value=0
    ).values
    union = counts.sum(axis=1)[:, None] + counts.sum(axis=0)[None, :] - counts
    return counts, counts / np.maximum(union, 1)
 
 
def topic_overlap_table(results_df: pd.DataFrame, num_topics: int = 6) -> pd.DataFrame:
    """
    Measure the stability of the topics of one run. Each 2024 topic is paired with the 2025 topic that
    receives most of its companies, and the overlap of the two topics is calculated. The overlap takes
    both the companies leaving the topic and the new companies joining it into account.
 
    :param results_df: LDA comparison results of one run.
    :param num_topics: Number of topics of the run.
    :return: A DataFrame with one row per 2024 topic: its 2025 topic and the overlap (0 to 1).
    """
    df = _clean(results_df)
    counts, overlap = _overlap(df[YEAR_COLUMNS["2024"]], df[YEAR_COLUMNS["2025"]], num_topics)
    best = counts.argmax(axis=1)
    return pd.DataFrame({
        "2024 topic": range(1, num_topics + 1),
        "2025 topic": best + 1,
        "overlap": overlap[range(num_topics), best].round(2),
    })
 
 
def create_topic_mapping_table(
    results: dict[str, pd.DataFrame],
    reference_name: str,
    stable_topics: list[tuple[int, int]],
    num_topics: int = 6,
    min_overlap: float = 0.3,
) -> pd.DataFrame:
    """
    Create the topic mapping table: the stable topics of the reference run (rows) and their best matching
    topics in every other run, separately for each year (columns). The topics of two runs are matched on the
    same companies in the same year. A cell contains the matching topic and the overlap, or "-" if the
    best overlap is below min_overlap. The table is saved to process_outputs/topic_mapping.csv and .png.
 
    :param results: A dictionary mapping the name of each run to its LDA comparison results.
    :param reference_name: The name of the reference run, a key of results.
    :param stable_topics: The stable topics of the reference run as (2024 topic, 2025 topic) pairs,
        e.g. [(4, 6), (5, 5), (6, 3)].
    :param num_topics: Number of topics of the runs.
    :param min_overlap: The smallest overlap that still counts as a match.
    :return: The mapping table as a DataFrame of text cells.
    """
    reference = _clean(results[reference_name])
    columns = {}
    for name, results_df in results.items():
        if name == reference_name:
            continue
        run = _clean(results_df)
        companies = run.index.intersection(reference.index)
        for position, (year, column) in enumerate(YEAR_COLUMNS.items()):
            _, overlap = _overlap(run.loc[companies, column], reference.loc[companies, column], num_topics)
            cells = []
            for pair in stable_topics:
                topic = pair[position] - 1
                match = overlap[:, topic].argmax()
                cells.append(
                    "-" if overlap[match, topic] < min_overlap
                    else f"{match + 1} ({overlap[match, topic]:.2f})".replace(".", ",")
                )
            columns[(name, year)] = cells
    table = pd.DataFrame(columns, index=[f"{previous} → {current}" for previous, current in stable_topics])
    table.to_csv("process_outputs/topic_mapping.csv")
 
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.5 * len(table)))
    ax.axis("off")
    ax.table(
        cellText=table.values, rowLabels=table.index, colLabels=[f"{name}\n{year}" for name, year in table.columns],
        cellLoc="center", loc="center",
    ).scale(1, 1.8)
    ax.set_title(f"Témák megfeleltetése, referencia: {reference_name} (sorok: stabil témák 2024 → 2025)")
    fig.savefig("process_outputs/topic_mapping.png", bbox_inches="tight", dpi=200)
    plt.show()
    return table
 
 
def run_topic_stability(run_folders: dict[str, str], reference_name: str, num_stable: int = 3) -> pd.DataFrame:
    """
    Run the whole topic stability analysis: print the stability of the topics of every run, choose the
    stable topics of the reference run (the ones with the highest overlap) and create the topic mapping table.
 
    :param run_folders: A dictionary mapping the name of each run to the folder of its LDA_comparison_results.csv.
    :param reference_name: The name of the reference run, a key of run_folders.
    :param num_stable: Number of stable topics of the reference run.
    :return: The topic mapping table (see create_topic_mapping_table).
    """
    results = {name: pd.read_csv(f"{folder}/LDA_comparison_results.csv") for name, folder in run_folders.items()}
    for name, results_df in results.items():
        print(f"{name}\n{topic_overlap_table(results_df)}\n")
    overlap = topic_overlap_table(results[reference_name])
    stable = overlap.nlargest(num_stable, "overlap").sort_values("2024 topic")
    stable_topics = list(stable[["2024 topic", "2025 topic"]].itertuples(index=False, name=None))
    return create_topic_mapping_table(results, reference_name, stable_topics)


if __name__ == "__main__":
    data = get_data([2024, 2025])
    coherence_data = create_coherence_graph(data, max_topics=10, use_cv=True)
    elbow_data = create_elbow_graph(data,max_topics=10, bigrams=False)

    run_topic_stability(RUNS, reference_name="Bigram, évenkénti")

    print(f"Coherence Data: {coherence_data}")
    print(f"Elbow Data: {elbow_data}")
