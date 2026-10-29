"""
The following functions are not used in the main.py file but are useful for analyzing results.
Run this file directly to create the coherence graph for 2024 and 2025.
The elbow graph call is commented out at the bottom of the file, uncomment it to create the elbow graph as well.

Use-cases:
 - Determine the optimal number of topics for LDA using the elbow method
 - Determine the optimal number of topics for LDA using topic coherence (u_mass, optionally c_v)
 - Calculate topic stability statistics from the LDA comparison results

Input:
    - Processed data for each year from main.py: process_outputs/processed_data_{year}.json

Generated outputs (in process_outputs/):
    - Coherence graph: coherence_graph.png
    - Elbow graph (only if its call is uncommented): elbow_graph.png (elbow_graph_bigrams.png when bigrams are used)
    - The graph labels are in Hungarian

Settings that differ from main.py:
    - The elbow graph models use random_state=10, the coherence graph models use random_state=42
    - Every company-year is a separate document, so both years are always modelled combined
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaModel, CoherenceModel
from gensim import corpora
import json
import pandas as pd
from scipy.stats import chi2_contingency
from concurrent.futures import ProcessPoolExecutor

BagOfWords = list[tuple[int, int]]


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




if __name__ == "__main__":
    data = get_data([2024, 2025])
    coherence_data = create_coherence_graph(data, max_topics=10, use_cv=True)
    #elbow_data = create_elbow_graph(data,max_topics=10, bigrams=False)
    print(f"Coherence Data: {coherence_data}")
    #print(f"Elbow Data: {elbow_data}")
