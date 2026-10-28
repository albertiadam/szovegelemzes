"""
The following functions are not used in the main.py file but are useful for analyzing results

Use-cases:
 - Determine the optimal number of topics for LDA using the elbow method
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


def get_data(years):
    output = {}
    for year in years:
        with open(f"process_outputs/processed_data_{year}.json", "r") as f:
            for company, words in json.load(f).items():
                output[f"{company}_{year}"] = words
    return output

def _normalized_std_for_k(args):
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


def create_elbow_graph(data, max_topics=10, bigrams=False, passes=15, iterations=50):
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
    plt.plot(k_values, stds, marker="o", label="Normalized Std of Topic Probabilities")
    plt.title("Elbow Method for Optimal Number of Topics")
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Normalized Std")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    return k_values, stds

def _score_k(args):
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


def create_coherence_graph(data, k_values, use_cv=False, passes=15, iterations=50):
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
    ax.set_xlabel("Number of Topics (k)")
    ax.set_ylabel("u_mass (closer to 0 is better)")
    ax.set_xticks(ks)
    ax.grid(True)
    if use_cv:
        ax2 = ax.twinx()
        ax2.plot(ks, [r[2] for r in results], marker="s", color="tab:orange", label="c_v")
        ax2.set_ylabel("c_v (higher is better)")
    plt.title("Topic coherence by number of topics")
    plt.show()
    return results

def topic_stats(results_df):
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
    res = create_elbow_graph(get_data([2024, 2025]))
    print(res)
