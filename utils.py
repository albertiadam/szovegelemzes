"""
The following functions are not used in the main.py file but are useful for analyzing results

Use-cases:
 - Determine the optimal number of topics for LDA using the elbow method
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim import corpora
import json

def get_data() -> dict[str, list[str]]:
    with open("process_outputs/processed_data_2024.json", "r") as f:
        prev_data = json.load(f)
    with open("process_outputs/processed_data_2025.json", "r") as f:
        curr_data = json.load(f)

    all_data = {**prev_data, **curr_data}
    return all_data

def create_elbow_graph(data: dict[str, list[str]],max_topics:int=10,bigrams:bool=False) -> tuple[list[int], list[float]]:
    k_values = list(range(1, max_topics + 1))
    stds = []
    if bigrams:
        data = {
            company: [
                f"{doc[index]} {doc[index + 1]}"
                for index in range(len(doc) - 1)
            ]
            for company, doc in data.items()
        }
    dictionary = corpora.Dictionary(data.values())

    corpus = [dictionary.doc2bow(doc) for doc in data.values()]
    for k in k_values:
        print("Training LDA model for k =", k)
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=k,
            id2word=dictionary,
            passes=15,
            iterations=50
        )
        gamma, _ = lda_model.inference(corpus)
        gamma_probs = gamma / np.sum(gamma, axis=1, keepdims=True)
        mean_std = np.mean(np.std(gamma_probs, axis=1))
        max_possible_std = np.sqrt((k - 1) / k)

        normalized_std = mean_std / max_possible_std
        stds.append(normalized_std)


    plt.figure(figsize=(10, 6))
    plt.plot(k_values, stds, marker='o', label='Normalized Std of Topic Probabilities')
    plt.title('Elbow Method for Optimal Number of Topics')
    plt.xlabel('Number of Topics (k)')
    plt.ylabel('Normalized Std')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    return k_values, stds



data = get_data()
k_values, stds = create_elbow_graph(data)
