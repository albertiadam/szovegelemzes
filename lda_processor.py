"""
LDA topic modelling of the "Item 1C - Cybersecurity" sections.

Trains LDA either on both years combined (one shared set of topics) or on each year separately,
then finds the dominant topic of every company that appears in both years.

Notes:
 - Dictionary filtering: words in fewer than 20 documents or in more than 50% of documents are dropped.
 - With separate models the topic numbers of the two years are not matched: topic 0 in one year
   is not necessarily the same theme as topic 0 in the other year.
 - With bigrams, each document is replaced by its pairs of neighbouring words ("data breach", ...).

Generated outputs (in process_outputs/):
    - Interactive pyLDAvis topic visualizations: prev_lda_visualization.html, curr_lda_visualization.html
"""
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models

BagOfWords = list[tuple[int, int]]

class LDAProcessor:
    """ LDAProcessor class to train LDA models and compare the dominant topics of companies between two years. """
    def __init__(
        self,
        curr_data: dict[str, list[str]],
        prev_data: dict[str, list[str]],
        num_topics: int,
        run_separately: bool = False,
        use_bigrams: bool = False,
        num_of_passes: int = 15,
        num_of_iterations: int = 50,
        random_state: int = 42
    ) -> None:
        """
        Initialize the LDAProcessor with the processed data of the two years and the model settings.

        :param curr_data: Current year data (company name -> list of words).
        :param prev_data: Previous year data (company name -> list of words).
        :param num_topics: Number of LDA topics.
        :param run_separately: True to train one model per year, False to train one model on both years combined.
        :param use_bigrams: True to use pairs of neighbouring words instead of single words.
        :param num_of_passes: Number of passes through the corpus during training.
        :param num_of_iterations: Maximum number of iterations per document during training.
        :param random_state: Seed for reproducible models.
        """
        self.curr_data = curr_data
        self.prev_data = prev_data
        self.num_topics = num_topics
        self.run_separately = run_separately
        self.use_bigrams = use_bigrams
        self.num_of_passes = num_of_passes
        self.num_of_iterations = num_of_iterations
        self.random_state = random_state

    def _prepare_doc(self, doc: list[str]) -> list[str]:
        """
        Prepare a document for LDA: turn it into bigrams if use_bigrams is set, otherwise return it unchanged.

        :param doc: List of words of one company.
        :return: List of words, or list of bigrams ("word1 word2").
        """
        if not self.use_bigrams:
            return doc

        bigrams = [
            f"{doc[index]} {doc[index + 1]}"
            for index in range(len(doc) - 1)
        ]

        return bigrams

    def _get_dictionary_and_corpus(
        self, data: dict[str, list[str]] | None = None
    ) -> tuple[corpora.Dictionary, list[BagOfWords]]:
        """
        Build the gensim dictionary and bag-of-words corpus.
        Words in fewer than 20 documents or in more than 50% of documents are dropped from the dictionary.

        :param data: Data of one year (company name -> list of words). If None, both years are used combined.
        :return: A tuple containing the dictionary and the corpus.
        """
        raw_docs = list(data.values()) if data is not None else list(self.curr_data.values()) + list(self.prev_data.values())
        docs = [self._prepare_doc(doc) for doc in raw_docs]
        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=None)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        return dictionary, corpus

    def _train_lda_model(self, dictionary: corpora.Dictionary, corpus: list[BagOfWords]) -> LdaModel:
        """
        Train an LDA model with the settings given at initialization.

        :param dictionary: Gensim dictionary of the corpus.
        :param corpus: Bag-of-words corpus.
        :return: The trained LDA model.
        """
        return LdaModel(
            corpus=corpus,
            num_topics=self.num_topics,
            id2word=dictionary,
            passes=self.num_of_passes,
            iterations=self.num_of_iterations,
            random_state=self.random_state
        )

    def _get_dominant_topic(self, doc: list[str], dictionary: corpora.Dictionary, model: LdaModel) -> int | None:
        """
        Find the topic with the highest probability for one document.

        :param doc: Prepared list of words (or bigrams) of one company.
        :param dictionary: Gensim dictionary the model was trained with.
        :param model: Trained LDA model.
        :return: The dominant topic number, or None if the document is empty or has no words in the dictionary.
        """
        if not doc:
            return None
        bow = dictionary.doc2bow(doc)
        if not bow:
            return None
        topic_distribution = model.get_document_topics(bow)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        return dominant_topic

    def _export_vis_data(self, output_file: str, dictionary: corpora.Dictionary, model: LdaModel, data: dict[str, list[str]]) -> None:
        """
        Save an interactive pyLDAvis visualization of the model's topics as an HTML file.

        :param output_file: Path of the HTML file.
        :param dictionary: Gensim dictionary the model was trained with.
        :param model: Trained LDA model.
        :param data: Data of the year to visualize (company name -> list of words).
        """
        corpus = [dictionary.doc2bow(self._prepare_doc(doc)) for doc in data.values()]
        vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False)

        pyLDAvis.save_html(vis, f'{output_file}')

    def create_comparison(self) -> pd.DataFrame:
        """
        Train the LDA model(s) and find the dominant topic of each company in both years.
        Only companies present in both years are compared. Also saves the pyLDAvis visualizations.

        :return: DataFrame with columns company, previous_dominant_topic and current_dominant_topic.
            A topic is empty (NaN) when the company's document had no words left in the dictionary.
        """
        if self.run_separately:
            print("[LDA_PROCESS] Running LDA separately for previous and current year data...")
            prev_dictionary, prev_corpus = self._get_dictionary_and_corpus(self.prev_data)
            curr_dictionary, curr_corpus = self._get_dictionary_and_corpus(self.curr_data)
            prev_model = self._train_lda_model(prev_dictionary, prev_corpus)
            curr_model = self._train_lda_model(curr_dictionary, curr_corpus)
        else:
            print("[LDA_PROCESS] Running LDA on combined data...")
            dictionary, corpus = self._get_dictionary_and_corpus()
            model = self._train_lda_model(dictionary, corpus)
            prev_dictionary = curr_dictionary = dictionary
            prev_model = curr_model = model

        results = []

        for company in list(set(self.prev_data) & set(self.curr_data)):
            prev_words = self._prepare_doc(self.prev_data.get(company, []))
            curr_words = self._prepare_doc(self.curr_data.get(company, []))

            prev_topic = self._get_dominant_topic(prev_words, prev_dictionary, prev_model)
            curr_topic = self._get_dominant_topic(curr_words, curr_dictionary, curr_model)

            results.append({
                "company": company,
                "previous_dominant_topic": prev_topic,
                "current_dominant_topic": curr_topic
            })
        results_df = pd.DataFrame(results)

        self._export_vis_data("process_outputs/prev_lda_visualization.html", prev_dictionary, prev_model, self.prev_data)
        self._export_vis_data("process_outputs/curr_lda_visualization.html", curr_dictionary, curr_model, self.curr_data)

        return results_df
