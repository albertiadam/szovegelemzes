import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models

class LDAProcessor:
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
        self.curr_data = curr_data
        self.prev_data = prev_data
        self.num_topics = num_topics
        self.run_separately = run_separately
        self.use_bigrams = use_bigrams
        self.num_of_passes = num_of_passes
        self.num_of_iterations = num_of_iterations
        self.random_state = random_state

    def _prepare_doc(self, doc: list[str]) -> list[str]:
        if not self.use_bigrams:
            return doc

        bigrams = [
            f"{doc[index]} {doc[index + 1]}"
            for index in range(len(doc) - 1)
        ]

        return bigrams

    def _get_dictionary_and_corpus(self, data=None):
        raw_docs = list(data.values()) if data is not None else list(self.curr_data.values()) + list(self.prev_data.values())
        docs = [self._prepare_doc(doc) for doc in raw_docs]
        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=None)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        return dictionary, corpus

    def _train_lda_model(self, dictionary, corpus):
        return LdaModel(
            corpus=corpus,
            num_topics=self.num_topics,
            id2word=dictionary,
            passes=self.num_of_passes,
            iterations=self.num_of_iterations,
            random_state=self.random_state
        )

    def _get_dominant_topic(self, doc: list[str], dictionary, model) -> int:
        if not doc:
            return None
        bow = dictionary.doc2bow(doc)
        if not bow:
            return None
        topic_distribution = model.get_document_topics(bow)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        return dominant_topic

    def _export_vis_data(self, output_file: str, dictionary: corpora.Dictionary, model: LdaModel, data: dict[str, list[str]]) -> None:
        corpus = [dictionary.doc2bow(self._prepare_doc(doc)) for doc in data.values()]
        vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False)

        pyLDAvis.save_html(vis, f'{output_file}')

    def create_comparison(self) -> pd.DataFrame:
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

