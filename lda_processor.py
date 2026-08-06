import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models

class LDAProcessor:
    def __init__(self, curr_data:dict[str, list[str]], prev_data:dict[str, list[str]], num_topics:int) -> None:
        self.curr_data = curr_data
        self.prev_data = prev_data
        self.num_topics = num_topics
    def _get_dictionary_and_corpus(self):
        self.all_docs = list(self.curr_data.values()) + list(self.prev_data.values())
        self.all_companies = list(self.curr_data.keys()) + list(self.prev_data.keys())
        self.dictionary = corpora.Dictionary(self.all_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.all_docs]

    def _train_lda_model(self):
        self.lda_model = LdaModel(corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary)

    def _get_dominant_topic(self, doc:list[str]) -> int:
        if not doc:
            return None
        bow = self.dictionary.doc2bow(doc)
        topic_distribution = self.lda_model.get_document_topics(bow)
        dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0][0]
        return dominant_topic

    def _export_vis_data(self, output_file:str) -> None:
        prev_corpus = [self.dictionary.doc2bow(doc) for doc in self.prev_data.values()]
        curr_corpus = [self.dictionary.doc2bow(doc) for doc in self.curr_data.values()]

        prev_vis = pyLDAvis.gensim_models.prepare(self.lda_model, prev_corpus, self.dictionary, sort_topics=False)
        curr_vis = pyLDAvis.gensim_models.prepare(self.lda_model, curr_corpus, self.dictionary, sort_topics=False)

        pyLDAvis.save_html(prev_vis, f'prev_{output_file}')
        pyLDAvis.save_html(curr_vis, f'curr_{output_file}')

    def create_comparison(self) -> pd.DataFrame:
        self._get_dictionary_and_corpus()
        self._train_lda_model()

        results = []

        for company in self.all_companies:
            prev_words = self.prev_data.get(company, [])
            curr_words = self.curr_data.get(company, [])

            prev_topic = self._get_dominant_topic(prev_words)
            curr_topic = self._get_dominant_topic(curr_words)

            results.append({
                "company": company,
                "previous_dominant_topic": prev_topic,
                "current_dominant_topic": curr_topic
            })
        results_df = pd.DataFrame(results)

        self._export_vis_data(output_file="lda_visualization.html")

        return results_df

