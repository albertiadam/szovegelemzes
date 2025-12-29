import re
import os
import json
import multiprocessing


import pandas as pd
import numpy as np

from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS

import spacy

from nltk.stem import SnowballStemmer

import pyLDAvis
import pyLDAvis.gensim_models

NLTK_USE:bool = False # Set to True to use NLTK SnowballStemmer, False to use spaCy lemmatization
CACHE_FILE:str = 'processed_tokens_lemmed.jsonl'
worker: SnowballStemmer | spacy.language.Language = None

def __init__worker():
    global worker
    if NLTK_USE:
        worker = SnowballStemmer("english")
    else:
        worker = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        worker.max_length = 4_000_000

def clean(string:str) -> str:
    CHARS = ['[',']','(',')','{','}',"'",'"','.',',',':','\\','-','_',';']
    pattern = r'<([a-zA-Z]+)[^>]*>.*?</\1>|<[^>]+>|[\n\t\r]'
    cleaned = re.sub(pattern,'',string,flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\(.\)','',cleaned)
    cleaned = re.sub(r'\b\w+(?:\.\w+)+\b','',cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\d+', '', cleaned)
    for char in CHARS:
        cleaned = cleaned.replace(char,' ')
    splitted = cleaned.split(" ")
    words = []
    for word in splitted:
        words += word.split("/")
    words = [word.lower() for word in words if word not in STOPWORDS and len(word) not in (0,1)]
    return words

def read_one_file(path:str):
    with open(path,"r") as f:
        word_list = clean(f.read().strip())
    if NLTK_USE:
        word_list = [worker.stem(word) for word in word_list if len(worker.stem(word)) > 1 and worker.stem(word) not in STOPWORDS]
    else:
        word_list = [token.lemma_ for token in worker(" ".join(word_list)) if len(token.lemma_) > 1 and token.lemma_ not in STOPWORDS]
    return word_list

def create_cache_file(folder:str,workers:int=4):
    file_paths = [os.path.join(folder,f) for f in os.listdir(folder)]
    file_num = len(file_paths)
    if NLTK_USE:
        chunk_size = 50
    else:
        chunk_size = 10

    print("Fajlok szama: ",file_num)
    with multiprocessing.Pool(processes=workers,initializer=__init__worker) as pool:
        with open(CACHE_FILE,'w') as cache_f:
            for index, word_list in enumerate(pool.imap_unordered(read_one_file,file_paths,chunksize=chunk_size)):
                json.dump(word_list,cache_f)
                cache_f.write('\n')
                if (index+1) % 100 == 0 or (index+1) == file_num:
                    print(f"Feldolgozott fajlok: {index+1}/{file_num}")

def read_cached_tokens():
    with open(CACHE_FILE,'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())

if __name__ == "__main__":
    print("Szotar keszitese...")
    FOLDER = r"C:\Users\User\Downloads\10-X_C_2024\2024\QTR1"
    if not os.path.exists(CACHE_FILE):
        create_cache_file(FOLDER,workers=5)
    doc_stream = read_cached_tokens()
    dictionary = corpora.Dictionary(doc_stream)
    print(f"Szotar merete: {len(dictionary)}")
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in read_cached_tokens()]

    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=20,
        workers=5,
        passes=20,
        chunksize=200,
        iterations=100,
        random_state=42,
        alpha='symmetric',
        eta='auto',
        eval_every=None
    )
    vis_data = pyLDAvis.gensim_models.prepare(
        lda_model, 
        corpus, 
        dictionary, 
        sort_topics=False
    )

    pyLDAvis.save_html(vis_data, 'lda_eredmeny_vizualizacio2.html')