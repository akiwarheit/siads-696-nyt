import pickle
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import make_scorer
from gensim.test.utils import common_dictionary, common_texts

def calc_coherence(topics, texts, dictionary):
    dictionary = Dictionary(dictionary)

    chmodel = CoherenceModel(
        topics=topics, texts=texts, dictionary=dictionary, coherence='c_v'
    )

    return chmodel.get_coherence()

def get_topic_words(model, vectorizer, num_top_words=50):
    feature_names = vectorizer.get_feature_names_out()

    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)

    return topic_words

def create_coherence_scorer(dictionary, vectorizer):
    def coherence_score(model, X, y, **kwargs):
        topic_words = get_topic_words(model, vectorizer)
        coherence_score = calc_coherence(topic_words, [dictionary])

        return coherence_score

    return make_scorer(coherence_score, greater_is_better=True)