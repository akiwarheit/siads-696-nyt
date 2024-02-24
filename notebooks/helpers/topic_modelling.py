import pickle
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import make_scorer

with open('out/count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Dictionary can accept an iterable of iterable str.
#
# documents : iterable of iterable of str, optional
#     Documents to be used to initialize the mapping and collect corpus statistics.
# prune_at : int, optional
#     Dictionary will try to keep no more than `prune_at` words in its mapping, to limit its RAM
#     footprint, the correctness is not guaranteed.
#     Use :meth:`~gensim.corpora.dictionary.Dictionary.filter_extremes` to perform proper filtering.

# Examples
# --------
# .. sourcecode:: pycon

#     >>> from gensim.corpora import Dictionary
#     >>>
#     >>> texts = [['human', 'interface', 'computer']]
#     >>> dct = Dictionary(texts)  # initialize a Dictionary
#     >>> dct.add_documents([["cat", "say", "meow"], ["dog"]])  # add more document (extend the vocabulary)
#     >>> dct.doc2bow(["dog", "computer", "non_existent_word"])
#     [(0, 1), (6, 1)]
def calc_coherence(words, texts):
    dictionary = Dictionary(texts)

    chmodel = CoherenceModel(
        topics=words, texts=texts, dictionary=dictionary, coherence='c_v'
    )

    return chmodel.get_coherence()

def get_topic_words(model, num_top_words=50):
    feature_names = vectorizer.get_feature_names_out()

    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)

    return topic_words

def create_coherence_scorer(dictionary):
    def coherence_score(model, X, y, **kwargs):
        topic_words = get_topic_words(model)
        coherence_score = calc_coherence(topic_words, [dictionary])

        return coherence_score

    return make_scorer(coherence_score, greater_is_better=True)