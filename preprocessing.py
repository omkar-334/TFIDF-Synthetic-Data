import asyncio
import re
import string

import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def get_common_words():
    common_words = set(stopwords.words("english"))

    verbs = {"let", "see", "need", "make", "want", "look", "like", "try", "come", "take", "use", "find", "get", "give", "tell", "work", "call", "ask"}
    for verb in verbs:
        common_words.update([verb, verb + "s", verb + "ed", verb + "ing"])

    all_words = set(words.words())
    common_words.update([w for w in all_words if len(w) <= 3])

    fd = nltk.FreqDist(w.lower() for w in words.words())
    common_words.update([w for w, _ in fd.most_common(1000)])
    return common_words


common_words = get_common_words()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"['“”’]", "", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation and token not in common_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = set([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens])

    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text


def calculate_tfidf(documents, ngram_range=(1, 5), top_n=1000):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=top_n)

    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    feature_names = tfidf_vectorizer.get_feature_names_out()

    avg_tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()

    tfidf_scores = list(zip(feature_names, avg_tfidf_scores))

    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    return tfidf_scores[:top_n]


