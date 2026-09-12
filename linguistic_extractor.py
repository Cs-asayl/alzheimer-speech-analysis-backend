import math
import re
from collections import Counter

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# Load NLP models once
nlp = spacy.load("en_core_web_sm")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


LINGUISTIC_FEATURE_NAMES = [
    "pos_noun_count",
    "pos_verb_count",
    "pos_adj_count",
    "pos_pron_count",
    "pos_adv_count",
    "word_count",
    "unique_words",
    "avg_word_length",
    "ttr",
    "brunets_index",
    "honores_statistic",
    "avg_sentence_length",
    "avg_dependency_tree_depth",
    "max_dependency_tree_depth",
    "semantic_similarity",
    "tfidf_mean",
    "mean_length_of_utterance",
    "word_repetition_ratio",
    "hesitation_count",
]


def _tree_depth(token):
    children = list(token.children)

    if not children:
        return 1

    return 1 + max(_tree_depth(child) for child in children)


def extract_linguistic_features(text):
    text = text.strip()

    if not text:
        return {
            name: 0.0
            for name in LINGUISTIC_FEATURE_NAMES
        }

    doc = nlp(text)

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]

    word_count = len(words)
    unique_words = len(set(words))

    # -------------------------
    # POS FEATURES
    # -------------------------

    pos_noun_count = sum(
        1 for token in doc
        if token.pos_ in ["NOUN", "PROPN"]
    )

    pos_verb_count = sum(
        1 for token in doc
        if token.pos_ in ["VERB", "AUX"]
    )

    pos_adj_count = sum(
        1 for token in doc
        if token.pos_ == "ADJ"
    )

    pos_pron_count = sum(
        1 for token in doc
        if token.pos_ == "PRON"
    )

    pos_adv_count = sum(
        1 for token in doc
        if token.pos_ == "ADV"
    )

    # -------------------------
    # LEXICAL FEATURES
    # -------------------------

    if word_count > 0:
        avg_word_length = np.mean(
            [len(word) for word in words]
        )

        ttr = unique_words / word_count
    else:
        avg_word_length = 0.0
        ttr = 0.0

    # Brunet's Index
    if word_count > 0 and unique_words > 0:
        brunets_index = word_count ** (
            unique_words ** -0.165
        )
    else:
        brunets_index = 0.0

    # Honore's Statistic
    if word_count > 0 and unique_words > 0:
        frequencies = Counter(words)

        hapax = sum(
            1 for count in frequencies.values()
            if count == 1
        )

        denominator = 1 - (
            hapax / unique_words
        )

        if denominator > 0:
            honores_statistic = (
                100
                * math.log(word_count)
                / denominator
            )
        else:
            honores_statistic = 0.0
    else:
        honores_statistic = 0.0

    # -------------------------
    # SENTENCE FEATURES
    # -------------------------

    sentences = list(doc.sents)

    sentence_lengths = [
        len([
            token
            for token in sentence
            if token.is_alpha
        ])
        for sentence in sentences
    ]

    if sentence_lengths:
        avg_sentence_length = float(
            np.mean(sentence_lengths)
        )
    else:
        avg_sentence_length = 0.0

    # Dependency tree depth
    depths = []

    for sentence in sentences:
        roots = [
            token
            for token in sentence
            if token.head == token
        ]

        for root in roots:
            depths.append(
                _tree_depth(root)
            )

    if depths:
        avg_dependency_tree_depth = float(
            np.mean(depths)
        )

        max_dependency_tree_depth = float(
            max(depths)
        )
    else:
        avg_dependency_tree_depth = 0.0
        max_dependency_tree_depth = 0.0

    # -------------------------
    # SEMANTIC SIMILARITY
    # -------------------------

    sentence_texts = [
        sentence.text.strip()
        for sentence in sentences
        if sentence.text.strip()
    ]

    if len(sentence_texts) >= 2:

        embeddings = semantic_model.encode(
            sentence_texts,
            normalize_embeddings=True
        )

        similarities = []

        for i in range(
            len(embeddings) - 1
        ):
            similarity = float(
                np.dot(
                    embeddings[i],
                    embeddings[i + 1]
                )
            )

            similarities.append(
                similarity
            )

        semantic_similarity = float(
            np.mean(similarities)
        )

    else:
        semantic_similarity = 0.0

    # -------------------------
    # TF-IDF
    # -------------------------

    if words:
        try:
            vectorizer = TfidfVectorizer()

            matrix = vectorizer.fit_transform(
                [text]
            )

            tfidf_mean = float(
                matrix.mean()
            )

        except ValueError:
            tfidf_mean = 0.0

    else:
        tfidf_mean = 0.0

    # -------------------------
    # MEAN LENGTH OF UTTERANCE
    # -------------------------

    if sentence_lengths:
        mean_length_of_utterance = float(
            np.mean(sentence_lengths)
        )
    else:
        mean_length_of_utterance = 0.0

    # -------------------------
    # WORD REPETITION
    # -------------------------

    if word_count > 0:
        frequencies = Counter(words)

        repeated_words = sum(
            count - 1
            for count in frequencies.values()
            if count > 1
        )

        word_repetition_ratio = (
            repeated_words / word_count
        )
    else:
        word_repetition_ratio = 0.0

    # -------------------------
    # HESITATION COUNT
    # -------------------------

    hesitation_pattern = (
        r"\b(um+|uh+|erm+|hmm+|mm+)\b"
    )

    hesitation_count = len(
        re.findall(
            hesitation_pattern,
            text.lower()
        )
    )

    features = {
        "pos_noun_count": float(pos_noun_count),
        "pos_verb_count": float(pos_verb_count),
        "pos_adj_count": float(pos_adj_count),
        "pos_pron_count": float(pos_pron_count),
        "pos_adv_count": float(pos_adv_count),
        "word_count": float(word_count),
        "unique_words": float(unique_words),
        "avg_word_length": float(avg_word_length),
        "ttr": float(ttr),
        "brunets_index": float(brunets_index),
        "honores_statistic": float(honores_statistic),
        "avg_sentence_length": float(avg_sentence_length),
        "avg_dependency_tree_depth": float(
            avg_dependency_tree_depth
        ),
        "max_dependency_tree_depth": float(
            max_dependency_tree_depth
        ),
        "semantic_similarity": float(
            semantic_similarity
        ),
        "tfidf_mean": float(tfidf_mean),
        "mean_length_of_utterance": float(
            mean_length_of_utterance
        ),
        "word_repetition_ratio": float(
            word_repetition_ratio
        ),
        "hesitation_count": float(
            hesitation_count
        ),
    }

    return features


if __name__ == "__main__":

    test_text = (
        "The picture shows a family in a kitchen. "
        "The mother is washing dishes while the "
        "children are taking cookies from a jar. "
        "Um, the water is overflowing from the sink."
    )

    features = extract_linguistic_features(
        test_text
    )

    print("\n------------------------------")
    print("LINGUISTIC FEATURES")
    print("------------------------------")

    for name in LINGUISTIC_FEATURE_NAMES:
        print(
            f"{name}: {features[name]}"
        )

    print("------------------------------")
    print(
        "Total linguistic features:",
        len(features)
    )