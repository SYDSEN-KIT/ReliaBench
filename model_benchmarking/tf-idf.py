import json
import pickle
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# =====================================================
# SEED
# =====================================================


SEED = 123


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


set_seed(SEED)


print("Seed:", SEED)


# =====================================================
# CONFIG
# =====================================================


NUM_LABELS = 6


# =====================================================
# FILE PATHS
# =====================================================


TRAIN_PATH = "../ReliaBench/training/training_snorkel_and_gold.jsonl"


VAL_PATH = "../ReliaBench/validation/validation_gold.jsonl"


TEST_PATH_1 = "../ReliaBench/test/test_ood_manufacturing_specific_gold.jsonl"


TEST_PATH_2 = "../ReliaBench/test/test_in_domain_gold.jsonl"


TEST_PATH_3 = "../ReliaBench/test/test_ood_NLP_specific_gold.jsonl"


# =====================================================
# LOAD JSONL
# =====================================================


def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:

        for lineno, line in enumerate(f, start=1):

            try:

                ex = json.loads(line)

            except json.JSONDecodeError as e:

                print("\nERROR IN FILE:")

                print(path)

                print("Line:", lineno)

                print(e)

                print(repr(line[:500]))

                raise

            data.append(ex)

    return data


# =====================================================
# LOAD RAW DATA
# =====================================================


train_raw = load_jsonl(TRAIN_PATH)


val_raw = load_jsonl(VAL_PATH)


test1_raw = load_jsonl(TEST_PATH_1)


test2_raw = load_jsonl(TEST_PATH_2)


test3_raw = load_jsonl(TEST_PATH_3)


print("\nRAW SIZES:")

print("Train:", len(train_raw))

print("Validation:", len(val_raw))

print("Test 1:", len(test1_raw))

print("Test 2:", len(test2_raw))

print("Test 3:", len(test3_raw))


# =====================================================
# DATA BUILDER
# =====================================================


def build_dataset(raw):

    texts = []

    labels = []

    dropped = 0

    for ex in raw:

        y = ex.get("labels", None)

        if y is None:

            dropped += 1

            continue

        if not isinstance(y, list):

            dropped += 1

            continue

        try:

            y = [float(x) for x in y]

        except:

            dropped += 1

            continue

        # force exactly 6 labels

        if len(y) < NUM_LABELS:

            y += [0.0] * (NUM_LABELS - len(y))

        elif len(y) > NUM_LABELS:

            y = y[:NUM_LABELS]

        # convert to binary

        y = [1 if x >= 0.5 else 0 for x in y]

        text = ex.get("text", None)

        if text is None:

            dropped += 1

            continue

        texts.append(text)

        labels.append(y)

    labels = np.array(labels, dtype=np.int32)

    print(f"Built dataset: {len(texts)} " f"(dropped {dropped})")

    return texts, labels


# =====================================================
# BUILD DATASETS
# =====================================================


train_texts, train_labels = build_dataset(train_raw)


val_texts, val_labels = build_dataset(val_raw)


test1_texts, test1_labels = build_dataset(test1_raw)


test2_texts, test2_labels = build_dataset(test2_raw)


test3_texts, test3_labels = build_dataset(test3_raw)


print("\nLABEL SHAPES:")

print("Train:", train_labels.shape)

print("Validation:", val_labels.shape)

print("Test 1:", test1_labels.shape)

print("Test 2:", test2_labels.shape)

print("Test 3:", test3_labels.shape)


# =====================================================
# MODEL
# =====================================================


model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                )
            ),
        ),
    ]
)


# =====================================================
# TRAIN
# =====================================================


print("\nTraining TF-IDF model...")


model.fit(train_texts, train_labels)


print("Training complete")


# =====================================================
# THRESHOLD TUNING
# =====================================================


print("\nTuning thresholds...")


val_probs = model.predict_proba(val_texts)


thresholds = np.zeros(NUM_LABELS)


for i in range(NUM_LABELS):

    best_f1 = 0

    best_t = 0.5

    for t in np.linspace(0.1, 0.9, 17):

        preds = (val_probs[:, i] > t).astype(np.int32)

        score = f1_score(val_labels[:, i], preds, zero_division=0)

        if score > best_f1:

            best_f1 = score

            best_t = t

    thresholds[i] = best_t


print("\nOptimal thresholds:")


for i, t in enumerate(thresholds):

    print(f"Class {i}: {t:.3f}")


print("\nAverage threshold:", thresholds.mean())


# =====================================================
# EVALUATION
# =====================================================
def evaluate(texts, labels):

    probs = model.predict_proba(texts)

    preds = (probs > thresholds.reshape(1, -1)).astype(np.int32)

    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(labels, preds),
    }


# =====================================================
# RESULTS
# =====================================================


print("\n==========================")

print("VALIDATION RESULTS")

print("==========================")

print(evaluate(val_texts, val_labels))


print("\n==========================")

print("TEST SET 1 RESULTS")

print("==========================")

print(evaluate(test1_texts, test1_labels))


print("\n==========================")

print("TEST SET 2 RESULTS")

print("==========================")

print(evaluate(test2_texts, test2_labels))


print("\n==========================")

print("TEST SET 3 RESULTS")

print("==========================")

print(evaluate(test3_texts, test3_labels))


# =====================================================
# SAVE MODEL
# =====================================================


SAVE_PATH = "./tfidf_multilabel_model.pkl"


with open(SAVE_PATH, "wb") as f:

    pickle.dump({"model": model, "thresholds": thresholds}, f)


print("\nModel saved:")

print(SAVE_PATH)
