import json
import random

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

SEED = 123  # change for different runs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# =====================================================
# CUDA INFO
# =====================================================

print("cuda:", torch.cuda.is_available())
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)


# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "microsoft/deberta-v3-base"

NUM_LABELS = 6

MAX_LENGTH = 256

STRIDE = 64

# =====================================================
# LOAD DATA
# =====================================================


def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:

        for lineno, line in enumerate(f, start=1):

            try:
                ex = json.loads(line)

            except json.JSONDecodeError as e:

                print(f"\nERROR IN {path}")
                print(f"Line number: {lineno}")
                print(f"Error: {e}")

                print("\nBeginning of line:")
                print(repr(line[:500]))

                raise

            data.append(ex)

    return data


train_raw = load_jsonl("../ReliaBench/training/training_snorkel_and_gold.jsonl")

val_raw = load_jsonl("../ReliaBench/validation/validation_gold.jsonl")

test_raw = load_jsonl("../ReliaBench/test/test_ood_manufacturing_specific_gold.jsonl")

print("\nRAW SIZES:")
print(len(train_raw))
print(len(val_raw))
print(len(test_raw))

# =====================================================
# SAFE DATASET BUILDER
# =====================================================


def build_dataset(raw):

    data = []

    dropped = 0

    for ex in raw:

        labels = ex.get("labels", None)

        if labels is None:

            dropped += 1
            continue

        if not isinstance(labels, list):

            dropped += 1
            continue

        try:

            labels = [float(x) for x in labels]

        except:

            dropped += 1
            continue

        # force fixed length
        if len(labels) < NUM_LABELS:

            labels = labels + ([0.0] * (NUM_LABELS - len(labels)))

        elif len(labels) > NUM_LABELS:

            labels = labels[:NUM_LABELS]

        # force binary
        labels = [1.0 if x >= 0.5 else 0.0 for x in labels]

        text = ex.get("text", None)

        if text is None:

            dropped += 1
            continue

        data.append({"text": text, "labels": labels})

    print(f"Built dataset: {len(data)} " f"(dropped {dropped})")

    return Dataset.from_list(data)


# =====================================================
# BUILD DATASETS
# =====================================================

train_ds = build_dataset(train_raw)

val_ds = build_dataset(val_raw)

test_ds = build_dataset(test_raw)

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =====================================================
# PREPROCESS
# =====================================================


def preprocess(examples):

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    labels = [np.array(examples["labels"][i], dtype=np.float32) for i in sample_mapping]

    tokenized["labels"] = labels

    return tokenized


# =====================================================
# MAP DATASETS
# =====================================================


def map_ds(ds):

    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    return ds


train_ds = map_ds(train_ds)

val_ds = map_ds(val_ds)

test_ds = map_ds(test_ds)

print("\nFINAL SIZES:")

print(len(train_ds))
print(len(val_ds))
print(len(test_ds))

# =====================================================
# SANITY CHECK
# =====================================================

print("\nSAMPLE LABEL:")

print(train_ds[0]["labels"])

print(type(train_ds[0]["labels"]))

# =====================================================
# CUSTOM COLLATOR
# =====================================================


class MultiLabelCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):

        labels = torch.tensor([f["labels"] for f in features], dtype=torch.float32)

        # remove labels before padding
        texts = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(texts, padding=True, return_tensors="pt")

        batch["labels"] = labels

        return batch


data_collator = MultiLabelCollator(tokenizer)

# =====================================================
# MODEL
# =====================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# =====================================================
# METRICS
# =====================================================


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    preds = (probs > 0.5).astype(np.int32)

    labels = np.array(labels)

    labels = (labels >= 0.5).astype(np.int32)

    labels = labels.reshape(-1, NUM_LABELS)

    preds = preds.reshape(-1, NUM_LABELS)

    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


# =====================================================
# TRAINING ARGS
# =====================================================

args = TrainingArguments(
    output_dir="./deberta_123",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    remove_unused_columns=False,
    logging_steps=50,
    seed=SEED,
    data_seed=SEED,
)

# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2, early_stopping_threshold=0.0005
        )
    ],
)

# =====================================================
# VERIFY COLLATOR
# =====================================================

sample = data_collator([train_ds[0], train_ds[1]])

print("\nLABEL DTYPE:")

print(sample["labels"].dtype)

print("\nINPUT SHAPE:")

print(sample["input_ids"].shape)

# =====================================================
# TRAIN
# =====================================================
trainer.train()

# =====================================================
# THRESHOLD TUNING
# =====================================================

print("\nTuning thresholds...")

val_out = trainer.predict(val_ds)

val_probs = torch.sigmoid(torch.tensor(val_out.predictions)).numpy()

val_labels = np.array(val_out.label_ids)

val_labels = (val_labels >= 0.5).astype(np.int32)

val_labels = val_labels.reshape(-1, NUM_LABELS)

thresholds = np.zeros(NUM_LABELS)

for i in range(NUM_LABELS):

    best_t = 0.5

    best_f1 = 0.0

    for t in np.linspace(0.1, 0.9, 17):

        preds = (val_probs[:, i] > t).astype(np.int32)

        f1 = f1_score(val_labels[:, i], preds, zero_division=0)

        if f1 > best_f1:

            best_f1 = f1

            best_t = t

    thresholds[i] = best_t

print("\nOptimal thresholds per class:")

for i, t in enumerate(thresholds):
    print(f"Class {i}: " f"threshold={t:.3f}")

avg_threshold = np.mean(thresholds)

print(f"\nAverage optimal threshold: " f"{avg_threshold:.3f}")

# =====================================================
# EVALUATION
# =====================================================


def evaluate(ds):

    out = trainer.predict(ds)

    logits = out.predictions

    y_true = np.array(out.label_ids)

    y_true = (y_true >= 0.5).astype(np.int32)

    y_true = y_true.reshape(-1, NUM_LABELS)

    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    y_pred = (probs > thresholds.reshape(1, -1)).astype(np.int32)

    y_pred = y_pred.reshape(-1, NUM_LABELS)

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


# =====================================================
# FINAL RESULTS
# =====================================================

print("\nVAL METRICS:")

print(evaluate(val_ds))

print("\nTEST METRICS:")

print(evaluate(test_ds))

# =====================================================
# SAVE
# =====================================================

SAVE_DIR = "./deberta_123_final_model"

model.save_pretrained(SAVE_DIR)

tokenizer.save_pretrained(SAVE_DIR)

print("\nModel saved successfully.")
