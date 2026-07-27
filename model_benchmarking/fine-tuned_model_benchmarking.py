import json
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    coverage_error,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartConfig,
    BartModel,
    BartPreTrainedModel,
    T5Config,
    T5EncoderModel,
    T5PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# =====================================================
# CONFIG
# =====================================================

# Multiple models
MODEL_PATHS = [
    "bert_21_final_model",
    "bert_42_final_model",
    "bert_123_final_model",
    "deberta_21_final_model",
    "deberta_42_final_model",
    "deberta_123_final_model",
    "distilbert_21_final_model",
    "distilbert_42_final_model",
    "distilbert_123_final_model",
    "electra_21_final_model",
    "electra_42_final_model",
    "electra_123_final_model",
    "roberta_21_final_model",
    "roberta_42_final_model",
    "roberta_123_final_model",
    "bart_21_final_model",
    "bart_42_final_model",
    "bart_123_final_model",
    "t5_21_final_model",
    "t5_42_final_model",
    "t5_123_final_model",
]

# Three test datasets
TEST_FILES = [
    "../ReliaBench/test/test_in_domain_gold.jsonl",
    "../ReliaBench/test/test_ood_NLP_specific_gold.jsonl",
    "../ReliaBench/test/test_ood_manufacturing_specific_gold.jsonl",
]


NUM_LABELS = 6
MAX_LENGTH = 256
BATCH_SIZE = 32
THRESHOLD = 0.95

SAVE_RESULTS = True
RESULTS_DIR = "evaluation_results_v4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


class T5ForMultiLabelClassification(T5PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = 6

        self.encoder = T5EncoderModel(config)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):

        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (hidden * mask).sum(dim=1)

        pooled = pooled / (mask.sum(dim=1).clamp(min=1e-9))

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        loss = None

        if labels is not None:

            labels = labels.float()

            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class BartForMultiLabelClassification(BartPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = 6

        self.model = BartModel(config)

        self.dropout = nn.Dropout(
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.dropout
        )

        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (hidden * mask).sum(dim=1)

        pooled = pooled / (mask.sum(dim=1).clamp(min=1e-9))

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        loss = None

        if labels is not None:

            labels = labels.float()

            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def find_best_threshold(true_labels, probs, metric="micro_f1"):
    best_t = 0.5
    best_score = -1

    for t in np.arange(0.1, 0.9, 0.05):

        preds = (probs > t).astype(np.int32)

        score = f1_score(true_labels, preds, average="micro", zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


def load_model_and_tokenizer(model_path):

    adapter_config = os.path.join(model_path, "adapter_config.json")

    # -------------------------
    # LoRA models
    # -------------------------

    if os.path.exists(adapter_config):

        print("Detected LoRA adapter")

        peft_config = PeftConfig.from_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )

        model = PeftModel.from_pretrained(base_model, model_path)

    # -------------------------
    # T5 custom model
    # -------------------------

    elif "t5" in model_path.lower():

        print("Detected custom T5 model")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = T5Config.from_pretrained(model_path)

        config.num_labels = NUM_LABELS

        model = T5ForMultiLabelClassification.from_pretrained(model_path, config=config)

    # -------------------------
    # BART custom model
    # -------------------------

    elif "bart" in model_path.lower():

        print("Detected custom BART model")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = BartConfig.from_pretrained(model_path)

        config.num_labels = NUM_LABELS

        model = BartForMultiLabelClassification.from_pretrained(
            model_path, config=config
        )

    # -------------------------
    # Standard HF classifiers
    # -------------------------

    else:

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model = model.to(device)

    model.eval()

    return tokenizer, model


# =====================================================
# LOAD JSONL
# =====================================================


def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            ex = json.loads(line)

            if "text" in ex and "labels" in ex:
                data.append(ex)

    return data


for test_file in TEST_FILES:

    test_data = load_jsonl(test_file)

# =====================================================
# PREP DATA
# =====================================================


def prepare_dataset(test_data):

    texts = []
    true_labels = []

    for ex in test_data:

        labels = ex["labels"]

        if labels is None:
            print("Found sample with labels=None", ex)
            continue

        labels = [float(x) for x in labels]

        # ensure fixed size
        labels = labels[:NUM_LABELS] + [0.0] * max(0, NUM_LABELS - len(labels))

        # binarize
        labels = [1 if x >= 0.5 else 0 for x in labels]

        texts.append(ex["text"])
        true_labels.append(labels)

    true_labels = np.array(true_labels, dtype=np.int32)

    return texts, true_labels


texts, true_labels = prepare_dataset(test_data)

# =====================================================
# INFERENCE
# =====================================================


def run_inference(model, tokenizer, texts):

    all_probs = []

    with torch.no_grad():

        for i in range(0, len(texts), BATCH_SIZE):

            batch_texts = texts[i : i + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            outputs = model(**enc).logits

            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    return all_probs


# =====================================================
# METRICS
# =====================================================
def find_best_threshold_per_class(true_labels, probs):
    thresholds = []

    for i in range(true_labels.shape[1]):
        best_t = 0.5
        best_f1 = -1

        for t in np.arange(0.1, 0.9, 0.05):

            preds = (probs[:, i] > t).astype(int)

            f1 = f1_score(true_labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    return np.array(thresholds)


def compute_metrics(true_labels, pred_labels, probs):

    metrics = {}

    # -------------------------------------------------
    # F1 Scores
    # -------------------------------------------------

    metrics["micro_f1"] = f1_score(
        true_labels, pred_labels, average="micro", zero_division=0
    )

    metrics["macro_f1"] = f1_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    metrics["weighted_f1"] = f1_score(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    metrics["samples_f1"] = f1_score(
        true_labels, pred_labels, average="samples", zero_division=0
    )

    # -------------------------------------------------
    # Precision
    # -------------------------------------------------

    metrics["micro_precision"] = precision_score(
        true_labels, pred_labels, average="micro", zero_division=0
    )

    metrics["macro_precision"] = precision_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    # -------------------------------------------------
    # Recall
    # -------------------------------------------------

    metrics["micro_recall"] = recall_score(
        true_labels, pred_labels, average="micro", zero_division=0
    )

    metrics["macro_recall"] = recall_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    # -------------------------------------------------
    # Multi-label metrics
    # -------------------------------------------------

    metrics["subset_accuracy"] = accuracy_score(true_labels, pred_labels)

    metrics["hamming_loss"] = hamming_loss(true_labels, pred_labels)

    metrics["jaccard_micro"] = jaccard_score(
        true_labels, pred_labels, average="micro", zero_division=0
    )

    metrics["jaccard_macro"] = jaccard_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    # -------------------------------------------------
    # Probability / ranking metrics
    # -------------------------------------------------

    try:
        metrics["roc_auc_micro"] = roc_auc_score(true_labels, probs, average="micro")
    except:
        metrics["roc_auc_micro"] = np.nan

    try:
        metrics["roc_auc_macro"] = roc_auc_score(true_labels, probs, average="macro")
    except:
        metrics["roc_auc_macro"] = np.nan

    try:
        metrics["avg_precision_micro"] = average_precision_score(
            true_labels, probs, average="micro"
        )
    except:
        metrics["avg_precision_micro"] = np.nan

    try:
        metrics["avg_precision_macro"] = average_precision_score(
            true_labels, probs, average="macro"
        )
    except:
        metrics["avg_precision_macro"] = np.nan

    try:
        metrics["coverage_error"] = coverage_error(true_labels, probs)
    except:
        metrics["coverage_error"] = np.nan

    try:
        metrics["label_ranking_loss"] = label_ranking_loss(true_labels, probs)
    except:
        metrics["label_ranking_loss"] = np.nan

    try:
        metrics["lrap"] = label_ranking_average_precision_score(true_labels, probs)
    except:
        metrics["lrap"] = np.nan

        # -------------------------------------------------
    # Per-class F1
    # -------------------------------------------------

    metrics["f1_per_class"] = f1_score(
        true_labels, pred_labels, average=None, zero_division=0
    ).tolist()

    LABELS = [
        "State Deviation",
        "Sequential",
        "Maintenance",
        "Operational",
        "Structural",
        "Temporal and Probabilistic",
    ]

    per_class = metrics["f1_per_class"]

    best_idx = int(np.argmax(per_class))
    worst_idx = int(np.argmin(per_class))

    metrics["best_class"] = LABELS[best_idx]
    metrics["best_class_f1"] = per_class[best_idx]

    metrics["worst_class"] = LABELS[worst_idx]
    metrics["worst_class_f1"] = per_class[worst_idx]

    # -------------------------------------------------
    # Confusion matrix (per label)
    # -------------------------------------------------

    try:
        cm = multilabel_confusion_matrix(true_labels, pred_labels)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception:
        metrics["confusion_matrix"] = None

    return metrics


# =====================================================
# STORE ALL RESULTS
# =====================================================

dataset_results = {}

# =====================================================
# MAIN EVALUATION LOOP
# =====================================================

for model_path in MODEL_PATHS:

    print("\n" + "=" * 80)
    print(f"LOADING MODEL: {model_path}")
    print("=" * 80)

    # -------------------------------------------------
    # Load tokenizer + model
    # -------------------------------------------------

    tokenizer, model = load_model_and_tokenizer(model_path)

    # -------------------------------------------------
    # Evaluate on all datasets
    # -------------------------------------------------

    for test_file in TEST_FILES:

        dataset_name = os.path.splitext(os.path.basename(test_file))[0]

        print("\n" + "-" * 80)
        print(f"TEST DATASET: {dataset_name}")
        print("-" * 80)

        # =============================================
        # Load dataset
        # =============================================

        test_data = load_jsonl(test_file)

        texts, true_labels = prepare_dataset(test_data)

        # =============================================
        # Inference
        # =============================================

        probs = run_inference(model, tokenizer, texts)

        best_t, best_f1 = find_best_threshold(true_labels, probs)

        print(f"\nBest threshold: {best_t:.2f} | Best micro-F1: {best_f1:.4f}")

        # =============================================
        # Threshold tuning (per class)
        # =============================================

        thresholds = find_best_threshold_per_class(true_labels, probs)

        print("\nBest thresholds per class:", thresholds)

        pred_labels = (probs > thresholds).astype(np.int32)

        # =============================================
        # Metrics
        # =============================================

        metrics = compute_metrics(
            true_labels=true_labels, pred_labels=pred_labels, probs=probs
        )

        # =============================================
        # Print metrics
        # =============================================

        print("\n===== RESULTS =====")

        for k, v in metrics.items():

            if isinstance(v, float):
                print(f"{k:30s}: {v:.4f}")
            elif isinstance(v, list) and k == "f1_per_class":
                print(f"{k:30s}: {[round(x, 4) for x in v]}")
            else:
                print(
                    f"{k:30s}: (array/list, shape={len(v) if hasattr(v,'__len__') else 'unknown'})"
                )

        # =============================================
        # Classification report
        # =============================================

        print("\n===== CLASSWISE REPORT =====")

        print(classification_report(true_labels, pred_labels, zero_division=0))

        # =============================================
        # Create result row
        # =============================================

        result_row = {"model_name": os.path.basename(model_path)}

        # Save each class F1 separately
        for i, score in enumerate(metrics["f1_per_class"]):
            result_row[f"class_{i}_f1"] = score

        # Add per-class probability scores
        for i in range(NUM_LABELS):
            result_row[f"class_{i}_score_mean"] = probs[:, i].mean()

        result_row.update(metrics)

        # =============================================
        # Store per dataset
        # =============================================

        if dataset_name not in dataset_results:
            dataset_results[dataset_name] = []

        dataset_results[dataset_name].append(result_row)

# =====================================================
# SAVE RESULTS
# =====================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

for dataset_name, rows in dataset_results.items():

    # -------------------------------------------------
    # Create dataframe
    # -------------------------------------------------

    df = pd.DataFrame(rows)

    # -------------------------------------------------
    # Reorder columns
    # -------------------------------------------------

    ordered_columns = [
        "model_name",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "samples_f1",
        "micro_precision",
        "macro_precision",
        "micro_recall",
        "macro_recall",
        "subset_accuracy",
        "hamming_loss",
        "jaccard_micro",
        "jaccard_macro",
        "roc_auc_micro",
        "roc_auc_macro",
        "avg_precision_micro",
        "avg_precision_macro",
        "coverage_error",
        "label_ranking_loss",
        "lrap",
        # per-class probabilities
        "class_0_score_mean",
        "class_1_score_mean",
        "class_2_score_mean",
        "class_3_score_mean",
        "class_4_score_mean",
        "class_5_score_mean",
        # per-class F1
        "class_0_f1",
        "class_1_f1",
        "class_2_f1",
        "class_3_f1",
        "class_4_f1",
        "class_5_f1",
        # summary
        "best_class",
        "best_class_f1",
        "worst_class",
        "worst_class_f1",
    ]

    df = df[ordered_columns]

    # -------------------------------------------------
    # Round numeric values
    # -------------------------------------------------

    numeric_cols = df.columns.drop("model_name")

    df[numeric_cols] = df[numeric_cols].round(4)

    # -------------------------------------------------
    # Print table
    # -------------------------------------------------

    print("\n")
    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    print(df.to_string(index=False))

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------

    if SAVE_RESULTS:

        csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results_v4.csv")

        df.to_csv(csv_path, index=False)

        print(f"\nSaved CSV: {csv_path}")


import pandas as pd


def print_mistakes_only(texts, true_labels, pred_labels, probs, max_samples=50):
    rows = []

    for i in range(len(texts)):
        if not (true_labels[i] == pred_labels[i]).all():
            rows.append(
                {
                    "text": texts[i][:200],
                    "true": true_labels[i].tolist(),
                    "pred": pred_labels[i].tolist(),
                    "probs": [round(float(x), 3) for x in probs[i]],
                }
            )

        if len(rows) >= max_samples:
            break

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# print_mistakes_only(texts, true_labels, pred_labels, probs)

print("\nDONE.")
