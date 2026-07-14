import evaluate
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from data import load_data, train_test_split_data
from config import FILE_PATH, CHECKPOINT, TRANSFORMER_TOKENIZER_PATH


df = load_data(FILE_PATH)
X_test, X_test, y_train, y_test = train_test_split_data(df, text_col="Clean_Text",
                                             label_col="Label")

tokenizer_path = TRANSFORMER_TOKENIZER_PATH
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def tokenizer(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

def build_train_test_dataset(X_train, X_test, y_train, y_test, tokenizer):
    
    train_dataset = Dataset.from_dict({
        "text": X_train.tolist(),
        "label": y_train.tolist(),
    })

    test_dataset = Dataset.from_dict({
        "text": X_test.tolist(),
        "label": y_test.tolist(),
    })

    train_dataset = train_dataset.map(tokenizer, batched=True)
    test_dataset = test_dataset.map(tokenizer, batched=True)


    train_dataset.set_fromat(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    test_dataset.set_fromat(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return train_dataset, test_dataset


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }
    