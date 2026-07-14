import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from config import CHECKPOINT, TRANSFORMER_MODEL_PATH, TRANSFORMER_TOKENIZER_PATH, FILE_PATH
from data import load_data, train_test_split_data
from transformers_utils import build_train_test_dataset, compute_metrics, tokenizer

model_path = TRANSFORMER_MODEL_PATH

model = AutoModelForSequenceClassification.from_pretrained(model_path)

df = load_data(FILE_PATH)
_, X_test, _, y_test = train_test_split_data(df)

train_dataset, test_dataset = build_train_test_dataset(X_test, y_test, tokenizer)

mlflow.set_experiment("sentiment140_transformers")

with mlflow.start_run():
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="../models/distilbert_flow",
            per_device_eval_batch_size=16,
            report_to="none"
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    params = {
        "model": model,
        "tokenizer": tokenizer,
        "checkpoint": CHECKPOINT,
    }

    mlflow.log_param(params)
    mlflow.log_metrics(metrics)

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="transformer_model"
    )

    print(metrics)