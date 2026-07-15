import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from config import CHECKPOINT, TRANSFORMER_MODEL_PATH, TRANSFORMER_TOKENIZER_PATH, FILE_PATH
from data import load_data, train_test_split_data
from transformers_utils import build_train_test_dataset, compute_metrics, tokenizer

model_path = TRANSFORMER_MODEL_PATH
tokenizer_path = TRANSFORMER_TOKENIZER_PATH

model = AutoModelForSequenceClassification.from_pretrained(model_path)

df = load_data(FILE_PATH)
X_train, X_test, y_train, y_test = train_test_split_data(df, text_col="Clean_Text",
                                                        lable_col="Label")

train_dataset, test_dataset = build_train_test_dataset(X_train, X_test, y_train, y_test, tokenizer)

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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    params = {
        "checkpoint": CHECKPOINT,
        "model_name": model_path,
        "tokenizer_name": tokenizer_path,
    }

    for k, v in params.items():
        mlflow.log_param(k, str(v))
    
    for k, v in metrics.items():
        mlflow.log_metrics(k, float(v))

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="transformer_model"
    )

    print(metrics)