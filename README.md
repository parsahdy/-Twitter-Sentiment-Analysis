Sentiment Analysis Project

Project Overview
This project explores sentiment classification on social media text using a progressive modeling pipeline, starting with traditional machine learning baselines and moving toward deep learning and transformer-based approaches. The goal is to compare methods across complexity, representation quality, and practical performance while keeping the workflow reproducible and suitable for experiment tracking.

Dataset
The project is built around a labeled sentiment dataset of cleaned text samples and target labels. The data is split into training and test sets and used consistently across baseline, deep learning, and transformer workflows to support fair comparison between approaches.

EDA
Exploratory data analysis focuses on understanding class balance, text length distribution, vocabulary characteristics, and common linguistic patterns in the cleaned corpus. This stage helps identify data quality issues, informs preprocessing choices, and provides context for model behavior.

Preprocessing
The preprocessing pipeline includes text cleaning, normalization, and preparation for different modeling families. Traditional baselines use TF-IDF features, deep learning models rely on tokenized and sequence-based inputs, and transformer workflows use pretrained tokenizers aligned with the selected checkpoint.

Baseline Models
Baseline experiments are implemented with classical machine learning methods to establish a strong and interpretable reference point. Current baseline training scripts use TF-IDF vectorization and linear classifiers, making it possible to quickly benchmark performance before moving to more computationally intensive architectures.

Word Embeddings
Word embedding experiments are designed to represent semantic relationships between tokens more effectively than sparse vectorization alone. These embeddings support downstream neural architectures and provide a foundation for sequence-based sentiment modeling.

Deep Learning
The deep learning stage focuses on neural sequence models such as LSTM-based architectures for sentiment classification. This part of the project is intended to capture contextual patterns in text beyond what can be modeled with traditional feature engineering.

Transformers (Training in Progress)
Fine-tuning scripts implemented; training and evaluation pending due to limited GPU resources.

Results
Results are organized to compare baseline, embedding-based, deep learning, and transformer-oriented approaches using consistent evaluation metrics such as accuracy, precision, recall, and F1 score. As experiments mature, this section can be expanded with model comparisons, error analysis, and experiment tracking outputs.

Installation
Clone the repository, create a virtual environment, and install the required dependencies.


bash
git clone <https://github.com/parsahdy/-Twitter-Sentiment-Analysis.git>
cd <SENTIMENT140>
python -m venv .venv
script\.venv\activate
pip install -r requirements.txt
For TensorFlow, PyTorch, and transformer training, dependency versions should match the project configuration used during development.

Usage
Run each training workflow independently depending on the experiment type.


bash
python train_baseline.py
python train_deeplearning.py
python train_transformers.py
If MLflow is enabled in the environment, experiment runs, metrics, and model artifacts can be tracked and reviewed through the MLflow UI.

Future Work
Future development will focus on completing transformer fine-tuning, improving experiment tracking consistency, expanding hyperparameter search, and strengthening reproducibility across environments. Additional work may include better embedding strategies, richer evaluation reports, deployment-oriented packaging, and optimization for limited hardware settings.