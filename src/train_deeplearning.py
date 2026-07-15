import mlflow
import mlflow.tensorflow
import tensorflow as tf

from config import TENSOR_FILE_PATH, FILE_PATH
from deeplearning_utils import deep_learning_data
from data import load_data, train_test_split_data

df = load_data(FILE_PATH)
X_train, X_test, y_train, y_test = train_test_split_data(df, text_col="Clean_Text",
                                                         label_col="Label")

model = tf.keras.models.load_model(TENSOR_FILE_PATH)

mlflow.tensorflow.autolog()

X_train_padded, X_test_padded = deep_learning_data(X_train, X_test)

model.fit(X_train_padded, y_train, epochs=5,
          validation_data=(X_test_padded, y_test))