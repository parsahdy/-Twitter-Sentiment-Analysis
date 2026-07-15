import tensorflow as tf


def tokenizer(X_train, X_test):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=50000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    char_level=False,
    oov_token="<OOV>",
    analyzer=None,
    )

    tokenizer.fit_on_texts(X_train)  # Vocabulary

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    return X_train_sequences, X_test_sequences


def deep_learning_data(X_train, X_test):

    X_train_sequences, X_test_sequences = tokenizer(X_train, X_test)

    X_train_padded = tf.keras.utils.pad_sequences(
    X_train_sequences,
    maxlen=18,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0
    )

    X_test_padded = tf.keras.utils.pad_sequences(
    X_test_sequences,
    maxlen=18,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0
    )

    return X_train_padded, X_test_padded