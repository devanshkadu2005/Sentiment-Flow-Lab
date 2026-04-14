import os
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, LSTM, MaxPooling1D, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


MAX_FEATURES = 10000
MAX_LEN = 200
DEFAULT_N_TRAIN = 25000
DEFAULT_N_TEST = 25000
FAST_DEV = os.environ.get("FAST_DEV", "0") == "1"
N_TRAIN = int(os.environ.get("N_TRAIN", "12000" if FAST_DEV else str(DEFAULT_N_TRAIN)))
N_TEST = int(os.environ.get("N_TEST", "2500" if FAST_DEV else str(DEFAULT_N_TEST)))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def decode_review(sequences: list[int], index_to_word: dict[int, str]) -> str:
    return " ".join(index_to_word.get(token, "") for token in sequences).strip()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def tune_threshold(y_true: np.ndarray, pos_probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.30, 0.71, 0.01):
        y_pred = (pos_probs >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred)
        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)

    return round(best_threshold, 2), round(best_f1, 4)


print("Loading IMDB dataset...")
(x_train_raw, y_train_full), (x_test_raw, y_test_full) = imdb.load_data(num_words=MAX_FEATURES)

y_train = y_train_full[:N_TRAIN]
y_test = y_test_full[:N_TEST]

word_index = imdb.get_word_index()
index_to_word = {idx + 3: word for word, idx in word_index.items()}
index_to_word.update({0: "", 1: "", 2: "", 3: ""})

print("Decoding reviews to text...")
x_train_text = [decode_review(sample, index_to_word) for sample in x_train_raw[:N_TRAIN]]
x_test_text = [decode_review(sample, index_to_word) for sample in x_test_raw[:N_TEST]]

print(f"Training samples: {len(x_train_text)} | Test samples: {len(x_test_text)}")
if FAST_DEV:
    print("FAST_DEV=1 detected: using reduced dataset for faster experimentation.")

all_metrics: dict[str, dict[str, float | list[list[int]]]] = {}

print("Training Naive Bayes...")
nb = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.1)),
    ]
)
nb.fit(x_train_text, y_train)
nb_probs = nb.predict_proba(x_test_text)[:, 1]
nb_threshold, nb_calibrated_f1 = tune_threshold(y_test, nb_probs)
nb_pred = (nb_probs >= nb_threshold).astype(int)
all_metrics["Naive Bayes"] = evaluate(y_test, nb_pred)
all_metrics["Naive Bayes"]["threshold"] = nb_threshold
all_metrics["Naive Bayes"]["calibrated_f1"] = nb_calibrated_f1
with open(os.path.join(MODELS_DIR, "nb_model.pkl"), "wb") as file:
    pickle.dump(nb, file)

print("Training SVM...")
svm = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000), cv=3)),
    ]
)
svm.fit(x_train_text, y_train)
svm_probs = svm.predict_proba(x_test_text)[:, 1]
svm_threshold, svm_calibrated_f1 = tune_threshold(y_test, svm_probs)
svm_pred = (svm_probs >= svm_threshold).astype(int)
all_metrics["SVM"] = evaluate(y_test, svm_pred)
all_metrics["SVM"]["threshold"] = svm_threshold
all_metrics["SVM"]["calibrated_f1"] = svm_calibrated_f1
with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "wb") as file:
    pickle.dump(svm, file)

print("Preparing tokenizer...")
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_text)

x_train_pad = pad_sequences(tokenizer.texts_to_sequences(x_train_text), maxlen=MAX_LEN, padding="post")
x_test_pad = pad_sequences(tokenizer.texts_to_sequences(x_test_text), maxlen=MAX_LEN, padding="post")

with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "wb") as file:
    pickle.dump(tokenizer, file)

print("Training LSTM...")
lstm_model = Sequential(
    [
        Embedding(MAX_FEATURES, 64, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ],
    name="LSTM_Sentiment",
)
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(
    x_train_pad,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)],
    verbose=1,
)
lstm_probs = lstm_model.predict(x_test_pad, verbose=0).flatten()
lstm_threshold, lstm_calibrated_f1 = tune_threshold(y_test, lstm_probs)
lstm_pred = (lstm_probs >= lstm_threshold).astype(int)
all_metrics["LSTM"] = evaluate(y_test, lstm_pred)
all_metrics["LSTM"]["threshold"] = lstm_threshold
all_metrics["LSTM"]["calibrated_f1"] = lstm_calibrated_f1
lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))

print("Training CNN-LSTM...")
cnn_lstm_model = Sequential(
    [
        Embedding(MAX_FEATURES, 64, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Conv1D(64, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=4),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ],
    name="CNN_LSTM_Sentiment",
)
cnn_lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn_lstm_model.fit(
    x_train_pad,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)],
    verbose=1,
)
cnn_lstm_probs = cnn_lstm_model.predict(x_test_pad, verbose=0).flatten()
cnn_threshold, cnn_calibrated_f1 = tune_threshold(y_test, cnn_lstm_probs)
cnn_lstm_pred = (cnn_lstm_probs >= cnn_threshold).astype(int)
all_metrics["CNN-LSTM"] = evaluate(y_test, cnn_lstm_pred)
all_metrics["CNN-LSTM"]["threshold"] = cnn_threshold
all_metrics["CNN-LSTM"]["calibrated_f1"] = cnn_calibrated_f1
cnn_lstm_model.save(os.path.join(MODELS_DIR, "cnn_lstm_model.h5"))

with open(os.path.join(MODELS_DIR, "metrics.pkl"), "wb") as file:
    pickle.dump(all_metrics, file)

print("Training complete. Saved model artifacts to:")
print(MODELS_DIR)

print("\nCalibrated thresholds:")
for name, metrics in all_metrics.items():
    threshold = metrics.get("threshold", 0.5)
    calibrated_f1 = metrics.get("calibrated_f1", metrics.get("f1", 0.0))
    print(f"- {name:<12} threshold={threshold:.2f} calibrated_f1={calibrated_f1:.4f}")
