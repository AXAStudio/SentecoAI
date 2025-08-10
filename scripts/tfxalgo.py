import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    TextVectorization, Embedding, Bidirectional, LSTM,
    Dense, Dropout, SpatialDropout1D, GlobalAveragePooling1D
)

# Load and Prepare Data
df = pd.read_csv("balanced_utf8.csv", encoding="utf-8")

# Overfitting guard: drop exact duplicate headlines
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

texts = df['text'].astype(str).tolist()
labels = df['label'].replace({-1: 0, 1: 1}).tolist()
labels = np.array(labels)

# K-Fold Cross-Validation Setup
max_vocab_size = 10000
max_length = 100
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
all_metrics = []

# K-Fold Training
for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
    X_train, X_val = np.array(texts, dtype=object)[train_idx], np.array(texts, dtype=object)[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # New vectorizer per fold to prevent leakage
    vectorizer = TextVectorization(
        max_tokens=max_vocab_size,
        output_mode='int',
        output_sequence_length=max_length,
        standardize='lower_and_strip_punctuation',
        split='whitespace'
    )
    text_ds_train = tf.data.Dataset.from_tensor_slices(X_train).batch(64)
    vectorizer.adapt(text_ds_train)

    # Use the actual learned vocab size (+2 for safety re: PAD/OOV indexing)
    vocab_size = int(vectorizer.vocabulary_size())

    # Model (regularization-focused changes only)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        Embedding(input_dim=vocab_size + 2, output_dim=64, mask_zero=True),
        SpatialDropout1D(0.25),
        Bidirectional(LSTM(48, return_sequences=True, recurrent_dropout=0.1)),
        # Mask-aware pooling (helps generalization vs max-pooling on padded seqs)
        GlobalAveragePooling1D(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        Dropout(0.4),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    # Overfitting-related compile settings
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    opt  = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    # Train Model
    print(f"\nTraining Fold {fold}...")
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).prefetch(tf.data.AUTOTUNE)

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1),
        tf.keras.callbacks.ModelCheckpoint(f'best_fold_{fold}.keras', monitor='val_loss', save_best_only=True)
    ]
    model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=cb, verbose=1)

    # Evaluate
    val_pred_ds = tf.data.Dataset.from_tensor_slices(X_val).batch(64)
    y_pred_probs = model.predict(val_pred_ds, vekrbose=0).ravel()
    y_pred = (y_pred_probs >= 0.5).astype("int32")

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

    print(f"✅ Fold {fold} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    all_metrics.append({
        'fold': fold,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Final Metrics
results_df = pd.DataFrame(all_metrics)
print("\n📊 Average Metrics Across Folds:")
print(results_df.mean(numeric_only=True).round(4))

# Save Final Model (last fold's model with best weights restored)
keras.models.save_model(model, "model.keras")

# Inference Loop
print("\n🔍 Sentiment Prediction (type 'exit' to quit):")

def predict_sentiment(text_input: str):
    arr = tf.constant([text_input])
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob

while True:
    user_input = input("Enter a headline: ").strip()
    if user_input.lower() == "exit":
        print("Exiting.")
        break
    if not user_input:
        continue
    label, prob = predict_sentiment(user_input)
    print(f"Prediction: {label} ({prob:.2f})\n")
