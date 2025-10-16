import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


def create_model(max_vocab_size, embedding_dim, max_sequence_length):
    model = Sequential([
        Embedding(
            input_dim=max_vocab_size,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            mask_zero=True
        ),
        Bidirectional(GRU(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(GRU(32)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    return model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

def save_model_and_tokenizer(model, tokenizer, model_filename='code_vulnerability_detector.h5',
                             tokenizer_filename='tokenizer.pkl'):
    print("\nСохранение модели...")
    model.save(model_filename)

    with open(tokenizer_filename, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Модель и токенизатор сохранены!")
    print(f"Модель: {model_filename}")
    print(f"Токенизатор: {tokenizer_filename}")