import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def remove_comments(code_snippet: str) -> str:
    if not code_snippet or not isinstance(code_snippet, str):
        return ""

    lines = code_snippet.split('\n')
    cleaned_lines = []
    in_multiline_comment = False
    multiline_comment_start = None

    for i, line in enumerate(lines):
        current_line = line
        original_line = line

        if in_multiline_comment:
            end_index = current_line.find('*/')
            if end_index != -1:
                # Нашли конец комментария - удаляем всё до */
                current_line = current_line[end_index + 2:]
                in_multiline_comment = False
                multiline_comment_start = None
            else:
                current_line = ''

        if not in_multiline_comment:
            multiline_start = current_line.find('/*')
            if multiline_start != -1:
                multiline_comment_start = (i, multiline_start)
                multiline_end = current_line.find('*/', multiline_start + 2)

                if multiline_end != -1:
                    current_line = (current_line[:multiline_start] +
                                    current_line[multiline_end + 2:])
                else:
                    current_line = current_line[:multiline_start]
                    in_multiline_comment = True


        if not in_multiline_comment and current_line:
            comment_positions = []


            for comment_marker in ['//', '#', '!']:
                pos = current_line.find(comment_marker)
                if pos != -1:
                    comment_positions.append(pos)


            html_comment_pos = current_line.find('<!--')
            if html_comment_pos != -1:
                comment_positions.append(html_comment_pos)

            if comment_positions:

                first_comment_pos = min(comment_positions)
                current_line = current_line[:first_comment_pos]


        current_line = current_line.rstrip()


        if current_line.strip():
            cleaned_lines.append(current_line)


    if in_multiline_comment:
        print(f"Предупреждение: Незакрытый многострочный комментарий, начатый в строке {multiline_comment_start}")

    cleaned_code = '\n'.join(cleaned_lines)
    return cleaned_code


def load_and_preprocess_data():
    print("Загрузка датасета...")
    dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")
    df = pd.DataFrame(dataset['train'])

    print("Структура датасета:")
    print(df.head())
    print(f"\nРазмер датасета: {len(df)} примеров")
    print(f"Колонки: {df.columns.tolist()}")

    return df


def create_final_dataset(df):
    vulnerable_examples = []
    safe_examples = []

    for _, row in df.iterrows():
        rejected_code = row['rejected'][0]['content'] if isinstance(row['rejected'], list) else row['rejected']
        cleaned_rejected = remove_comments(rejected_code)

        vulnerable_examples.append({
            'code': cleaned_rejected,
            'label': 1  # Уязвимый
        })

        chosen_code = row['chosen'][0]['content'] if isinstance(row['chosen'], list) else row['chosen']
        cleaned_chosen = remove_comments(chosen_code)

        safe_examples.append({
            'code': cleaned_chosen,
            'label': 0
        })


    final_df = pd.DataFrame(vulnerable_examples + safe_examples)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nФинальный датасет:")
    print(f"Уязвимых примеров: {len(final_df[final_df['label'] == 1])}")
    print(f"Безопасных примеров: {len(final_df[final_df['label'] == 0])}")
    print(f"Все примеры очищены от комментариев")

    return final_df


def prepare_training_data(final_df, max_vocab_size, max_sequence_length):
    print("\nТокенизация данных...")
    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(final_df['code'])

    sequences = tokenizer.texts_to_sequences(final_df['code'])
    X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    y = np.array(final_df['label'])

    print(f"Размерность данных: {X.shape}")


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Валидационная выборка: {X_val.shape}")
    print(f"Тестовая выборка: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer
