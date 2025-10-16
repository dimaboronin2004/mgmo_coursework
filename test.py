from sklearn.metrics import classification_report
from vulnerability_detection import predict_vulnerability


def evaluate_model(model, X_test, y_test):
    print("\nОценка на тестовых данных...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

    return test_results


def test_model_on_examples(model, tokenizer, max_sequence_length):
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА ПРИМЕРАХ")
    print("=" * 50)

    vulnerable_code = """
    from flask import request
    import sqlite3
    import os

    def user_login():
        username = request.form['username']
        password = request.form['password']


        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        sql = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        cursor.execute(sql)


        if cursor.fetchone():
            return f"<h1>Добро пожаловать, {username}!</h1><script>trackUser('{username}')</script>"


        log_entry = f"Failed login for {username}"
        os.system(f"echo '{log_entry}' >> /var/log/auth.log")

        return "Ошибка входа"
    """

    safe_code = """
import re

def validate_username(username):
    if not re.match("^[a-zA-Z0-9_]{3,20}$", username):
        raise ValueError("Недопустимое имя пользователя")
    return username

def get_user_data(username):
    validated_username = validate_username(username)
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (validated_username,))
    return cursor.fetchall()
"""

    print("\nТест 1 - Уязвимый код (SQL injection):")
    predict_vulnerability(model, tokenizer, vulnerable_code, max_sequence_length)

    print("\nТест 2 - Безопасный код:")
    predict_vulnerability(model, tokenizer, safe_code, max_sequence_length)


def generate_classification_report(y_test, y_pred):
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=['Безопасный', 'Уязвимый']))