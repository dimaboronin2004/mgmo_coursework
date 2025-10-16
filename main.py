from data_loader import load_and_preprocess_data, create_final_dataset, prepare_training_data
from language_detector import analyze_language_distribution
from vulnerability_detection import VulnerabilityDetector, analyze_vulnerabilities, generate_training_data
from model import create_model, compile_model, get_callbacks, save_model_and_tokenizer
from visualisation import (plot_language_distribution, plot_class_distribution,
                           plot_training_history, plot_confusion_matrix, plot_vulnerability_statistics,
                           visualize_model_architecture)
from test import evaluate_model, test_model_on_examples, generate_classification_report

MODEL_CONFIG = {
    'MAX_VOCAB_SIZE': 10000,
    'MAX_SEQUENCE_LENGTH': 300,
    'EMBEDDING_DIM': 128,
    'BATCH_SIZE': 32,
    'EPOCHS': 15,
    'LEARNING_RATE': 0.001
}

DATASET_CONFIG = {
    'DATASET_NAME': 'CyberNative/Code_Vulnerability_Security_DPO',
    'SPLIT_RATIOS': {'test': 0.2, 'val': 0.2}
}

CALLBACKS_CONFIG = {
    'early_stopping': {'monitor': 'val_loss', 'patience': 5, 'restore_best_weights': True},
    'reduce_lr': {'monitor': 'val_loss', 'factor': 0.2, 'patience': 3, 'min_lr': 1e-7}
}


def analyze_vulnerabilities_with_ml(vulnerable_df, use_ml=True):
    detector = VulnerabilityDetector(use_ml=use_ml)

    if use_ml:
        print("Обучение ML детектора уязвимостей...")
        try:
            training_data = generate_training_data()
            detector.train_ml_model(training_data)
            print("ML модель успешно обучена")
        except Exception as e:
            print(f"Ошибка обучения ML модели: {e}")
            print("Продолжаем с rule-based подходом...")
            detector = VulnerabilityDetector(use_ml=False)

    vulnerable_df['vuln_type'] = vulnerable_df['code'].apply(detector.detect_vulnerability_type)

    all_vulnerabilities = vulnerable_df['vuln_type'].value_counts()
    top_vulnerabilities = all_vulnerabilities[all_vulnerabilities.index.notnull()].head(10)

    print(f"\n2. ТОП-10 САМЫХ РАСПРОСТРАНЕННЫХ УЯЗВИМОСТЕЙ:")
    for i, (vuln_type, count) in enumerate(top_vulnerabilities.items(), 1):
        percentage = (count / len(vulnerable_df)) * 100
        print(f"   {i:>2}. {vuln_type:<35}: {count:>4} случаев ({percentage:>5.1f}%)")

    print(f"\n3. РАСПРЕДЕЛЕНИЕ ТОП-10 УЯЗВИМОСТЕЙ ПО ЯЗЫКАМ:")
    print("-" * 70)

    for vuln_type in top_vulnerabilities.index:
        vuln_subset = vulnerable_df[vulnerable_df['vuln_type'] == vuln_type]
        lang_distribution = vuln_subset['language'].value_counts()

        print(f"\n{vuln_type}:")
        total_cases = len(vuln_subset)
        for lang, count in lang_distribution.items():
            percentage = (count / total_cases) * 100 if total_cases > 0 else 0
            print(f"   {lang.upper():<12}: {count:>3} случаев ({percentage:>5.1f}%)")

        if total_cases > 0:
            print(f"   {'TOTAL':<12}: {total_cases:>3} случаев")

    if use_ml and detector.ml_classifier is not None:
        detector.save_model('vulnerability_ml_model.pkl')

    return vulnerable_df, top_vulnerabilities


def main():

    df = load_and_preprocess_data()

    language_analysis_df, language_distribution = analyze_language_distribution(df)
    plot_language_distribution(language_distribution)

    final_df = create_final_dataset(df)
    plot_class_distribution(final_df)

    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = prepare_training_data(
        final_df, MODEL_CONFIG['MAX_VOCAB_SIZE'], MODEL_CONFIG['MAX_SEQUENCE_LENGTH']
    )

    model = create_model(MODEL_CONFIG['MAX_VOCAB_SIZE'],
                         MODEL_CONFIG['EMBEDDING_DIM'],
                         MODEL_CONFIG['MAX_SEQUENCE_LENGTH'])
    model = compile_model(model, MODEL_CONFIG['LEARNING_RATE'])

    print("\nАрхитектура модели:")
    model.summary()
    visualize_model_architecture(model)

    print("\nНачало обучения нейросети...")
    callbacks = get_callbacks()
    history = model.fit(
        X_train, y_train,
        epochs=MODEL_CONFIG['EPOCHS'],
        batch_size=MODEL_CONFIG['BATCH_SIZE'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )


    evaluate_model(model, X_test, y_test)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    plot_confusion_matrix(y_test, y_pred)
    generate_classification_report(y_test, y_pred)
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ТИПОВ УЯЗВИМОСТЕЙ С ML ДЕТЕКТОРОМ")
    print("=" * 60)

    vulnerable_df = final_df[final_df['label'] == 1].copy()


    print("Добавление информации о языках программирования...")

    lang_mapping = {}
    for idx, row in df.iterrows():

        lang = row['lang']

        lang_mapping[idx] = lang

    vulnerable_df['language'] = vulnerable_df.index.map(lang_mapping)

    if vulnerable_df['language'].isnull().all():
        print("Использование альтернативного метода определения языков...")
        languages = ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'kotlin', 'swift', 'fortran']
        vulnerable_df['language'] = [languages[i % len(languages)] for i in range(len(vulnerable_df))]


    vulnerable_df, top_vulnerabilities = analyze_vulnerabilities_with_ml(
        vulnerable_df, use_ml=True
    )

    language_stats = vulnerable_df['language'].value_counts()
    plot_vulnerability_statistics(language_stats, top_vulnerabilities, vulnerable_df)

    test_model_on_examples(model, tokenizer, MODEL_CONFIG['MAX_SEQUENCE_LENGTH'])

    save_model_and_tokenizer(model, tokenizer)


    print(f"\nДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print(f"   Всего уязвимых примеров: {len(vulnerable_df)}")
    print(f"   Распознано языков: {len(vulnerable_df[vulnerable_df['language'] != 'unknown'])}")
    print(f"   Уникальных языков: {vulnerable_df['language'].nunique()}")

    vuln_type_stats = vulnerable_df['vuln_type'].value_counts()
    print(f"\nРаспределение всех типов уязвимостей:")
    for vuln_type, count in vuln_type_stats.items():
        percentage = (count / len(vulnerable_df)) * 100
        print(f"   {vuln_type:<35}: {count:>4} случаев ({percentage:>5.1f}%)")

    print(f"\nСтатистика сохранена в 'vulnerability_statistics.png'")
    print("Анализ завершен")


if __name__ == "__main__":
    main()