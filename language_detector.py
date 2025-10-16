import pandas as pd

def analyze_language_distribution(df):
    print("\nАнализ распределения языков в датасете...")

    all_code_samples = []
    for _, row in df.iterrows():
        langu = row['lang']

        all_code_samples.extend([langu])

    language_analysis_df = pd.DataFrame({'language': all_code_samples})

    language_distribution = language_analysis_df['language'].value_counts()

    print("\nРАСПРЕДЕЛЕНИЕ ЯЗЫКОВ ПРОГРАММИРОВАНИЯ В ДАТАСЕТЕ:")
    print("=" * 50)
    for lang, count in language_distribution.items():
        percentage = (count / len(language_analysis_df)) * 100
        print(f"{lang.upper():<12}: {count:>5} примеров ({percentage:>5.1f}%)")

    return language_analysis_df, language_distribution