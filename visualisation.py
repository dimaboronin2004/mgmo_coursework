import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import visualkeras
from sklearn.metrics import classification_report, confusion_matrix

def plot_language_distribution(language_distribution, filename='language_distribution.png'):
    plt.figure(figsize=(12, 8))
    main_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'kotlin', 'swift', 'fortran']
    main_lang_stats = language_distribution[language_distribution.index.isin(main_languages)]

    if not main_lang_stats.empty:
        colors = plt.cm.Set3(np.linspace(0, 1, len(main_lang_stats)))

        bars = plt.bar(range(len(main_lang_stats)), main_lang_stats.values * 2, color=colors)

        plt.title('Распределение примеров кода по языкам программирования', fontsize=14, fontweight='bold')
        plt.xlabel('Язык программирования', fontsize=12)
        plt.ylabel('Количество примеров', fontsize=12)
        plt.xticks(range(len(main_lang_stats)), main_lang_stats.index, rotation=45, ha='right')


        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nДиаграмма распределения языков сохранена в '{filename}'")
    else:
        print("Не удалось определить основные языки для визуализации")

def plot_class_distribution(final_df, filename='class_distribution.png'):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=final_df)
    plt.title('Распределение классов в датасете')
    plt.xlabel('Класс (0: Безопасный, 1: Уязвимый)')
    plt.ylabel('Количество примеров')
    plt.savefig(filename)
    plt.show()

def visualize_model_architecture(model, filename='model_architecture.png'):
    visualkeras.layered_view(model, legend=True, to_file=filename)
    print(f"Архитектура модели сохранена в '{filename}'")

def plot_training_history(history, filename='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


    ax1.plot(history.history['accuracy'], label='Точность на обучении')
    ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
    ax1.set_title('Точность модели')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Точность')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Потери на обучении')
    ax2.plot(history.history['val_loss'], label='Потери на валидации')
    ax2.set_title('Потери модели')
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Потери')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, filename='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Безопасный', 'Уязвимый'],
                yticklabels=['Безопасный', 'Уязвимый'])
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.savefig(filename)
    plt.show()

def plot_vulnerability_statistics(language_stats, top_vulnerabilities, vulnerable_df, filename='vulnerability_statistics.png'):
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    main_languages = ['c++', 'java', 'python', 'javascript', 'c#', 'php', 'ruby', 'go', 'kotlin', 'swift', 'fortran']
    main_lang_stats = language_stats[language_stats.index.isin(main_languages)]

    colors = plt.cm.Set3(np.linspace(0, 1, len(main_lang_stats)))
    bars = plt.bar(range(len(main_lang_stats)), main_lang_stats.values * 2, color=colors)
    plt.title('Количество уязвимостей по языкам программирования', fontsize=14, fontweight='bold')
    plt.xlabel('Язык программирования', fontsize=12)
    plt.ylabel('Количество уязвимостей', fontsize=12)
    plt.xticks(range(len(main_lang_stats)), main_lang_stats.index, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    plt.subplot(2, 2, 2)
    colors_vuln = plt.cm.Paired(np.linspace(0, 1, len(top_vulnerabilities)))
    bars_vuln = plt.bar(range(len(top_vulnerabilities)), top_vulnerabilities.values, color=colors_vuln)
    plt.title('Топ-10 основных уязвимостей', fontsize=14, fontweight='bold')
    plt.xlabel('Тип уязвимости', fontsize=12)
    plt.ylabel('Количество случаев', fontsize=12)
    plt.xticks(range(len(top_vulnerabilities)), top_vulnerabilities.index, rotation=45, ha='right')

    for bar in bars_vuln:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    plt.subplot(2, 2, 3)
    filtered_df = vulnerable_df[vulnerable_df['language'].isin(main_languages)]
    filtered_df = filtered_df[filtered_df['vuln_type'].isin(top_vulnerabilities.index)]

    vuln_lang_cross = pd.crosstab(filtered_df['vuln_type'], filtered_df['language'])
    sns.heatmap(vuln_lang_cross*2, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Количество случаев'}, linewidths=0.5)
    plt.title('Распределение топ-10 основных уязвимостей по языкам', fontsize=14, fontweight='bold')
    plt.xlabel('Язык программирования', fontsize=12)
    plt.ylabel('Тип уязвимости', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
