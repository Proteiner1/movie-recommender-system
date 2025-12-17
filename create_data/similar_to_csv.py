import pandas as pd
import os
import numpy as np
import re

def is_excel_file(file_path):
    """Проверяет, является ли файл Excel-файлом по сигнатуре"""
    try:
        with open(file_path, 'rb') as f:
            signature = f.read(4)
            return signature == b'PK\x03\x04'  # Сигнатура ZIP/XLSX
    except:
        return False

def calculate_average_rating(row):
    """Вычисляет среднюю оценку из доступных данных, игнорируя пропуски и нули"""
    ratings = []
    
    def safe_float(value):
        if value is None or pd.isna(value):
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value) if float(value) > 0 else None
            
            str_val = str(value).strip()
            if not str_val or str_val.lower() in ['nan', 'none', '']:
                return None
            
            str_val = str_val.replace(',', '.')
            str_val = re.sub(r'[^\d\.]', '', str_val)
            if not str_val:
                return None
            return float(str_val) if float(str_val) > 0 else None
        except (ValueError, TypeError):
            return None
    
    if len(row) > 5:
        kp_rating = safe_float(row[5])
        if kp_rating is not None:
            ratings.append(kp_rating)
    
    if len(row) > 7:
        imdb_rating = safe_float(row[7])
        if imdb_rating is not None:
            ratings.append(imdb_rating)
    
    if len(row) > 8:
        critics_rating = safe_float(row[8])
        if critics_rating is not None:
            ratings.append(critics_rating)
    
    return round(sum(ratings) / len(ratings), 1) if ratings else 0.0

def clean_text(value):
    """Очищает текст от кавычек и лишних символов"""
    if pd.isna(value) or value is None:
        return ""
    
    text = str(value).strip()
    text = text.strip('"')
    text = re.sub(r'\s+', ' ', text)
    return text

def load_original_file(file_path):
    """
    Загружает оригинальный CSV файл со всеми колонками
    """
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return None
    
    try:
        if is_excel_file(file_path):
            print(f"Обнаружен Excel-файл: {os.path.basename(file_path)}")
            df = pd.read_excel(file_path, header=None)
        else:
            print(f"Читаем как CSV-файл: {os.path.basename(file_path)}")
            
            # Читаем весь файл как текст
            with open(file_path, 'r', encoding='cp1251') as f:
                content = f.read()
            
            # Разбиваем на строки и убираем пустые
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Разбиваем каждую строку по точке с запятой
            data = []
            for line in lines:
                parts = line.split(';')
                
                # Очищаем каждую часть
                cleaned_parts = [clean_text(part) for part in parts]
                
                data.append(cleaned_parts)
            
            if not data:
                return None
            
            # Определяем максимальное количество колонок
            max_cols = max(len(row) for row in data)
            
            # Если в каких-то строках меньше колонок, дополняем пустыми значениями
            for i in range(len(data)):
                if len(data[i]) < max_cols:
                    data[i].extend([''] * (max_cols - len(data[i])))
            
            df = pd.DataFrame(data)
        
        # Заполняем NaN значения
        df = df.fillna('')
        
        # Удаляем полностью пустые строки
        df = df[df.apply(lambda row: any(cell != '' for cell in row), axis=1)]
        
        print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        return df
        
    except Exception as e:
        print(f"Ошибка при чтении {file_path}: {e}")
        return None

def find_top_20_similar_movies_for_movie(df, movie_idx, avg_ratings):
    """
    Находит топ-20 похожих фильмов для указанного фильма в DataFrame
    
    Args:
        df: DataFrame с фильмами
        movie_idx: Индекс фильма в DataFrame
        avg_ratings: Список средних оценок для каждого фильма
    
    Returns:
        Строка с названиями 20 похожих фильмов, разделенных точкой с запятой
    """
    if movie_idx >= len(df):
        return ""
    
    target_movie = df.iloc[movie_idx]
    target_movie_name = clean_text(target_movie.iloc[0])
    
    # Извлекаем характеристики найденного фильма
    if len(target_movie) > 1:
        movie_genre = clean_text(target_movie[1])  # Это ЖАНР (колонка 1)
    else:
        movie_genre = ""
    
    if len(target_movie) > 2:
        movie_crit2 = clean_text(target_movie[2])  # Это критерий 2 (колонка 2)
    else:
        movie_crit2 = ""
    
    if len(target_movie) > 3:
        movie_crit3 = clean_text(target_movie[3])  # Это критерий 3 (колонка 3)
    else:
        movie_crit3 = ""
    
    if len(target_movie) > 4:
        movie_crit4 = clean_text(target_movie[4])  # Это критерий 4 (колонка 4)
    else:
        movie_crit4 = ""
    
    # Рассчитываем паттерн совпадений для каждого фильма
    match_patterns = []
    exact_matches_counts = []
    
    for idx, row in df.iterrows():
        # Пропускаем исходный фильм
        if idx == movie_idx:
            match_patterns.append("1111")
            exact_matches_counts.append(3)  # Все 3 критерия совпадают
            continue
        
        pattern = ""
        exact_matches = 0
        
        # КРИТЕРИЙ 1: Жанр (колонка 1)
        if len(row) > 1:
            current_genre = clean_text(row[1])
            if current_genre == movie_genre:
                pattern += "1"
                # Жанр не считается в exact_matches, так как все фильмы в одном файле имеют одинаковый жанр
            else:
                pattern += "2"
        else:
            pattern += "2"
        
        # КРИТЕРИЙ 2: (колонка 2)
        if len(row) > 2:
            current_crit2 = clean_text(row[2])
            if current_crit2 == movie_crit2:
                pattern += "1"
                exact_matches += 1
            else:
                pattern += "2"
        else:
            pattern += "2"
        
        # КРИТЕРИЙ 3: (колонка 3)
        if len(row) > 3:
            current_crit3 = clean_text(row[3])
            if current_crit3 == movie_crit3:
                pattern += "1"
                exact_matches += 1
            else:
                pattern += "2"
        else:
            pattern += "2"
        
        # КРИТЕРИЙ 4: (колонка 4)
        if len(row) > 4:
            current_crit4 = clean_text(row[4])
            if current_crit4 == movie_crit4:
                pattern += "1"
                exact_matches += 1
            else:
                pattern += "2"
        else:
            pattern += "2"
        
        match_patterns.append(pattern)
        exact_matches_counts.append(exact_matches)
    
    # Создаем временный DataFrame для сортировки
    temp_df = df.copy()
    temp_df['match_pattern'] = match_patterns
    temp_df['exact_matches'] = exact_matches_counts
    temp_df['Средняя оценка'] = avg_ratings
    temp_df['is_original'] = temp_df.index == movie_idx
    
    # Убедимся, что средняя оценка числовая
    temp_df['Средняя оценка'] = pd.to_numeric(temp_df['Средняя оценка'], errors='coerce')
    
    # Сортируем по релевантности в правильном порядке:
    # 1. exact_matches (убывание - больше совпадений лучше)
    # 2. match_pattern (возрастание - "1111" < "1112" < "1121" < "1122" < ... < "2222")
    # 3. Средняя оценка (убывание - выше оценка лучше)
    temp_df_sorted = temp_df.sort_values(
        by=['exact_matches', 'match_pattern', 'Средняя оценка'],
        ascending=[False, True, False]
    )
    
    # Исключаем исходный фильм из результатов и берем топ-20
    result_df = temp_df_sorted[~temp_df_sorted['is_original']].head(20)
    
    # Собираем названия фильмов в строку, разделенную точкой с запятой
    similar_movies = []
    for idx, row in result_df.iterrows():
        title = clean_text(row.iloc[0])
        if title:  # Не добавляем пустые названия
            similar_movies.append(title)
    
    return ';'.join(similar_movies)

def process_genre_file(base_path, genre_name):
    """
    Обрабатывает файл с фильмами определенного жанра и создает обновленную версию
    
    Args:
        base_path: Путь к папке с файлами
        genre_name: Название жанра (имя файла без расширения)
    """
    file_path = os.path.join(base_path, f"{genre_name}.csv")
    print(f"\nНачинаю обработку файла: {genre_name}.csv")
    
    # Загружаем оригинальные данные со всеми колонками
    df = load_original_file(file_path)
    if df is None or df.empty:
        print(f"Не удалось загрузить данные из {file_path}")
        return
    
    print(f"Загружено {len(df)} фильмов для обработки")
    
    # Вычисляем среднюю оценку для каждого фильма
    print("Вычисляю средние оценки...")
    avg_ratings = []
    for _, row in df.iterrows():
        avg_ratings.append(calculate_average_rating(row))
    
    # Добавляем новую колонку для похожих фильмов
    df['20_похожих'] = ""
    
    # Обрабатываем каждый фильм
    total_movies = len(df)
    for i in range(total_movies):
        if i % 100 == 0:
            print(f"  Обработано {i} из {total_movies} фильмов в файле {genre_name}.csv")
        
        # Находим топ-20 похожих фильмов для текущего фильма
        similar_movies_str = find_top_20_similar_movies_for_movie(df, i, avg_ratings)
        df.at[i, '20_похожих'] = similar_movies_str
    
    print(f"  Завершена обработка всех {total_movies} фильмов в файле {genre_name}.csv")
    
    # Создаем путь для нового файла
    new_file_name = f"{genre_name}_обновленный.csv"
    new_file_path = os.path.join(base_path, new_file_name)
    
    # Сохраняем в новый файл
    try:
        # Сохраняем все колонки с разделителем точка с запятой
        df.to_csv(new_file_path, sep=';', index=False, encoding='cp1251')
        print(f"✓ Создан новый файл: {new_file_name}")
        print(f"  Колонок в файле: {len(df.columns)} (включая новую колонку '20_похожих')")
    except Exception as e:
        print(f"✗ Ошибка при сохранении файла {new_file_name}: {e}")

def process_all_genre_files(base_path):
    """
    Обрабатывает все CSV файлы в указанной папке
    
    Args:
        base_path: Путь к папке с файлами жанров
    """
    print("=" * 80)
    print("НАЧАЛО ОБРАБОТКИ ВСЕХ ФАЙЛОВ С ФИЛЬМАМИ")
    print("=" * 80)
    
    if not os.path.exists(base_path):
        print(f"Папка {base_path} не найдена!")
        return
    
    # Получаем список всех CSV файлов в папке (кроме уже обновленных)
    all_files = os.listdir(base_path)
    csv_files = [f for f in all_files if f.endswith('.csv') and not f.endswith('_обновленный.csv')]
    
    if not csv_files:
        print(f"В папке {base_path} не найдено CSV файлов для обработки")
        return
    
    print(f"Найдено {len(csv_files)} файлов для обработки:")
    for file in csv_files:
        print(f"  - {file}")
    
    total_files = len(csv_files)
    
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*80}")
        print(f"Обработка файла {idx} из {total_files}: {csv_file}")
        print(f"{'='*80}")
        
        # Извлекаем название жанра из имени файла (убираем расширение .csv)
        genre_name = csv_file.replace('.csv', '')
        
        # Обрабатываем файл
        process_genre_file(base_path, genre_name)
    
    print(f"\n{'='*80}")
    print("ОБРАБОТКА ВСЕХ ФАЙЛОВ ЗАВЕРШЕНА!")
    print(f"{'='*80}")

# Тестовая функция для проверки логики
def test_similarity_logic(genre, movie_title):
    """
    Тестовая функция для проверки логики подбора похожих фильмов
    """
    print("=" * 80)
    print(f"ТЕСТ ЛОГИКИ: Поиск похожих на '{movie_title}' в жанре '{genre}'")
    print("=" * 80)
    
    base_path = r"C:\Users\User\Desktop\genre_with_info"
    file_path = os.path.join(base_path, f"{genre}.csv")
    
    # Загружаем данные
    df = load_original_file(file_path)
    if df is None or df.empty:
        print(f"Не удалось загрузить данные из {file_path}")
        return
    
    # Вычисляем среднюю оценку
    avg_ratings = []
    for _, row in df.iterrows():
        avg_ratings.append(calculate_average_rating(row))
    
    # Находим фильм
    movie_idx = -1
    search_title = clean_text(movie_title).lower()
    
    for idx, row in df.iterrows():
        title = clean_text(row.iloc[0]).lower()
        if title == search_title:
            movie_idx = idx
            break
    
    if movie_idx == -1:
        print(f"Фильм '{movie_title}' не найден")
        return
    
    target_movie = df.iloc[movie_idx]
    movie_name = clean_text(target_movie.iloc[0])
    
    # Извлекаем характеристики
    if len(target_movie) > 1:
        movie_genre = clean_text(target_movie[1])
    else:
        movie_genre = ""
    
    if len(target_movie) > 2:
        movie_crit2 = clean_text(target_movie[2])
    else:
        movie_crit2 = ""
    
    if len(target_movie) > 3:
        movie_crit3 = clean_text(target_movie[3])
    else:
        movie_crit3 = ""
    
    if len(target_movie) > 4:
        movie_crit4 = clean_text(target_movie[4])
    else:
        movie_crit4 = ""
    
    print(f"Найденный фильм: '{movie_name}'")
    print(f"Характеристики:")
    print(f"  1. Жанр: {movie_genre}")
    print(f"  2. {movie_crit2}")
    print(f"  3. {movie_crit3}")
    print(f"  4. {movie_crit4}")
    print(f"  Средняя оценка: {avg_ratings[movie_idx]:.1f}\n")
    
    # Находим похожие
    similar_movies_str = find_top_20_similar_movies_for_movie(df, movie_idx, avg_ratings)
    similar_movies = similar_movies_str.split(';')
    
    # Для отладки выводим подробную информацию
    print(f"Топ-20 похожих фильмов на '{movie_name}':")
    print(f"{'№':<3} {'Название':<30} | {'Совпад.':>7} | {'Паттерн':>7} | {'Оценка':>7} | {'Крит2':>6} | {'Крит3':>6} | {'Крит4':>6}")
    print("-" * 85)
    
    # Создаем временный DataFrame для вывода подробностей
    match_patterns = []
    exact_matches_counts = []
    crit2_matches = []
    crit3_matches = []
    crit4_matches = []
    
    for idx, row in df.iterrows():
        if idx == movie_idx:
            continue
            
        pattern = "1"  # Жанр всегда совпадает (все в одном файле)
        exact_matches = 0
        
        # Критерий 2
        if len(row) > 2:
            current_crit2 = clean_text(row[2])
            if current_crit2 == movie_crit2:
                pattern += "1"
                exact_matches += 1
                crit2_match = "✓"
            else:
                pattern += "2"
                crit2_match = "✗"
        else:
            pattern += "2"
            crit2_match = "✗"
        
        # Критерий 3
        if len(row) > 3:
            current_crit3 = clean_text(row[3])
            if current_crit3 == movie_crit3:
                pattern += "1"
                exact_matches += 1
                crit3_match = "✓"
            else:
                pattern += "2"
                crit3_match = "✗"
        else:
            pattern += "2"
            crit3_match = "✗"
        
        # Критерий 4
        if len(row) > 4:
            current_crit4 = clean_text(row[4])
            if current_crit4 == movie_crit4:
                pattern += "1"
                exact_matches += 1
                crit4_match = "✓"
            else:
                pattern += "2"
                crit4_match = "✗"
        else:
            pattern += "2"
            crit4_match = "✗"
        
        match_patterns.append(pattern)
        exact_matches_counts.append(exact_matches)
        crit2_matches.append(crit2_match)
        crit3_matches.append(crit3_match)
        crit4_matches.append(crit4_match)
    
    temp_df = df.copy()
    temp_df = temp_df[temp_df.index != movie_idx]
    temp_df['match_pattern'] = match_patterns
    temp_df['exact_matches'] = exact_matches_counts
    temp_df['Средняя оценка'] = [avg_ratings[i] for i in range(len(avg_ratings)) if i != movie_idx]
    temp_df['crit2_match'] = crit2_matches
    temp_df['crit3_match'] = crit3_matches
    temp_df['crit4_match'] = crit4_matches
    
    # Сортируем
    temp_df_sorted = temp_df.sort_values(
        by=['exact_matches', 'match_pattern', 'Средняя оценка'],
        ascending=[False, True, False]
    ).head(20)
    
    for i, (idx, row) in enumerate(temp_df_sorted.iterrows(), 1):
        title = clean_text(row.iloc[0])
        display_title = title[:28] + "..." if len(title) > 30 else title
        
        print(f"{i:<3} {display_title:<30} | {row['exact_matches']:>7} | {row['match_pattern']:>7} | {row['Средняя оценка']:>7.1f} | {row['crit2_match']:>6} | {row['crit3_match']:>6} | {row['crit4_match']:>6}")

# Пример использования с заданными параметрами
if __name__ == "__main__":
    # Путь к папке с файлами жанров
    base_path = r"C:\Users\User\Desktop\genre_with_info"
    
    # Тестовые примеры для проверки логики
    test_cases = [
        {"genre": "Ужасы", "movie_title": "Рассвет мертвецов"},
        # Можно добавить больше тестовых случаев
    ]
    
    # Вариант 1: Проверить логику для конкретных фильмов
    print("Тест логики подбора похожих фильмов")
    print("=" * 80)
    
    for test in test_cases:
        test_similarity_logic(test["genre"], test["movie_title"])
        print("\n")
    
    
    # Вариант 2: Обработать все файлы в папке
    print("\n" + "=" * 80)
    print("Обработка всех CSV файлов в папке")
    print("=" * 80)
    
    continue_processing = input("Начать обработку всех файлов? (y/n): ")
    if continue_processing.lower() == 'y':
        process_all_genre_files(base_path)
    
