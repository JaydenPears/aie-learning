from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_has_constant_columns_true():
    """
    Проверка: has_constant_columns = True

    DataFrame с одной константной колонкой (все значения одинаковые).
    """
    df = pd.DataFrame({
        "const_col": [1, 1, 1, 1],  # все одинаковые
        "normal_col": [1, 2, 3, 4],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert "const_col" in flags.get("constant_columns_list", [])


def test_has_constant_columns_false():
    """
    Проверка: has_constant_columns = False

    DataFrame без константных колонок.
    """
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4],
        "col2": [5, 6, 7, 8],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is False
    assert len(flags.get("constant_columns_list", [])) == 0


def test_has_suspicious_id_duplicates_true():
    """
    Проверка: has_suspicious_id_duplicates = True

    DataFrame с колонкой 'user_id', где есть дубликаты.
    """
    df = pd.DataFrame({
        "user_id": [1, 1, 3, 4],  # дубликат: 1 встречается 2 раза
        "name": ["Alice", "Bob", "Charlie", "David"],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags.get("suspicious_id_columns", [])


def test_has_suspicious_id_duplicates_false():
    """
    Проверка: has_suspicious_id_duplicates = False

    DataFrame с уникальными ID (нет дубликатов).
    """
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],  # все уникальные
        "value": [10, 20, 30, 40],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_suspicious_id_duplicates"] is False
    assert len(flags.get("suspicious_id_columns", [])) == 0


def test_has_suspicious_id_duplicates_with_client_id():
    """
    Проверка: has_suspicious_id_duplicates = True

    DataFrame с колонкой 'client_id', где есть дубликаты.
    """
    df = pd.DataFrame({
        "client_id": [100, 100, 102],  # дубликат: 100 встречается 2 раза
        "amount": [50.0, 75.0, 100.0],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_suspicious_id_duplicates"] is True


def test_quality_score_with_constant_columns():
    """
    Проверка: quality_score корректно учитывает константные колонки

    DataFrame с константной колонкой должен иметь более низкий score.
    """
    df = pd.DataFrame({
        "const": [1, 1, 1],
        "normal": [1, 2, 3],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Score должен быть < 1.0 из-за штрафа за константные колонки
    assert flags["quality_score"] < 1.0
    assert flags["quality_score"] >= 0.0


def test_quality_score_with_id_duplicates():
    """
    Проверка: quality_score корректно учитывает дубликаты в ID

    DataFrame с дублями в ID должен иметь более низкий score.
    """
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 3],  # дубликат
        "value": [10, 20, 30, 40],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Score должен быть < 1.0 из-за штрафа за дубли
    assert flags["quality_score"] < 1.0
    assert flags["quality_score"] >= 0.0


def test_quality_score_good_data():
    """
    Проверка: quality_score для "хороших" данных должен быть высоким
    """
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],  # уникальные ID
        "col1": [10, 20, 30, 40, 50],
        "col2": [100, 200, 300, 400, 500],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Хороший датасет должен иметь высокий score (близкий к 1.0)
    assert flags["quality_score"] > 0.7
    assert flags["has_constant_columns"] is False
    assert flags["has_suspicious_id_duplicates"] is False


def test_multiple_constant_columns():
    """
    Проверка: несколько константных колонок
    """
    df = pd.DataFrame({
        "const1": ["A", "A", "A"],
        "const2": [999, 999, 999],
        "normal": [1, 2, 3],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert len(flags.get("constant_columns_list", [])) >= 2