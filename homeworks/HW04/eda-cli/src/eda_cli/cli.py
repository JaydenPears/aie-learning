from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
        max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
        top_k_categories: int = typer.Option(5, help="Сколько top-значений выводить для категорий."),
        title: str = typer.Option("EDA Report", help="Заголовок отчёта (в файле Markdown)."),
        min_missing_share: float = typer.Option(0.05,
                                                help="Порог доли пропусков (0..1) для выделения проблемных колонок."),
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    quality_flags = compute_quality_flags(summary, missing_df)

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Параметры отчета\n\n")
        f.write(f"- Top-K Categories: {top_k_categories}\n")
        f.write(f"- Min Missing Share Threshold: {min_missing_share:.0%}\n")
        f.write(f"- Max Histograms: {max_hist_columns}\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"**Общая оценка качества: {quality_flags['quality_score']:.2f}** (0.0 — плохо, 1.0 — отлично)\n\n")

        f.write("### Базовые проверки\n\n")
        f.write(f"- Слишком мало строк (<100): **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок (>100): **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком много пропусков (>50%): **{quality_flags['too_many_missing']}**\n\n")

        f.write("### Обнаруженные проблемы\n\n")

        if quality_flags["has_constant_columns"]:
            f.write(f"⚠️ **Найдены константные (неизменные) колонки:**\n")
            for col_name in quality_flags["constant_columns_list"]:
                f.write(f"  - `{col_name}` (все значения одинаковые)\n")
            f.write("\n_Такие колонки не содержат информации и могут быть удалены._\n\n")
        else:
            f.write("✓ Константные колонки не найдены\n\n")

        if quality_flags["has_suspicious_id_duplicates"]:
            f.write(f"⚠️ **Найдены дубликаты в ID-колонках:**\n")
            for col_name in quality_flags["suspicious_id_columns"]:
                f.write(f"  - `{col_name}` (содержит повторяющиеся значения)\n")
            f.write("\n_Это может указывать на проблемы с целостностью данных._\n\n")
        else:
            f.write("✓ Дубликаты в ID-колонках не найдены\n\n")

        f.write("## Колонки\n\n")
        f.write("Подробная информация в файле `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            high_missing = missing_df[missing_df["missing_share"] > min_missing_share]
            if not high_missing.empty:
                f.write(f"### Колонки с >{min_missing_share:.0%} пропусков:\n\n")
                f.write(high_missing.to_markdown())
                f.write("\n\n")

            f.write("Подробнее см. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"\n✓ Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"  📄 Основной markdown: {md_path}")
    typer.echo(f"  📊 Качество данных: {quality_flags['quality_score']:.2f}")

    if quality_flags["has_constant_columns"]:
        typer.echo(f"  ⚠️  Найдено константных колонок: {len(quality_flags['constant_columns_list'])}")
    if quality_flags["has_suspicious_id_duplicates"]:
        typer.echo(f"  ⚠️  Найдено ID-колонок с дубликатами: {len(quality_flags['suspicious_id_columns'])}")

    typer.echo(f"  📁 Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo(f"  🖼️  Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png\n")


if __name__ == "__main__":
    app()
