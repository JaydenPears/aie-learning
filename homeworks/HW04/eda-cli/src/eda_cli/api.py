from __future__ import annotations

import logging
import time
from io import StringIO
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse

from .core import (
    compute_quality_flags,
    correlation_matrix,
    missing_table,
    summarize_dataset,
    top_categories,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EDA-CLI Web Service (Extended)",
    description="Расширенный HTTP-сервис для анализа качества и EDA CSV-файлов.",
    version="0.4.0",
)

_processing_stats: List[Dict[str, Any]] = []


# ==================== PYDANTIC MODELS ====================

class QualityResponse(BaseModel):
    """Стандартный ответ для эндпоинтов качества"""
    ok_for_model: Dict[str, bool]
    latency_ms: float


class QualityRequest(BaseModel):
    """Запрос для POST /quality с данными"""
    data: Dict[str, Any]
    quality_threshold: float = 0.7


# ==================== HEALTH CHECK ====================

@app.get("/health", summary="Проверка доступности сервиса")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "version": "0.4.0"}


# ==================== ОБЯЗАТЕЛЬНЫЕ ЭНДПОИНТЫ ====================

@app.post("/quality", response_model=QualityResponse, summary="Вычислить метрики качества")
async def quality(request: QualityRequest) -> QualityResponse:
    """
    Вычисляет метрики качества для переданных данных.

    Возвращает:
    - ok_for_model: словарь с результатами проверки моделей
    - latency_ms: время обработки в миллисекундах
    """
    start_time = time.time()

    try:
        # Преобразуем данные в DataFrame для анализа
        df = pd.DataFrame([request.data] if isinstance(request.data, dict) else request.data)

        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        # Определяем, пригодны ли данные для различных моделей
        ok_for_model = {
            "regression": not flags.get("has_missing_values", False),
            "classification": not flags.get("has_missing_values", False),
            "clustering": not flags.get("has_constant_columns", False),
            "neural_network": flags.get("quality_score", 0) >= request.quality_threshold,
        }

        latency_ms = (time.time() - start_time) * 1000

        return QualityResponse(
            ok_for_model=ok_for_model,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Ошибка в /quality: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")


@app.post("/quality-from-csv", response_model=QualityResponse, summary="Метрики качества из CSV файла")
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Вычисляет метрики качества для всех данных из CSV файла.

    Возвращает:
    - ok_for_model: словарь с результатами проверки для каждой модели
    - latency_ms: время обработки в миллисекундах
    """
    start_time = time.time()

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Некорректный формат файла. Пожалуйста, загрузите .csv файл.",
        )

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(csv_data)

        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        # Определяем пригодность для различных типов моделей
        ok_for_model = {
            "regression": not flags.get("has_missing_values", False) and summary.n_rows > 10,
            "classification": not flags.get("has_missing_values", False) and summary.n_rows > 10,
            "clustering": not flags.get("has_constant_columns", False) and summary.n_cols > 2,
            "neural_network": flags.get("quality_score", 0) >= 0.7,
        }

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Обработан файл '{file.filename}' за {latency_ms:.2f} мс.")

        return QualityResponse(
            ok_for_model=ok_for_model,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {e}"
        )


# ==================== РАСШИРЕННЫЕ ЭНДПОИНТЫ ====================

@app.post("/quality-flags-from-csv", summary="Получить все флаги качества из CSV")
async def get_quality_flags_from_csv(file: UploadFile = File(...)) -> JSONResponse:
    start_time = time.time()

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Некорректный формат файла. Пожалуйста, загрузите .csv файл.",
        )

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(csv_data)

        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        flags_to_return = {
            key: value for key, value in flags.items()
            if isinstance(value, (bool, int, float)) and "share" not in key
        }

        latency = time.time() - start_time

        _processing_stats.append({
            "filename": file.filename,
            "timestamp": pd.Timestamp.now().isoformat(),
            "rows": summary.n_rows,
            "cols": summary.n_cols,
            "quality_score": flags["quality_score"],
            "processing_time": latency,
        })

        logger.info(f"Обработан файл '{file.filename}' за {latency:.4f} сек.")

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "rows": summary.n_rows,
                "columns": summary.n_cols,
                "flags": flags_to_return,
                "processing_time_sec": round(latency, 4),
            },
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {e}"
        )


@app.post("/summary-from-csv", summary="Получить полную сводку о датасете")
async def get_summary_from_csv(file: UploadFile = File(...)) -> JSONResponse:
    start_time = time.time()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Используйте CSV файл")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(csv_data)

        summary = summarize_dataset(df)

        columns_info = []
        for col in summary.columns:
            columns_info.append({
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": round(col.missing_share, 4),
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
                "example_values": col.example_values,
            })

        latency = time.time() - start_time

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "dataset_info": {
                    "rows": summary.n_rows,
                    "columns": summary.n_cols,
                },
                "columns": columns_info,
                "processing_time_sec": round(latency, 4),
            },
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")


@app.post("/missing-analysis-from-csv", summary="Детальный анализ пропусков")
async def get_missing_analysis(file: UploadFile = File(...)) -> JSONResponse:
    start_time = time.time()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Используйте CSV файл")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(csv_data)

        missing_df = missing_table(df)

        missing_by_column = {}
        if not missing_df.empty:
            for col_name in missing_df.index:
                missing_by_column[col_name] = {
                    "missing_count": int(missing_df.loc[col_name, "missing_count"]),
                    "missing_share": round(float(missing_df.loc[col_name, "missing_share"]), 4),
                }

        total_missing = int(missing_df["missing_count"].sum()) if not missing_df.empty else 0

        high_missing = []
        if not missing_df.empty:
            high_missing = [
                col for col in missing_df.index
                if missing_df.loc[col, "missing_share"] > 0.5
            ]

        latency = time.time() - start_time

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "missing_by_column": missing_by_column,
                "total_missing_cells": total_missing,
                "high_missing_columns": high_missing,
                "has_any_missing": len(missing_by_column) > 0,
                "processing_time_sec": round(latency, 4),
            },
        )

    except Exception as e:
        logger.error(f"Ошибка при анализе пропусков: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")


@app.post("/quality-report-json", summary="Полный отчет в JSON формате")
async def get_quality_report_json(
        file: UploadFile = File(...),
        include_correlation: bool = Query(False, description="Включить матрицу корреляций"),
        include_categories: bool = Query(False, description="Включить top-категории"),
) -> JSONResponse:
    start_time = time.time()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Используйте CSV файл")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(csv_data)

        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        report = {
            "filename": file.filename,
            "dataset": {
                "rows": summary.n_rows,
                "columns": summary.n_cols,
            },
            "quality": {
                "score": round(flags["quality_score"], 2),
                "flags": {
                    key: value for key, value in flags.items()
                    if isinstance(value, (bool, int, float)) and "list" not in key
                },
                "problems": []
            },
        }

        if flags.get("has_constant_columns"):
            report["quality"]["problems"].append({
                "type": "constant_columns",
                "columns": flags.get("constant_columns_list", []),
                "severity": "medium",
            })

        if flags.get("has_suspicious_id_duplicates"):
            report["quality"]["problems"].append({
                "type": "id_duplicates",
                "columns": flags.get("suspicious_id_columns", []),
                "severity": "high",
            })

        if include_correlation:
            corr_df = correlation_matrix(df)
            if not corr_df.empty:
                report["correlation"] = corr_df.to_dict()

        if include_categories:
            top_cats = top_categories(df, max_columns=5, top_k=5)
            report["categories"] = {
                name: table.to_dict(orient="records")
                for name, table in top_cats.items()
            }

        latency = time.time() - start_time

        return JSONResponse(
            status_code=200,
            content={
                "report": report,
                "processing_time_sec": round(latency, 4),
            },
        )

    except Exception as e:
        logger.error(f"Ошибка при генерации отчета: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")


@app.get("/quality-score-benchmark", summary="Статистика по обработанным файлам")
def get_benchmark_stats(
        limit: int = Query(10, ge=1, le=100, description="Максимум записей"),
) -> JSONResponse:
    if not _processing_stats:
        return JSONResponse(
            status_code=200,
            content={
                "message": "Нет обработанных файлов",
                "stats": [],
                "summary": None,
            },
        )

    recent_stats = _processing_stats[-limit:]
    scores = [s["quality_score"] for s in _processing_stats]

    summary = {
        "total_processed": len(_processing_stats),
        "avg_quality_score": round(sum(scores) / len(scores), 3),
        "min_quality_score": round(min(scores), 3),
        "max_quality_score": round(max(scores), 3),
        "avg_rows": round(sum(s["rows"] for s in _processing_stats) / len(_processing_stats)),
        "avg_cols": round(sum(s["cols"] for s in _processing_stats) / len(_processing_stats)),
    }

    return JSONResponse(
        status_code=200,
        content={
            "recent_files": recent_stats,
            "summary": summary,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)