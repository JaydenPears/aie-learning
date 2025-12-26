# S04 – eda_cli: Web API для анализа EDA

Расширенный HTTP-сервис для анализа качества и EDA CSV-файлов.
Используется в рамках Семинара 04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта:

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск API

### Запуск uvicorn через uv

Для запуска веб-сервера используйте команду:

```bash
uv run uvicorn eda_cli.api:app --reload
```

Или с полными параметрами:

```bash
uv run uvicorn eda_cli.api:app --host 0.0.0.0 --port 8000 --reload
```

**Примечание:** Флаг `--reload` автоматически перезагружает сервер при изменении кода (удалите его для production).

Сервер будет доступен по адресу:
```
http://localhost:8000
```

## API Эндпоинты

### Здоровье сервиса

#### GET /health
Проверка доступности сервиса.

**Ответ:**
```json
{
  "status": "ok",
  "version": "0.4.0"
}
```

---

### Обязательные эндпоинты качества

#### POST /quality
**Вычисляет метрики качества для переданных данных.**

Основной эндпоинт для оценки пригодности данных для различных типов моделей машинного обучения.

**Request body:**
```json
{
  "data": {
    "column1": "value1",
    "column2": 123,
    "column3": 45.6
  },
  "quality_threshold": 0.7
}
```

**Response:**
```json
{
  "ok_for_model": {
    "regression": true,
    "classification": true,
    "clustering": false,
    "neural_network": true
  },
  "latency_ms": 12.34
}
```

**Параметры:**
- `data` (dict): Словарь с данными для анализа
- `quality_threshold` (float, опционально): Порог качества для нейросетей (по умолчанию 0.7)

**Возвращаемые поля:**
- `ok_for_model`: Словарь с булевыми значениями для каждого типа модели
- `latency_ms`: Время обработки в миллисекундах

---

#### POST /quality-from-csv
**Вычисляет метрики качества из загруженного CSV файла.**

Предназначена для анализа полных CSV файлов и определения их пригодности для различных моделей.

**Request:**
Multipart form-data с CSV файлом

**Response:**
```json
{
  "ok_for_model": {
    "regression": true,
    "classification": true,
    "clustering": true,
    "neural_network": false
  },
  "latency_ms": 45.67
}
```

**Параметры:**
- `file` (UploadFile): CSV файл для анализа

**Возвращаемые поля:**
- `ok_for_model`: Словарь с оценкой пригодности для каждого типа модели
- `latency_ms`: Время обработки в миллисекундах

---

### Расширенные эндпоинты анализа

#### POST /quality-flags-from-csv
**Получить все флаги качества из CSV файла.**

Возвращает детальные флаги качества и статистику по датасету.

**Response:**
```json
{
  "filename": "data.csv",
  "rows": 1000,
  "columns": 15,
  "flags": {
    "has_missing_values": true,
    "has_constant_columns": false,
    "has_suspicious_id_duplicates": false,
    "quality_score": 0.85
  },
  "processing_time_sec": 0.123
}
```

---

#### POST /summary-from-csv
**Получить полную сводку о датасете.**

Детальная информация о каждой колонке в CSV файле.

**Response:**
```json
{
  "filename": "data.csv",
  "dataset_info": {
    "rows": 1000,
    "columns": 15
  },
  "columns": [
    {
      "name": "age",
      "dtype": "int64",
      "non_null": 998,
      "missing": 2,
      "missing_share": 0.002,
      "unique": 87,
      "is_numeric": true,
      "min": 18,
      "max": 85,
      "mean": 45.3,
      "std": 12.4,
      "example_values": [25, 34, 45]
    }
  ],
  "processing_time_sec": 0.234
}
```

---

#### POST /missing-analysis-from-csv
**Детальный анализ пропусков в данных.**

**Response:**
```json
{
  "filename": "data.csv",
  "missing_by_column": {
    "age": {
      "missing_count": 2,
      "missing_share": 0.002
    },
    "email": {
      "missing_count": 15,
      "missing_share": 0.015
    }
  },
  "total_missing_cells": 17,
  "high_missing_columns": [],
  "has_any_missing": true,
  "processing_time_sec": 0.089
}
```

---

#### POST /quality-report-json
**Полный EDA отчет в JSON формате.**

Комплексный отчет о качестве и статистике датасета с опциональными матрицами корреляций и топ-категориями.

**Query параметры:**
- `include_correlation` (bool): Включить матрицу корреляций
- `include_categories` (bool): Включить топ-категории

**Response:**
```json
{
  "report": {
    "filename": "data.csv",
    "dataset": {
      "rows": 1000,
      "columns": 15
    },
    "quality": {
      "score": 0.85,
      "flags": {...},
      "problems": []
    },
    "correlation": {...},
    "categories": {...}
  },
  "processing_time_sec": 0.456
}
```

---

#### GET /quality-score-benchmark
**Статистика по всем обработанным файлам.**

Получить общую статистику по обработанным файлам и качеству датасетов.

**Query параметры:**
- `limit` (int): Максимум записей (по умолчанию 10, максимум 100)

**Response:**
```json
{
  "recent_files": [
    {
      "filename": "data1.csv",
      "timestamp": "2025-12-26T10:30:00",
      "rows": 1000,
      "cols": 15,
      "quality_score": 0.85,
      "processing_time": 0.123
    }
  ],
  "summary": {
    "total_processed": 5,
    "avg_quality_score": 0.82,
    "min_quality_score": 0.71,
    "max_quality_score": 0.92,
    "avg_rows": 1200,
    "avg_cols": 12
  }
}
```

---

## Примеры использования

### Пример 1: Проверить качество данных через /quality

```bash
curl -X POST "http://localhost:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature1": 10,
      "feature2": 20,
      "feature3": 30
    },
    "quality_threshold": 0.7
  }'
```

### Пример 2: Загрузить CSV файл для анализа

```bash
curl -X POST "http://localhost:8000/quality-from-csv" \
  -F "file=@data.csv"
```

### Пример 3: Получить детальный отчет

```bash
curl -X POST "http://localhost:8000/quality-report-json?include_correlation=true&include_categories=true" \
  -F "file=@data.csv"
```

### Пример 4: Анализ пропусков

```bash
curl -X POST "http://localhost:8000/missing-analysis-from-csv" \
  -F "file=@data.csv"
```

## Тесты

```bash
uv run pytest -q
```

## Структура проекта

```
homeworks/HW04/
├── eda_cli/
│   ├── __init__.py
│   ├── api.py              # HTTP API эндпоинты
│   ├── core.py             # Основная логика EDA
│   └── ...
├── pyproject.toml
├── uv.lock
└── README.md
```

## Версия

- **API версия**: 0.4.0
- **Python**: 3.11+
