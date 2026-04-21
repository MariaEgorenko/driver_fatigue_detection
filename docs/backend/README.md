# Backend API

REST API для системы обнаружения усталости водителя.

## Структура

```
backend/app/
├── api/                    # API роутеры
│   ├── health.py           # /health
│   ├── analysis.py         # /api/v1/analysis
│   └── events.py           # /api/v1/events
├── core/                   # Ядро приложения
│   ├── config.py           # Конфигурация
│   ├── database.py         # База данных
│   ├── dependencies.py     # Dependencies
│   ├── enums.py            # Перечисления
│   └── metrics.py          # Метрики
├── models/                 # SQLAlchemy модели
│   └── db_models.py
├── schemas/                # Pydantic схемы
│   ├── analysis.py
│   └── health.py
├── services/               # Бизнес-логика
│   └── analysis_service.py
└── main.py                 # Точка входа
```

## API Endpoints

### System

**GET /health** — Проверка здоровья системы
```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

**GET /metrics** — Метрики Prometheus

### Analysis

**POST /api/v1/analysis/frame** — Анализ одиночного кадра

```bash
curl -X POST http://localhost:8000/api/v1/analysis/frame \
  -F "frame=@image.jpg"
```
Ответ:
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "timestamp": 0,
  "frame_index": 0,
  "face_detected": true,
  "fatigue_score": {
    "level": "alert",
    "confidence": 1,
    "perclos": 1,
    "blink_rate": 0,
    "yawn_count": 0,
    "head_down": true,
    "face_absent": true,
    "events": [
      {
        "event_type": "eye_closure",
        "timestamp_sec": 0,
        "duration_sec": 0,
        "severity": "low",
        "metadata": {
          "additionalProp1": {}
        }
      }
    ]
  },
  "landmarks": {
    "points": [
      [
        0
      ]
    ],
    "frame_width": 0,
    "frame_height": 0
  }
}
```

**POST /api/v1/analysis/upload** — Асинхронная загрузка видео

```bash
curl -X POST http://localhost:8000/api/v1/analysis/upload \
  -F "file=@video.mp4"
```

Ответ:
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "pending",
  "progress": 0,
  "created_at": "2026-04-21T01:17:39.571Z",
  "estimated_completion": "2026-04-21T01:17:39.571Z",
  "overall_fatigue_level": "alert",
  "frames_analyzed": 0,
  "duration_sec": 0,
  "error": "string"
}
```

**GET /api/v1/analysis/{session_id}/status** — Статус сессии

### Events

**GET /api/v1/events/session/{session_id}** — События сессии

Параметры:
- `event_type` — фильтр по типу
- `severity` — фильтр по серьезности
- `start_time` / `end_time` — фильтр по времени
- `limit` / `offset` — пагинация

## Конфигурация

Переменные окружения в `.env`:

```bash
# База данных
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/fatigue

# Размеры файлов
MAX_FRAME_SIZE_MB=10
VIDEO_MAX_SIZE_MB=500

# ML параметры
EAR_THRESHOLD=0.20
MAR_THRESHOLD=0.50
YAWN_MIN_DURATION_SEC=1.5
HEAD_PITCH_THRESHOLD_DEG=20.0
FACE_ABSENT_THRESHOLD_SEC=3.0
PERCLOS_WINDOW_SEC=60.0
PERCLOS_MILD=0.35
PERCLOS_SEVERE=0.70
```

## Запуск

```bash
# Docker
docker-compose up backend
```

## Тесты

```bash
pytest backend/tests/ -v
```

Тесты:
- `tests/integration/` — Интеграционные тесты API
- `tests/e2e/` — E2E тесты