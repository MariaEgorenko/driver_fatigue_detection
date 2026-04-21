# API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health

#### GET /health

Проверка состояния системы.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

| Параметр | Тип | Описание |
|---------|-----|----------|
| status | string | "ok" или "degraded" |
| model_loaded | boolean | ML модель загружена |
| db_connected | boolean | Подключение к БД |

---

### Analysis

#### POST /api/v1/analysis/frame

Анализ одиночного кадра.

**Request:**
```
Content-Type: multipart/form-data

frame: file (JPEG/PNG)
session_id: string (optional)
timestamp: float (optional)
```

**Response:**
```json
{
  "session_id": "uuid",
  "timestamp": 1234567890.123,
  "frame_index": 1,
  "face_detected": true,
  "fatigue_score": {
    "level": "alert",
    "confidence": 0.9,
    "perclos": 0.05,
    "blink_rate": 15.2,
    "yawn_count": 0,
    "head_down": false,
    "face_absent": false,
    "events": []
  },
  "landmarks": {
    "points": [[x,y,z], ...],
    "frame_width": 1920,
    "frame_height": 1080
  }
}
```

#### POST /api/v1/analysis/upload

Асинхронная загрузка видео для анализа.

**Request:**
```
Content-Type: multipart/form-data

file: file (MP4/AVI/MOV)
```

**Response (202 Accepted):**
```json
{
  "id": "uuid",
  "status": "pending",
  "progress": 0.0,
  "created_at": "2024-01-01T12:00:00",
  "estimated_completion": null,
  "overall_fatigue_level": null,
  "frames_analyzed": null,
  "duration_sec": null,
  "error": null
}
```

#### GET /api/v1/analysis/{session_id}/status

Получение статуса сессии анализа.

**Path Parameters:**
| Параметр | Тип | Описание |
|----------|-----|----------|
| session_id | uuid | ID сессии |

**Response:**
```json
{
  "id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "created_at": "2024-01-01T12:00:00",
  "estimated_completion": null,
  "overall_fatigue_level": "mild_fatigue",
  "frames_analyzed": 1500,
  "duration_sec": 45.2,
  "error": null
}
```

---

### Events

#### GET /api/v1/events/session/{session_id}

Получение событий усталости для сессии.

**Path Parameters:**
| Параметр | Тип | Описание |
|----------|-----|----------|
| session_id | uuid | ID сессии |

**Query Parameters:**
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| event_type | string | null | Фильтр по типу |
| severity | string | null | Фильтр по серьезности |
| start_time | float | null | Начало диапазона |
| end_time | float | null | Конец диапазона |
| limit | int | 100 | Лимит |
| offset | int | 0 | Смещение |

**event_type значения:** eye_closure, yawn, head_down, face_absent

**severity значения:** low, medium, high

**Response:**
```json
{
  "events": [
    {
      "event_type": "yawn",
      "timestamp_sec": 1234567890.0,
      "duration_sec": 2.1,
      "severity": "medium",
      "metadata": {"max_mar": 0.65}
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

---

### Metrics

#### GET /metrics

Метрики Prometheus.

---

## Типы данных

### FatigueLevel

| Значение | Описание |
|----------|----------|
| alert | Водитель бодрствует |
| mild_fatigue | Легкая усталость |
| severe_fatigue | Тяжелая усталость |

### SessionStatus

| Значение | Описание |
|----------|----------|
| pending | Ожидает обработки |
| processing | В процессе |
| completed | Завершена |
| failed | Ошибка |

### EventType

| Значение | Описание |
|----------|----------|
| eye_closure | Длительное закрытие глаз |
| yawn | Зевание |
| head_down | Голова опущена |
| face_absent | Лицо не в кадре |

### SeverityLevel

| Значение | Описание |
|----------|----------|
| low | Низкая серьезность |
| medium | Средняя серьезность |
| high | Высокая серьезность |

## Примеры

### Анализ кадра с curl

```bash
# Загрузка кадра
curl -X POST http://localhost:8000/api/v1/analysis/frame \
  -F "frame=@driver.jpg"

# С указанием session_id
curl -X POST http://localhost:8000/api/v1/analysis/frame \
  -F "frame=@driver.jpg" \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000"
```

### Анализ видео

```bash
# Загрузка видео
curl -X POST http://localhost:8000/api/v1/analysis/upload \
  -F "file=@dashcam.mp4"

# Проверка статуса
curl http://localhost:8000/api/v1/analysis/550e8400-e29b-41d4-a716-446655440000/status
```

### Получение событий

```bash
# Все события
curl "http://localhost:8000/api/v1/events/session/550e8400-e29b-41d4-a716-446655440000"

# Фильтр по типу
curl "http://localhost:8000/api/v1/events/session/550e8400-e29b-41d4-a716-446655440000?event_type=yawn"

# Фильтр по серьезности
curl "http://localhost:8000/api/v1/events/session/550e8400-e29b-41d4-a716-446655440000?severity=high"
```

### Python клиент

```python
import httpx

async with httpx.AsyncClient() as client:
    # Анализ кадра
    response = await client.post(
        "http://localhost:8000/api/v1/analysis/frame",
        files={"frame": open("driver.jpg", "rb")}
    )
    result = response.json()
    print(result["fatigue_score"]["level"])

    # Загрузка видео
    response = await client.post(
        "http://localhost:8000/api/v1/analysis/upload",
        files={"file": open("video.mp4", "rb")}
    )
    session_id = response.json()["id"]

    # Проверка статуса
    response = await client.get(
        f"http://localhost:8000/api/v1/analysis/{session_id}/status"
    )
    print(response.json()["status"])
```

## Коды ошибок

| Код | Описание |
|-----|----------|
| 400 | Bad Request — некорректные данные |
| 404 | Not Found — сессия не найдена |
| 413 | Payload Too Large — файл слишком большой |
| 415 | Unsupported Media Type — неподдерживаемый тип |
| 422 | Unprocessable Entity — ошибка валидации |
| 500 | Internal Server Error — ошибка сервера |
| 503 | Service Unavailable — сервис недоступен |