# Архитектура системы

## Обзор

Система построена по принципу многоуровневой архитектуры с четким разделением ответственности между компонентами.

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (REST API)                      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│     │   Health    │  │  Analysis   │  │     Events      │   │
│     └─────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ ML Pipeline   │  │    Database   │  │  Metrics      │
    │ (VideoPipe)   │  │  (PostgreSQL) │  │ (Prometheus)  │
    └───────────────┘  └───────────────┘  └───────────────┘
```

## Компоненты

### Frontend Layer (Client)

- REST клиент для загрузки видео/изображений
- Получение результатов анализа

### API Layer (FastAPI)

**Роутеры:**
- `health.py` — системное здоровье
- `analysis.py` — анализ видео и кадров
- `events.py` — работа с событиями

**Middlewares:**
- CORS middleware
- Metrics middleware (Prometheus)

### Business Logic Layer

**Сервисы:**
- `AnalysisService` — управление сессиями и событиями

### Data Layer

**Модели БД:**
- `AnalysisSession` — сессия анализа
- `FatigueEventDB` — события усталости
- `FrameResultDB` — результаты кадров

### ML Layer

```
                    VideoFrame
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                VideoPipeline                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Face    │  │   Eye    │  │  Mouth   │    │
│  │ Detector |  │ Analyzer │  │ Analyzer │    │
│  └──────────┘  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   Head   │  │ Presence │  │ Feature  │    │
│  │   Pose   │  │ Tracker  │  │ Window   │    │
│  └──────────┘  └──────────┘  └──────────┘    │
└──────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                FatigueScorer                  │
│        • Aggregates metrics                   │
│        • Determines fatigue level             │
│        • Calculates confidence                │
└───────────────────────────────────────────────┘
                        │
                        ▼
FatigueScore {level, perclos, blink_rate, yawn_count, head_down, ...}
```

## Паттерны проектирования

### Dependency Injection

```python
# backend/app/core/dependencies.py
async def get_ml_pipeline(request: Request) -> VideoPipeline:
    return VideoPipeline(request.app.state.ml_config)
```

### Repository Pattern

```python
# backend/app/services/analysis_service.py
class AnalysisService:
    async def create_session(...) -> AnalysisSession:
        ...
    async def get_session(...) -> AnalysisSession:
        ...
```

### Factory Pattern

```python
# ml/src/features/face_detector.py
class FaceDetector:
    @classmethod
    def create(cls, config) -> "FaceDetector":
        ...
```

## Поток данных

### Синхронный (frame)

```
HTTP Request → FastAPI → VideoPipeline.process_frame()
                → FaceDetector.detect()
                    → FaceLandmarks
                → EyeAnalyzer.compute_ear()
                → MouthAnalyzer.compute_mar()
                → HeadPoseEstimator.estimate()
                → FatigueScorer.score()
                    → FatigueScore
                → HTTP Response
```

### Асинхронный (video)

```
HTTP Request → FastAPI → Create Session
                → Background Task
                    → VideoPipeline.process_video()
                        → for each frame:
                            → process_frame()
                            → save events every N frames
                → Session ID → HTTP 202 Response
```

## Мониторинг

### Метрики

| Метрика | Тип | Описание |
|---------|-----|----------|
| frame_processing_seconds | Histogram | Время обработки кадра |
| active_sessions | Gauge | Активные сессии |
| fatigue_events_total | Counter | События усталости |
| api_requests_total | Counter | HTTP запросы |

### Логирование

- Python logging с ротацией
- Loki для агрегации
- Grafana для визуализации

## База данных

### PostgreSQL

```
┌──────────────────┐     ┌──────────────────┐
│analysis_sessions │ 1:N │ fatigue_events   │
│  (id, status)    │─────│ (session_id)     │
│  (source_type)   │     │ (event_type)     │
│  (progress)      │     │ (timestamp_sec)  │
└──────────────────┘     └──────────────────┘
           │
           │ 1:N
           ▼
┌──────────────────┐
│  frame_results   │
│  (session_id)    │
│  (frame_index)   │
│  (fatigue_level) │
└──────────────────┘
```

## Конфигурация

### Слои конфигурации

1. `.env` — пользовательские настройки
2. `config.py` — значения по умолчанию
3. `docker-compose.yml` — инфраструктура

## Безопасность

- CORS настроен для всех источников (для разработки)
- Валидация размеров файлов
- Лимиты на загрузку
