# Driver Fatigue Detection System

Система обнаружения усталости водителя на основе компьютерного зрения и машинного обучения.

## Возможности

- **Анализ видеопотока** — обработка видео в реальном времени или видеофайлов
- **Детекция лица** — 468 точек через MediaPipe Face Mesh
- **Отслеживание глаз** — EAR (Eye Aspect Ratio), моргания, PERCLOS
- **Детекция зевания** — MAR (Mouth Aspect Ratio)
- **Положение головы** — оценка 3D ориентации (pitch, yaw, roll)
- **Классификация усталости** — alert / mild_fatigue / severe_fatigue

## Быстрый старт

### Требования

- Python 3.11+
- PostgreSQL 14+ (или SQLite для разработки)
- Docker & Docker Compose (опционально)

### Установка

```bash
# Клонирование репозитория
git clone <repo-url>
cd driver_fatigue_detection

# Создание виртуального окружения
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружений
copy .env.example .env
# Отредактируйте .env с вашими настройками
```

### Запуск

**Docker Compose:**
```bash
docker-compose -f docker/docker-compose.yml up --build
```

## API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | /health | Проверка состояния сервера |
| GET | /metrics | Метрики Prometheus |
| POST | /api/v1/analysis/frame | Анализ кадра |
| POST | /api/v1/analysis/upload | Загрузка видео |
| GET | /api/v1/analysis/{id}/status | Статус сессии |
| GET | /api/v1/events/session/{id} | События сессии |

Документация API: http://localhost:8000/docs

## Архитектура

```
driver_fatigue_detection/
├── ml/                  # Модуль машинного обучения
│   └── src/
│       ├── features/     # Извлечение признаков
│       └── inference/   # Инференс
├── backend/             # FastAPI приложение
│   └── app/
│       ├── api/        # Роутеры
│       ├── core/       # Ядро
│       ├── models/     # Модели БД
│       └── services/   # Сервисы
├── docker/             # Docker конфигурация
├── monitoring/         # Grafana, Prometheus
└── docs/             # Документация
```

## Конфигурация

Основные параметры в `backend/app/core/config.py`:

| Параметр | По умолчанию | Описание |
|---------|-------------|----------|
| EAR_THRESHOLD | 0.20 | Порог EAR для закрытых глаз |
| MAR_THRESHOLD | 0.50 | Порог MAR для зевания |
| PERCLOS_MILD | 0.35 | Порог легкой усталости |
| PERCLOS_SEVERE | 0.70 | Порог тяжелой усталости |
| HEAD_PITCH_THRESHOLD_DEG | 20.0 | Порог наклона головы |

## Тесты

```bash
# ML тесты
pytest ml/tests/ -v

# Backend тесты
pytest backend/tests/ -v
```

## Мониторинг

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Loki (логи)**: http://localhost:3100

## Технологический стек

- **FastAPI** — REST API
- **SQLAlchemy** — ORM
- **MediaPipe** — Детекция лица
- **OpenCV** — Обработка изображений
- **Prometheus** — Метрики
- **Grafana** — Визуализация