# Развертывание

## Локальная разработка

### Требования

- Python 3.11+
- PostgreSQL 14+ (или SQLite)
- Git

### Установка

```bash
# Клонирование
git clone <repo-url>
cd driver_fatigue_detection

# Виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux

# Зависимости
pip install -r requirements.txt

# Скачивание MediaPipe модели
# Модель face_landmarker.task должна быть в корне проекта
```

### Конфигурация

```bash
# Копирование примера конфигурации
copy .env.example .env

# Редактирование .env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/fatigue
# Или для SQLite:
DATABASE_URL=sqlite+aiosqlite:///fatigue.db
```

### Запуск

```bash
# База данных (PostgreSQL)
# Создайте базу данных fatigue

# Запуск API
uvicorn backend.app.main:app --reload

# Откройте http://localhost:8000/docs
```

## Docker Compose

### Требования

- Docker 24+
- Docker Compose 2.20+

### Конфигурация

```bash
# Создайте .env файл
cat > .env << EOF
POSTGRES_DB=fatigue
POSTGRES_USER=fatigue
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql+asyncpg://fatigue:your_secure_password@db:5432/fatigue
DATA_SOURCE_NAME=postgresql://fatigue:your_secure_password@db:5432/fatigue
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
EOF
```

### Запуск

```bash
# Сборка и запуск
docker-compose -f docker/docker-compose.yml up --build

# Остановка
docker-compose -f docker/docker-compose.yml down
```

### Сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| API | http://localhost:8000 | FastAPI |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Prometheus | http://localhost:9090 | Метрики |
| Grafana | http://localhost:3000 | Дашборды |
| Loki | http://localhost:3100 | Логи |

### Логины по умолчанию

- **Grafana**: admin / admin
- **Prometheus**: http://localhost:9090

## Production

### Nginx reverse proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Systemd сервис

```ini
[Unit]
Description=Driver Fatigue Detection API
After=network.target postgresql.service

[Service]
Type=simple
User=fatigue
Group=fatigue
WorkingDirectory=/opt/fatigue
ExecStart=/opt/fatigue/venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Установка
sudo cp fatigue.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable fatigue
sudo systemctl start fatigue
```

### PostgreSQL Production

```sql
-- Создание пользователя и базы
CREATE USER fatigue WITH PASSWORD 'your_secure_password';
CREATE DATABASE fatigue OWNER fatigue;
GRANT ALL PRIVILEGES ON DATABASE fatigue TO fatigue;
```

## Мониторинг

### Проверка состояния сервера

```bash
curl http://localhost:8000/health
```

### Метрики

```bash
curl http://localhost:8000/metrics
```

### Логи

```bash
docker-compose logs -f backend
```

## Обновление

```bash
# Docker
docker-compose pull
docker-compose up -d --build

# pip
pip install -r requirements.txt --upgrade
```

## Безопасность

### Переменные окружения

- Не коммитить `.env` в репозиторий
- Использовать Secrets в Kubernetes
- Регулярно обновлять пароли


## Troubleshooting

### Ошибки подключения к БД

```bash
# Проверка PostgreSQL
docker-compose exec db psql -U fatigue -d fatigue

# Проверка логов
docker-compose logs db
```

### Ошибки ML модели

```bash
# Проверка модели
ls -la face_landmarker.task

# Проверка логов
docker-compose logs backend
```

### Высокая нагрузка

- Увеличьте `FRAME_PROCESSING_INTERVAL`
- Используйте GPU для инференса
- Горизонтальное масштабирование