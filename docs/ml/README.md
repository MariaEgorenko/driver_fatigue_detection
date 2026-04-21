# ML Module

Модуль машинного обучения для обнаружения усталости водителя.

## Компоненты

### features/

| Файл | Описание |
|------|----------|
| `face_detector.py` | Детектор лица (MediaPipe Face Mesh) |
| `eye_features.py` | Анализ глаз (EAR, моргания, PERCLOS) |
| `mouth_features.py` | Анализ рта (MAR, зевание) |
| `head_pose.py` | Положение головы (pitch, yaw, roll) |
| `presence_tracker.py` | Трекер отсутствия лица |
| `utils.py` | Утилиты |

### inference/

| Файл | Описание |
|------|----------|
| `video_pipeline.py` | Конвейер обработки видео |
| `fatigue_scorer.py` | Оценка уровня усталости |

## Использование

### Обработка видео

```python
from ml.src.inference.video_pipeline import VideoPipeline
from backend.app.core.config import get_settings

config = get_settings()
pipeline = VideoPipeline(config)

# Обработка видеофайла
for result in pipeline.process_video("video.mp4"):
    print(result.fatigue_score.level)
    print(result.fatigue_score.perclos)
```

### Обработка кадра

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = pipeline.process_frame(frame)
    print(result.fatigue_score.level)
```

## Алгоритмы

### Eye Aspect Ratio (EAR)

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Точки глаза: p1,p4 — внешние углы, p2,p3 — верхние точки, p5,p6 — нижние точки.

### Mouth Aspect Ratio (MAR)

```
MAR = ||top-bottom|| / ||left-right||
```

### PERCLOS

```
PERCLOS = (количество кадров с EAR < порог) / (всего кадров в окне)
```

### Положение головы

Метод SolvePnP с 6 точечной 3D моделью лица.

## Типы данных

```python
from ml.src.types import (
    FaceLandmarks,    # 468 точек лица
    BlinkEvent,      # Событие моргания
    YawnEvent,      # Событие зевания
    HeadPose,      # Углы поворота
    FatigueScore,   # Итоговая оценка
    FrameResult,   # Результат кадра
)
```

## Тесты

```bash
pytest ml/tests/ -v
```

Доступные тесты:
- `test_face_detector.py`
- `test_eye_features.py`
- `test_mouth_features.py`
- `test_head_pose.py`
- `test_presence_tracker.py`
- `test_fatigue_scorer.py`