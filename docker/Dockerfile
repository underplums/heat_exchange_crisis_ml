# Базовый образ Python
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем только зависимости сначала
# это ускоряет сборку docker слоя
COPY requirements.txt .

# Устанавливаем python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Команда запуска API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]