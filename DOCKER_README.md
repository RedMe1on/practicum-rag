# RAG Bot — Docker

## Запуск

```bash
docker compose up -d
```

Команда поднимет:
- **ollama** — LLM-сервер с моделью `llama3.2`
- **rag-api** — FastAPI API на порту `8000`
- **ollama-init** — автоматически скачивает модель `llama3.2`

## Проверка

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Zyron Silvervein?"}'
```

Swagger UI: http://localhost:8000/docs

## Остановка

```bash
docker compose down
```

## Структура

```
.
├── docker-compose.yml
├── Dockerfile.rag
├── Task3/
│   └── chroma_index_db/    # ChromaDB индекс (монтируется в контейнер)
└── Task4/
    ├── rag_api.py
    ├── rag_bot.py
    └── requirements.txt
```

## Порты

| Сервис | Порт |
|--------|------|
| RAG API | 8000 |
| Ollama | 11434 |

## Переменные окружения

| Переменная | Значение | Описание |
|------------|----------|----------|
| `LLM_BACKEND` | `ollama` | Бэкенд LLM |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | URL Ollama внутри docker-network |
