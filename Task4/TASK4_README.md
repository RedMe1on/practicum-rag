# Задание 4. RAG-бот с техниками промптинга

## Описание

RAG-бот (Retrieval-Augmented Generation) с **бесплатной локальной LLM** — 
без необходимости API-ключа.

### Техники промптинга

1. **Few-shot prompting** — 2 примера ответов из базы знаний
2. **Chain-of-Thought (CoT)** — пошаговое рассуждение перед ответом

## Бесплатные LLM-бэкенды

| Бэкенд | Стоимость | Установка | Скорость |
|--------|-----------|-----------|----------|
| **Ollama** (по умолчанию) | Бесплатно | 2 минуты | Средняя (CPU) |
| **Groq** | Бесплатный tier | API-ключ | Быстрая |
| **Демо** | Бесплатно | Нет | Мгновенно (мок) |

### Вариант 1: Ollama (рекомендуется, бесплатно, локально)

```bash
# 1. Установка Ollama (Windows)
# Скачать с https://ollama.com

# 2. Запуск Ollama
ollama serve

# 3. Загрузка модели (3B параметров, ~2GB)
ollama pull llama3.2

# 4. Запуск RAG-бота
set LLM_BACKEND=ollama
python Task4\rag_bot.py
```

### Вариант 2: Groq (бесплатный API, быстро)

```bash
# 1. Получить бесплатный API-ключ: https://console.groq.com
# 2. Запуск
set GROQ_API_KEY=gsk_...
set LLM_BACKEND=groq
python Task4\rag_bot.py
```

### Вариант 3: Демо-режим (без LLM, мгновенно)

```bash
set LLM_BACKEND=demo
python Task4\rag_bot.py
```

## Архитектура пайплайна

```
Запрос пользователя
    │
    ▼
┌─────────────────────────────┐
│  1. Embedding (E5-large-v2) │  — преобразование запроса в вектор
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. ChromaDB search          │  — поиск top-k ближайших чанков
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Prompt builder           │  — контекст + few-shot + CoT + запрос
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. LLM (Ollama/Groq/Demo)  │  — генерация ответа
└─────────────┬───────────────┘
              │
              ▼
        Ответ пользователю
```

## Файлы

| Файл | Назначение |
|------|-----------|
| `rag_bot.py` | Основной модуль RAG-бота (Ollama/Groq/Demo) |
| `test_rag_bot.py` | Тестовый прогон (3 вопроса, сохранение JSON) |
| `TASK4_README.md` | Этот файл |
| `test_results/` | Результаты тестовых запусков |
| `demo_results/` | Результаты демо-запусков |

## FastAPI REST API

Сервер предоставляет 3 endpoint'а для работы с RAG-ботом.

### Запуск сервера

```bash
cd Task4
set LLM_BACKEND=demo
python rag_api.py
```

Сервер запускается на `http://localhost:8000`.
Документация Swagger UI доступна на `http://localhost:8000/docs`.

### Endpoints

#### `GET /health` — Проверка состояния

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "ok",
  "index_chunks": 949,
  "llm_backend": "demo",
  "ollama_available": false,
  "groq_available": false
}
```

#### `POST /query` — Задать вопрос (RAG)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Zyron Silvervein?", "top_k": 3, "backend": "demo"}'
```

**Request body:**
| Поле | Тип | Описание |
|------|-----|----------|
| `question` | string | Вопрос пользователя (обязательно) |
| `top_k` | int | Количество чанков для поиска (по умолчанию 5) |
| `backend` | string | LLM-бэкенд: `ollama`, `groq`, `demo` (авто-выбор если не указан) |
| `include_reasoning` | bool | Включить CoT-рассуждение в ответ (по умолчанию true) |

**Ответ:**
```json
{
  "question": "Who is Zyron Silvervein?",
  "answer": "Based on the knowledge base fragments...",
  "reasoning": "1. Analyzing the knowledge base...\n2. Found relevant...",
  "chunks": [
    {
      "chunk_id": "Johnny_Silverhand_chunk_000",
      "source_file": "Johnny_Silverhand.txt",
      "chunk_index": 0,
      "similarity": 0.8441,
      "text_preview": "Sub-Pages:\nMain\nGallery..."
    }
  ],
  "total_time": 0.25,
  "backend": "demo"
}
```

#### `GET /search` — Поиск без генерации ответа

```bash
curl "http://localhost:8000/search?q=What+is+Synthware&top_k=3"
```

**Параметры:**
| Параметр | Тип | Описание |
|----------|-----|----------|
| `q` | string | Поисковый запрос (обязательно) |
| `top_k` | int | Количество результатов (по умолчанию 5) |

**Ответ:**
```json
{
  "query": "What is Synthware",
  "results_count": 2,
  "chunks": [
    {
      "chunk_id": "Cyberware_chunk_002",
      "source_file": "Cyberware.txt",
      "chunk_index": 2,
      "similarity": 0.8944,
      "text": "Synthware is any cybernetic technology..."
    }
  ]
}
```

## Зависимости

```bash
pip install sentence-transformers chromadb fastapi uvicorn
```

## Запуск

### Интерактивный режим

```bash
python Task4\rag_bot.py
```

Автоматически выбирает доступный бэкенд: Ollama → Groq → Demo.

### Тестовый режим (автоматический)

```bash
python Task4\test_rag_bot.py
```

## Техники промптинга

### 1. Few-shot prompting

В промпт включены 2 примера, **извлечённых из самой базы знаний**:

```
Q: Who is Zyron Silvervein?
A: Zyron Silvervein (born Robert John Linder) was a famous rockerboy
   and lead singer of the band Synthwave...

Q: What is Synthware?
A: Synthware is cybernetic technology permanently installed
   into the body...

Q: [запрос пользователя]
A: ...
```

### 2. Chain-of-Thought (CoT)

Системный промпт с пошаговым рассуждением:

```
You are a knowledge assistant for the Voidpunk 2177 universe.
Answer using ONLY the provided context fragments.

Use Chain-of-Thought reasoning. Before answering:
1. Think through the fragments step by step.
2. Then give your final answer.

Format:
REASONING:
1. [Step]
2. [Step]
3. [Step]

ANSWER:
[Final answer]

SOURCES:
- [Filename] (chunk N)
```

## Пример работы

```
Q: Who is Zyron Silvervein and what is his connection to Xarn Velgor?

[1/4] Searching knowledge base...
  Found 5 relevant chunks
[2/4] Building prompt with few-shot + CoT...
[3/4] Calling LLM (ollama: llama3.2)...
[4/4] Done!

============================================================
RAG-BOT ANSWER:
============================================================
REASONING:
1. Searching for Zyron Silvervein in the context...
2. Fragment 1 states he was a rockerboy and lead singer of Synthwave...
3. Fragment 2 mentions his engram resides in Xarn Velgor's brain...

ANSWER:
Zyron Silvervein was a legendary rockerboy whose digitized consciousness
now lives in Xarn Velgor's brain via the Synthcore implant.

SOURCES:
- Johnny_Silverhand.txt (chunk 0, similarity: 83.8%)
- Johnny_Silverhand.txt (chunk 37, similarity: 83.2%)
```

## Конфигурация

| Параметр | Значение |
|----------|----------|
| **Embedding модель** | `intfloat/e5-large-v2` (1024 dim) |
| **Векторная БД** | ChromaDB (926+ чанков, cosine) |
| **LLM** | Ollama llama3.2 (3B) / Groq llama-3.1-8b |
| **Top-k чанков** | 5 |
| **Few-shot примеров** | 2 |
| **Temperature** | 0.3 |
