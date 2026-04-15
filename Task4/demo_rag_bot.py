"""
Демо-версия RAG-бота без LLM.
Показывает работу пайплайна: поиск → промпт → (мок ответ)
Не требует OPENAI_API_KEY.
"""
import os
import sys
import json
import time
import logging

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Task4.rag_bot import (
    search_chunks,
    build_prompt,
    format_few_shot,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
)

logger = logging.getLogger("demo-rag-bot")

TEST_QUESTIONS = [
    "Who is Zyron Silvervein and what is his connection to Xarn Velgor?",
    "What is Synthware and how is it installed in the body?",
    "Which corporations operate in Void City and what is Xarasaka's role?",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def mock_llm_answer(query, chunks):
    """
    Генерирует мок-ответ на основе найденных чанков.
    В реальном боте здесь вызывается LLM.
    """
    # Simple extraction-based mock answer
    top_chunk = chunks[0]
    source = top_chunk["source_file"]
    text_preview = top_chunk["text"][:300]

    mock_answer = (
        f"РАЗМЫШЛЕНИЕ:\n"
        f"1. По запросу \"{query}\" найдено {len(chunks)} релевантных фрагментов.\n"
        f"2. Наиболее релевантный источник: {source} (чанк {top_chunk['chunk_index']}, "
        f"сходство: {top_chunk['similarity']:.1%}).\n"
        f"3. Фрагмент содержит ключевую информацию по вопросу.\n\n"
        f"ОТВЕТ:\n"
        f"На основе найденных фрагментов из {source}:\n\n"
        f"{text_preview}...\n\n"
        f"ИСТОЧНИКИ:\n"
    )

    for chunk in chunks[:3]:
        mock_answer += (
            f"- {chunk['source_file']} "
            f"(чанк {chunk['chunk_index']}, "
            f"сходство: {chunk['similarity']:.1%})\n"
        )

    return mock_answer


def main():
    logger.info("=" * 60)
    logger.info("  RAG-бот: Демо пайплайна (без LLM)")
    logger.info("  Техники: Few-shot + Chain-of-Thought")
    logger.info("=" * 60)

    all_results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info("=" * 60)
        logger.info("  Вопрос %s/%s", i, len(TEST_QUESTIONS))
        logger.info("=" * 60)
        logger.info("Q: %s", question)

        # 1. Search
        logger.info("[1/4] Searching knowledge base...")
        start = time.time()
        chunks = search_chunks(question, top_k=5)
        search_time = time.time() - start
        logger.info("  Found %s chunks in %.2fs", len(chunks), search_time)

        # Show chunk similarity
        for j, chunk in enumerate(chunks):
            bar = "█" * int(chunk["similarity"] * 30) + "░" * (30 - int(chunk["similarity"] * 30))
            logger.info(
                "  [%s] %s %.1f%%  %s (chunk %s)",
                j + 1,
                bar,
                chunk["similarity"] * 100,
                chunk["source_file"],
                chunk["chunk_index"],
            )

        # 2. Build prompt
        logger.info("[2/4] Building prompt with few-shot + CoT...")
        few_shot_text = format_few_shot(FEW_SHOT_EXAMPLES)
        prompt = build_prompt(question, chunks, few_shot_text)

        # Show prompt structure
        prompt_lines = prompt.split("\n")
        logger.info("  Prompt length: %s chars, %s lines", len(prompt), len(prompt_lines))
        logger.info("  Few-shot examples: %s", len(FEW_SHOT_EXAMPLES))
        logger.info("  Retrieved chunks in context: %s", len(chunks))

        # 3. Mock LLM response
        logger.info("[3/4] Generating answer (mock, no LLM)...")
        answer = mock_llm_answer(question, chunks)

        # 4. Result
        logger.info("[4/4] Done!")

        result = {
            "query": question,
            "answer": answer,
            "chunks_used": [
                {
                    "chunk_id": c["chunk_id"],
                    "source_file": c["source_file"],
                    "chunk_index": c["chunk_index"],
                    "similarity": c["similarity"],
                    "text_preview": c["text"][:200] + "...",
                }
                for c in chunks[:3]
            ],
            "prompt_preview": prompt[:500] + "...",
        }
        all_results.append(result)

        # Print answer
        logger.info("-" * 60)
        logger.info("ОТВЕТ (MOCK):")
        logger.info("-" * 60)
        logger.info("%s", answer)

    # Save results
    output_file = os.path.join(RESULTS_DIR, "rag_demo_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "demo (no LLM)",
            "few_shot_examples": FEW_SHOT_EXAMPLES,
            "system_prompt": SYSTEM_PROMPT,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Results saved to: %s", output_file)
    logger.info("Для реального запуска с LLM:")
    logger.info("  set OPENAI_API_KEY=sk-...")
    logger.info("  python rag_bot.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
