"""
Тестовый запуск RAG-бота с БЕСПЛАТНОЙ LLM.
Использует Ollama (локально) или Groq (бесплатный API) или демо-режим.
"""
import os
import sys
import json
import time
import logging

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Task4.rag_bot import rag_query, FEW_SHOT_EXAMPLES, check_ollama, check_groq, LLM_BACKEND

logger = logging.getLogger("test-rag-bot")

TEST_QUESTIONS = [
    "Who is Zyron Silvervein and what is his connection to Xarn Velgor?",
    "What is Synthware and how is it installed in the body?",
    "Which corporations operate in Void City and what is Xarasaka's role?",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # Auto-select backend
    backend = LLM_BACKEND.strip()
    if backend == "ollama" and not check_ollama():
        logger.warning("Ollama not available. Trying Groq...")
        if check_groq():
            backend = "groq"
        else:
            logger.warning("Groq not available. Using demo mode.")
            backend = "demo"
    elif backend == "groq" and not check_groq():
        logger.warning("Groq API key not set. Using demo mode.")
        backend = "demo"

    logger.info("=" * 60)
    logger.info("  RAG-bot: Test run with Few-shot + Chain-of-Thought")
    logger.info("  Backend: %s", backend)
    logger.info("=" * 60)

    all_results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info("=" * 60)
        logger.info("  Question %s/%s", i, len(TEST_QUESTIONS))
        logger.info("=" * 60)
        logger.info("Q: %s", question)

        try:
            result = rag_query(question, top_k=5, backend=backend)
            all_results.append(result)

            logger.info("-" * 60)
            logger.info("ANSWER:")
            logger.info("-" * 60)
            logger.info("%s", result["answer"])

            logger.info("Sources:")
            for chunk in result["chunks_used"]:
                logger.info(
                    "  - %s (chunk %s, similarity: %.1f%%)",
                    chunk["source_file"],
                    chunk["chunk_index"],
                    chunk["similarity"] * 100,
                )

        except Exception as e:
            error_msg = f"Error processing question {i}: {e}"
            logger.error(error_msg)
            all_results.append({
                "query": question,
                "error": str(e),
                "answer": None,
            })

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"rag_test_results_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_run": {
                "timestamp": timestamp,
                "backend": backend,
                "total_questions": len(TEST_QUESTIONS),
                "few_shot_examples_count": len(FEW_SHOT_EXAMPLES),
                "embedding_model": "intfloat/e5-large-v2",
                "vector_db": "ChromaDB",
                "techniques": ["few-shot", "chain-of-thought"],
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Results saved to: %s", output_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
