"""
Оценка качества RAG-бота через RAGAS (Задание 5, дополнение).
Использует тот же LLM-бэкенд, что и основной бот (openrouter / demo).
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Mapping

from Task4.rag_bot import rag_query, SYSTEM_PROMPT, call_llm, LLM_BACKEND, embedding_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ragas-eval")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task5_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Кастомный LLM-обёртка для RAGAS поверх нашего бэкенда ───
class RagasBackendLLM(LLM):
    """LangChain-совместимый LLM, который проксирует вызовы в наш call_llm."""

    backend: str = LLM_BACKEND
    n: int = 1

    @property
    def _llm_type(self) -> str:
        return f"rag_bot_proxy_{self.backend}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"backend": self.backend}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        answer = call_llm(SYSTEM_PROMPT, prompt, backend=self.backend)
        if stop:
            for s in stop:
                if s in answer:
                    answer = answer.split(s)[0]
        return answer

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, run_manager, **kwargs)


# ─── Кастомный embeddings-обёртка для RAGAS ───
class RagasBackendEmbeddings:
    """LangChain-совместимый embeddings, проксирующий в нашу e5 модель."""

    def __init__(self, model=embedding_model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]


# ─── Тестовые вопросы с ground-truth ответами из базы знаний ───
TEST_CASES = [
    {
        "question": "Who is Johnny Silverhand and what happened to him?",
        "ground_truth": (
            "Johnny Silverhand is a famous rockerboy and lead singer of the band Samurai. "
            "He was a military veteran who fought against megacorporations. "
            "In 2077, his digitized engram resides in V's brain as a result of the Relic implant."
        ),
    },
    {
        "question": "What is cyberware and how does it work?",
        "ground_truth": (
            "Cyberware is cybernetic technology permanently installed into the body, "
            "especially technology that interfaces with the central nervous system. "
            "It is used for practical upgrades, combat enhancement, and fashion."
        ),
    },
    {
        "question": "Which corporations operate in Night City?",
        "ground_truth": (
            "Night City is home to several major megacorporations, including Arasaka and Militech. "
            "These corporations wield significant power and influence over the city."
        ),
    },
    {
        "question": "Who is Adam Smasher and what is his role?",
        "ground_truth": (
            "Adam Smasher is a towering cyborg and a full-borg soldier employed by Arasaka. "
            "He serves as a ruthless enforcer for the corporation."
        ),
    },
    {
        "question": "What is the relationship between V and Jackie Welles?",
        "ground_truth": (
            "V and Jackie Welles are close friends and partners in crime. "
            "Jackie is a proud son of Heywood and works with V as a mercenary in Night City."
        ),
    },
]


def build_ragas_dataset():
    """Собирает ответы и контексты от RAG-бота для RAGAS."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in TEST_CASES:
        q = case["question"]
        gt = case["ground_truth"]
        logger.info("Processing question: %s", q)

        result = rag_query(q, top_k=5)
        answer = result["answer"]
        chunks = result["chunks_used"]
        chunk_texts = [c["text"] for c in chunks]

        questions.append(q)
        answers.append(answer)
        contexts.append(chunk_texts)
        ground_truths.append(gt)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    logger.info("=" * 70)
    logger.info("  RAGAS Evaluation for RAG Bot")
    logger.info("  Backend: %s", LLM_BACKEND)
    logger.info("=" * 70)

    # 1. Собираем датасет
    logger.info("Building evaluation dataset...")
    dataset = build_ragas_dataset()
    logger.info("Dataset ready: %s samples", len(dataset))

    # 2. Настраиваем LLM и embeddings для RAGAS
    proxy_llm = RagasBackendLLM(backend=LLM_BACKEND)
    ragas_llm = LangchainLLMWrapper(proxy_llm)

    proxy_emb = RagasBackendEmbeddings(model=embedding_model)
    ragas_embeddings = LangchainLLMWrapper(proxy_emb)  # для embeddings тоже wrapper

    # 3. Привязываем LLM и embeddings к метрикам
    faithfulness.llm = ragas_llm
    context_recall.llm = ragas_llm
    context_precision.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    # 4. Запускаем evaluate
    logger.info("Running RAGAS metrics...")
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    # 5. Сохраняем результаты
    scores_df = result.to_pandas()
    scores = scores_df.to_dict(orient="records")

    # Вычисляем средние значения по каждой метрике
    import numpy as np
    avg_scores = {}
    for m in metrics:
        if m.name in scores_df.columns:
            col = scores_df[m.name]
            # Игнорируем NaN
            valid = col.dropna()
            avg_scores[m.name] = float(valid.mean()) if len(valid) > 0 else 0.0

    output = {
        "backend": LLM_BACKEND,
        "metrics": avg_scores,
        "per_sample": scores,
    }

    output_file = os.path.join(RESULTS_DIR, f"ragas_results_{LLM_BACKEND}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("RAGAS Results:")
    for name, val in avg_scores.items():
        logger.info("  %s: %.3f", name, val)
    logger.info("=" * 70)
    logger.info("Results saved to: %s", output_file)


if __name__ == "__main__":
    main()
