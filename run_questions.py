"""
run_questions.py - Project 7 Benchmark Questions Runner for Mirai RAG Assistant

Executes the 5 benchmark evaluation questions from Project 7:
- Q1: Model pre-training hyperparameters (Found in Section 4.2 / Page 16)
- Q2: Current CEO of OpenAI (Out of domain -> strictly grounded to '[not found]')
- Q3: Benchmark Exact Match (EM) accuracy gains (Found in Section 5.3 / Page 31)
- Q4: GPU cluster configuration & training days (Found in Section 7.1 / Page 45)
- Q5: Box-office revenue of Oppenheimer (Out of domain -> strictly grounded to '[not found]')

Prints questions, retrieved answers, and verifiable source citations.
"""

from rag_pipeline import answer

# The 5 Benchmark Questions from Project 7
QUESTIONS = [
    {
        "id": "Q1",
        "question": "What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4?"
    },
    {
        "id": "Q2",
        "question": "Who is the current Chief Executive Officer of OpenAI?"
    },
    {
        "id": "Q3",
        "question": "By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline?"
    },
    {
        "id": "Q4",
        "question": "What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model?"
    },
    {
        "id": "Q5",
        "question": "What was the worldwide box-office revenue of the film Oppenheimer in 2023?"
    }
]


def run_benchmark():
    """
    Runs each benchmark question through the RAG pipeline
    and prints formatted results with answer grounding and sources.
    """
    print("=" * 80)
    print("   MIRAI RAG RESEARCH ASSISTANT - BENCHMARK EVALUATION (PROJECT 7)")
    print("=" * 80)

    for item in QUESTIONS:
        q_id = item["id"]
        q_text = item["question"]

        print(f"\n[{q_id}] Question:")
        print(f"    {q_text}")

        # Call the exact required function: answer(question) -> dict
        result = answer(q_text)

        ans_text = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"--> Answer:")
        print(f"    {ans_text}")
        print(f"--> Sources:")
        print(f"    {sources}")
        print("-" * 80)

    print("\nBenchmark run completed successfully.")


if __name__ == "__main__":
    run_benchmark()
