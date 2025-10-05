📌 Overview
This project fine-tunes a LLaMA-3B model to accurately predict product prices solely from text descriptions — no category, brand, or metadata. Surprisingly, the fine-tuned 3B model outperforms larger 70B models on both accuracy and robustness, proving that good domain adaptation can beat brute force.

🚀 Highlights
📉 Regression with LLMs: Designed prompts and outputs for continuous value prediction.

🛠️ Fine-tuning Pipeline: Used LoRA-based efficient fine-tuning on LLaMA-3B with product data.

🔍 Evaluation: Beats Machine learning models and llama 3.3 - 70B models by a very significant margin on custom test sets.

🧪 Frontier Testing: Tested for out-of-distribution generalization on unseen product types.

🧠 Context Engineering: Tuned input formatting, added price cues, controlled hallucination.
