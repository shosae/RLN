# RLN Playground Brief

RLN is a small research-oriented workspace used to explore Retrieval Augmented Generation workflows.  Experiments normally focus on langgraph compositions plus Llama family models.  The core requirements for each prototype are:

- Always route user questions through a retriever so we can track context usage.
- Prefer lightweight vector stores such as FAISS so tests can run entirely on a laptop.
- Keep prompts transparent and easy to modify from code.

## Initial Knowledge Base

1. RLN stands for "Research Lab Notebook" and targets single-user experiments.
2. The baseline documents are short markdown notes kept under `data/seed`.
3. Each run should explain where the answer came from by echoing key source titles.
4. Llama 3.1 8B Instruct is the preferred chat model for generation steps.

