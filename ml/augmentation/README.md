# ml/augmentation/

Phase 2. Client-adaptive synthetic data generation: `ctgan_engine` (SDV's `CTGANSynthesizer`) for data-rich Banks A/B (Augment Mode), `llm_engine` (Groq API, probability-aware prompting per D1) for data-poor Banks C/D (Schema Mode). Never sends raw rows to the LLM API — aggregate statistics and few-shot examples only.
