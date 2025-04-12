Main Fine-tuned llm model with Dataset in uploaded on -> https://huggingface.co/Nilesh213/vaidyan_final/tree/main/Nilesh213/vaidyan_final

## encoder.py
- Loads text data from a CSV file.
- Uses `SentenceTransformer` to encode each text into a dense vector representation.
- Outputs a new CSV (`ganda_2.csv`) containing the original text along with their corresponding embeddings.
- Includes a progress bar for monitoring large-scale text processing.


## trainer.py

# Fine-Tuning Mistral-7B-Instruct with LoRA

This script fine-tunes the `mistralai/Mistral-7B-Instruct-v0.3` model using the `SFTTrainer` class with Low-Rank Adaptation (LoRA). The training dataset is loaded from a CSV file and fine-tuned for causal language modeling.

## 🚀 Features

- Loads a pre-trained model (`mistralai/Mistral-7B-Instruct-v0.3`).
- Fine-tunes the model using the `SFTTrainer` with LoRA for efficient training.
- Uses mixed precision training (FP16 or BF16) for faster training.
- Saves the fine-tuned model and tokenizer for later use.
