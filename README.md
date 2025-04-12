Main Fine-tuned llm model with Dataset in uploaded on -> https://huggingface.co/Nilesh213/vaidyan_final/tree/main/Nilesh213/vaidyan_final

## encoder.py
- Loads text data from a CSV file.
- Uses `SentenceTransformer` to encode each text into a dense vector representation.
- Outputs a new CSV (`ganda_2.csv`) containing the original text along with their corresponding embeddings.
- Includes a progress bar for monitoring large-scale text processing.


## app.py 
