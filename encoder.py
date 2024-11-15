import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Download the correct NLTK tokenizer
nltk.download('punkt')

df = pd.read_csv('vectData.csv')

title = df['title'].to_list()
text = df['text'].to_list()

embeddings_list = []

print('1 done')

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print('2 done')

# Generate embeddings for each text with a progress bar
for i in tqdm(range(len(text)), desc="Processing texts", unit="text"):
    # Pass the list of texts to model.encode() instead of an integer
    embeddings = model.encode([text[i]], show_progress_bar=True)
    embeddings_list.append(embeddings[0]) 
    print(embeddings[0]) # embeddings is a 2D array, take the first element
    # Uncomment the next line if you want to see each embedding
    #print(embeddings[0])
    break

print(len(text))
print(len(embeddings_list))

df_output = pd.DataFrame({'title': title, 'text': text, 'embeddings': embeddings_list})
df_output.to_csv('ganda_2.csv', index=False)

print('5 done')
