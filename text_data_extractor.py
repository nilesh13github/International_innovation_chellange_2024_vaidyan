import torch
from transformers import pipeline
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download the NLTK punkt tokenizer models
nltk.download('punkt')

# Load the question generation model
model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
generator = pipeline("text2text-generation", model=model_name)


# Read the text from the file
with open('harrison.txt', 'r') as f:
    text = f.read()

# Split text into chunks (split based on newline after period)
text_chunks = text.split('.\n')

# Lists to store the generated questions and their corresponding context
q_list = []
context_list = []

# Process each chunk to generate summaries and questions
for chunk in text_chunks:
    if chunk.strip():  # Skip empty lines
        try:
            # Replace unwanted parts in the chunk
            
            # Summarization
            parser = PlaintextParser.from_string(chunk, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 1)  # Get 1-sentence summary
            
            # Make sure summary is available
            if summary:
                summarized_text = ' '.join(str(sentence) for sentence in summary)
                context_list.append(chunk)

                # Generate questions using the summary
                question = generator(f"generate question: {summarized_text}")[0]['generated_text']
                question=question.replace("question: ", "")
                q_list.append(question)
                print(question)
                
        except Exception as e:
            print(f"Error processing chunk: {e}")

# Create a DataFrame to store the input (context) and output (questions)
df = pd.DataFrame({'title': q_list, 'text': context_list})

# Save the DataFrame to a CSV file
df.to_csv('qna_1.csv', index=False)

print('done')

