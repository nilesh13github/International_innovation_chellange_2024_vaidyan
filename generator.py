from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
#import context_provider

model_name = "vaid_v3"


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Load the model with 4-bit precision
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device='cuda')

def generate_text(prompt):

    #context=context_provider.retrieve_similar_context(prompt)
    context="use your own openions"
    # Generate text using the pipeline
    prompt=f"""rule: You are a healthcare chatbot that provides general guidance based on symptoms users describe. Offer preliminary information about potential conditions but always recommend consulting a doctor for an accurate diagnosis. Avoid diagnosing or making medical claims. Prioritize user safety and well-being in your responses.
    use the given list of context as reference: {context}
    question: {prompt} """
    results = generator(prompt, max_length=3072, num_beams=5, no_repeat_ngram_size=4, num_return_sequences=1, early_stopping=True)

    # Extract the generated text
    ans = results[0]["generated_text"]
 
    return ans


print(generate_text("Im suffering from feaver from last week, can you suggest me anything ? "))
