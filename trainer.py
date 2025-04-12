import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from trl import SFTTrainer
from peft import LoraConfig

model_name = 'vaid_v3'  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

data = load_dataset("csv", data_files="prompt_data_mistral.csv", split="train")         


print("working 3")



print("working 3.5")



model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto")




tokenizer = AutoTokenizer.from_pretrained(model_name) #, use_fast=False

print("working 5")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'

print("working 6")
fp16 = True
bf16 = False

print("working 7")

training_args = TrainingArguments(
    output_dir="vaid_v3//checkPoints", 
    per_device_train_batch_size=1, 
    num_train_epochs=1,
    learning_rate=2e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    weight_decay=0.001,
    fp16=fp16,
    bf16=bf16,
    save_steps= 1000,

)

print("working 8")

qlora_cfg = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)



print("working 9")

trainer=SFTTrainer(
    model=model,
    train_dataset=data,
    dataset_text_field="text",
    peft_config=qlora_cfg,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("working 10")

trainer.train()
trainer.model.save_pretrained('vaid_v3')
tokenizer.save_pretrained('vaid_v3//tokenizer')

print("------DONE------")
