from transformers import AutoModel, AutoTokenizer

# Path to the local directory containing the saved model
bert_path = "/opt/models/bert-base-uncased"
distil_path = "/opt/models/distilgpt2"

model_path = bert_path

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Test the model with a sample input
input_text = "Hello, Hugging Face!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

print("Model output:", outputs)
