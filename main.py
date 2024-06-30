from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat(input_text):
    # Tokenize input
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    
    # Generate response
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and return response
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chat(user_input)
    print("AI: ", response)