"""
A simple CLI chatbot using GPT-2 hosted locally with improved response handling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flatcar_ai_gpu_demo_pytorch.flatcar_ai_gpu_demo_pytorch import get_device_info


def chatbot_cli():
    """A simple CLI chatbot using GPT-2 hosted locally with improved response handling."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    device = get_device_info()
    model.to(device)
    print(f"Running on {device}. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Encode user input and set attention mask
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        # Generate a response with sampling
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Chatbot: {response}")
