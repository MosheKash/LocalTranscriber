import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

# Configuration
MODEL_PATH = "./llama-3.2-3b-hf"  # Update to your Instruct model path
MAX_NEW_TOKENS = 2048  # Reasonable limit for chat responses
TEMPERATURE = 0.7  # Lower = more focused, higher = more creative
TOP_P = 0.9  # Nucleus sampling
REPETITION_PENALTY = 1.2  # Increased to combat repetition

def load_model():
    """Load the model and tokenizer with memory optimization."""
    print("Loading model and tokenizer...")
    
    # Load tokenizer with regex fix
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        fix_mistral_regex=True
    )
    
    # Set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set the Llama 3 chat template manually if not present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    print(f"Model loaded successfully on {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, messages):
    """Generate a response using the chat template."""
    
    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Set up terminators - tell model when to stop
    terminators = [
        tokenizer.eos_token_id,  # End of sequence token
    ]
    
    # Add Llama 3 specific stop tokens
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        # Try to get the end-of-turn token ID
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id is not None and eot_token_id != tokenizer.unk_token_id:
            terminators.append(eot_token_id)
        
        # Also try the end header token
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        if end_header_id is not None and end_header_id != tokenizer.unk_token_id:
            terminators.append(end_header_id)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=3,
            eos_token_id=terminators,  # Use multiple stop tokens
        )
    generation_time = time.time() - start_time
    
    # Calculate tokens generated
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )
    
    # Print stats
    print(f"\n[Generated {tokens_generated}/{MAX_NEW_TOKENS} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)]")
    
    return response.strip()

def main():
    """Main conversation loop."""
    print("=" * 60)
    print("Llama 3.2 3B Instruct Conversational Test")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model()
    
    print("\nModel ready! Type 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to start a new conversation.\n")
    
    messages = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            messages = []
            print("Conversation cleared!\n")
            continue
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        response = generate_response(model, tokenizer, messages)
        print(response)
        
        # Add assistant response
        messages.append({"role": "assistant", "content": response})
        
        # Keep last 20 messages (10 exchanges) to manage context
        if len(messages) > 20:
            messages = messages[-20:]
        
        print()

if __name__ == "__main__":
    main()