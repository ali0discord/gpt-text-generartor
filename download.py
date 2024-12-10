from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def download_and_save_model(model_name, save_dir):
    """
    Download a model and tokenizer from Hugging Face and save it locally in the specified directory.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Download and load the model and tokenizer
    print(f"Downloading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=save_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=save_dir)

    # Save the model and tokenizer locally
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model and tokenizer for {model_name} saved to {save_dir}.")

# Example usage
model_name = "bolbolzaban/gpt2-persian"  # You can change this to any model name, e.g., "gpt2-medium", "flax-community/gpt2-medium-persian"
save_dir = "./models/bolbolzaban"  # Specify the local directory to save the model
download_and_save_model(model_name, save_dir)
