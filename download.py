import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# لیست مدل‌ها با مسیر ذخیره مشخص‌شده
MODEL_LIST = {
    "gpt2": {"path": "openai-community/gpt2", "save_dir": "./models/gpt2"},
    "gpt2-medium": {"path": "openai-community/gpt2-medium", "save_dir": "./models/gpt2-medium"},
    "gpt2-persian": {"path": "flax-community/gpt2-medium-persian", "save_dir": "./models/gpt2-medium-persian"},
    "codegen": {"path": "Salesforce/codegen-350M-mono", "save_dir": "./models/codegen"}
}

def download_and_save_models():
    """
    دانلود و ذخیره تمام مدل‌ها در مسیرهای مشخص‌شده.
    """
    for model_name, model_info in MODEL_LIST.items():
        model_path = model_info["path"]  # مسیر مدل در Hugging Face
        save_dir = model_info["save_dir"]  # مسیر ذخیره مدل
        
        print(f"Downloading and saving model: {model_name} to folder: {save_dir}")
        
        if not os.path.exists(save_dir):  # بررسی اینکه آیا فولدر ذخیره وجود دارد یا نه
            os.makedirs(save_dir, exist_ok=True)
            
            # دانلود و ذخیره مدل
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            print(f"Model {model_name} saved to {save_dir}")
        else:
            print(f"Model {model_name} already exists in {save_dir}")

if __name__ == "__main__":
    download_and_save_models()
