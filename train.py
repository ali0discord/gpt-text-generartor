import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from model import load_model
from database import fetch_all_inputs, clear_database  # مدیریت دیتابیس
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",  # پُر کردن توکن‌ها تا طول مشخص
            max_length=self.max_length,
            return_tensors="pt"
        )
        attention_mask = encodings.attention_mask.squeeze(0)
        return encodings.input_ids.squeeze(0), attention_mask

def train_model_with_text(selected_model, custom_text, epochs, batch_size):
    """
    آموزش مدل با متن سفارشی.
    """
    model, tokenizer = load_model(selected_model)
    dataset = TextDataset([custom_text], tokenizer)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    _train_model(model, tokenizer, dataloader, epochs, selected_model, "custom_text")

def train_model_with_database(selected_model, epochs, batch_size):
    """
    آموزش مدل با داده‌های موجود در دیتابیس.
    """
    model, tokenizer = load_model(selected_model)
    inputs_data = fetch_all_inputs()
    texts = [input_text for input_text, model_name in inputs_data if model_name == selected_model]

    if not texts:
        print("Error: No data found in the database for the selected model.")
        return

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    _train_model(model, tokenizer, dataloader, epochs, selected_model, "database")
    clear_database()

def train_model_with_dataset(selected_model, epochs, batch_size, dataset_path):
    """
    آموزش مدل با فایل دیتاست آپلود‌شده.
    """
    model, tokenizer = load_model(selected_model)

    # خواندن دیتاست
    with open(dataset_path, "r", encoding="utf-8") as f:
        texts = f.readlines()

    if not texts:
        print("Error: Dataset is empty.")
        return

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    _train_model(model, tokenizer, dataloader, epochs, selected_model, "dataset")

def _train_model(model, tokenizer, dataloader, epochs, model_name, method):
    """
    منطق مشترک آموزش مدل.
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # انتقال مدل به GPU در صورت وجود
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # محاسبه خروجی و خطا
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # ذخیره مدل
    save_path = f"trained_{model_name}_{method}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model {model_name} trained with {method} and saved to {save_path}.")

def train_model_with_hf_dataset(selected_model, epochs, batch_size, dataset_name, split="train"):
    """
    آموزش مدل با استفاده از دیتاست‌های Hugging Face.
    
    Args:
        selected_model (str): نام مدل برای آموزش.
        epochs (int): تعداد epochs.
        batch_size (int): اندازه batch.
        dataset_name (str): نام دیتاست در Hugging Face.
        split (str): بخش دیتاست برای بارگذاری (train, test, validation).
    """
    model, tokenizer = load_model(selected_model)

    # بارگذاری داده‌ها از Hugging Face
    texts = load_dataset(dataset_name, split)
    
    if not texts:
        print(f"Error: Dataset {dataset_name} ({split} split) is empty or invalid.")
        return

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    _train_model(model, tokenizer, dataloader, epochs, selected_model, f"huggingface_{dataset_name}")
