import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from model import load_model

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

        # Ensure attention_mask is included in the return
        attention_mask = encodings.attention_mask.squeeze(0)

        # Return both input_ids and attention_mask
        return encodings.input_ids.squeeze(0), attention_mask

def train_model_with_text(model_name, custom_text, epochs, batch_size):
    model, tokenizer = load_model(model_name)

    # Prepare dataset with custom text
    dataset = TextDataset([custom_text], tokenizer)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # Save trained model
    model.save_pretrained(f"trained_{model_name}_custom")
    tokenizer.save_pretrained(f"trained_{model_name}_custom")
    print(f"Model {model_name} trained with custom text and saved.")

def evaluate_model(model, tokenizer, input_texts, true_texts, max_length=50):
    """
    Evaluate the model using BLEU score and F1 score.
    """
    model.eval()

    predicted_texts = []
    
    for input_text in input_texts:
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_texts.append(predicted_text)

    # BLEU Score Calculation
    bleu_scores = [sentence_bleu([true.split()], pred.split()) for true, pred in zip(true_texts, predicted_texts)]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    # F1 Score Calculation (for simplicity, we compare exact matches)
    f1_scores = [f1_score([true == pred], [1]) for true, pred in zip(true_texts, predicted_texts)]
    avg_f1_score = sum(f1_scores) / len(f1_scores)

    return avg_bleu_score, avg_f1_score