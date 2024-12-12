import torch

seed = 42
def generate_text(models, model_name, input_text, max_new_token, seed=None):
    selected_model = models[model_name]
    tokenizer = selected_model["tokenizer"]
    model = selected_model["model"]

    # تنظیم pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # تنظیم seed برای خروجی ثابت
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # تولید input_ids و attention_mask
    encodings = tokenizer(
        input_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512  # تنظیم طول حداکثر برای ورودی
    )
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    try:
    # تولید متن با جلوگیری از تکرار
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_token,  # تعداد توکن‌های جدید تولید شده
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
    except TypeError as e:
        raise ValueError(f"error in model.generate {e}")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_code(models, model_name, prompt, max_new_tokens, seed=None):
    """
    Generate code based on the provided prompt using a code-specific model.
    """
    selected_model = models[model_name]
    tokenizer = selected_model["tokenizer"]
    model = selected_model["model"]

    # تنظیم seed برای خروجی ثابت
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # توکنایز کردن ورودی
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # ایجاد attention mask
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)  # ایجاد یک ماسک توجه برای ورودی‌ها

    try:
    # تولید کد
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # ارسال attention mask
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,  # تنظیم شناسه توکن پایان به عنوان پرکننده
            repetition_penalty=1.2,  # جلوگیری از تکرار
            no_repeat_ngram_size=2,  # جلوگیری از تکرار n-gram
        )
    except TypeError as e:
        raise ValueError(f"error in model.generate {e}")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)