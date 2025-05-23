"""
Capstone: Fine-Tuning a Language Model for Emotion Classification (Batch Script)
This script is designed to be run as a SLURM batch job for HPC environments.
"""

import os
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch

def main():
    # Print device info for clarity
    print("Torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))

    # 1. Load and Explore the Dataset
    dataset = load_dataset('emotion')
    print("Sample:", dataset['train'][0])
    print("Labels:", dataset['train'].features['label'].names)
    print("Train size:", len(dataset['train']))
    print("Validation size:", len(dataset['validation']))
    print("Test size:", len(dataset['test']))

    # 2. Tokenize the Dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding='max_length')
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(['text'])
    tokenized.set_format('torch')

    # 3. Load Pretrained Model
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=6
    )

    # 4. Fine-Tune the Model
    args = TrainingArguments(
        output_dir='./results',
        eval_strategy='steps',
        eval_steps=100,
        logging_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=64,  # Increased batch size for A100
        per_device_eval_batch_size=128,  # Increased eval batch size for A100
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy='steps',
        save_steps=500,
        report_to=[],  # disables wandb etc.
        fp16=True,  # Enable mixed precision for A100
        dataloader_num_workers=4,  # Faster data loading
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    # 5. Evaluate and Save the Model
    eval_results = trainer.evaluate(tokenized['test'])
    print("Test set evaluation:", eval_results)
    model.save_pretrained('./trained_emotion_model')
    tokenizer.save_pretrained('./trained_emotion_model')

    # 6. Predict on a sample
    def predict_emotion(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1).squeeze().tolist()
        labels = dataset['train'].features['label'].names
        predicted = labels[probs.index(max(probs))]
        print(f"\n📝 Input: {text}")
        print(f"🤖 Predicted Emotion: {predicted} ({max(probs)*100:.2f}% confidence)")
        print(f"📊 Probabilities: {dict(zip(labels, [f'{p:.3f}' for p in probs]))}")

    predict_emotion("I can't believe you did that!")

if __name__ == "__main__":
    main()
