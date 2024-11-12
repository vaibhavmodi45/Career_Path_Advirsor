# Career Path Advisor/model.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric

def load_and_preprocess_data():
    # Load data files
    questions = pd.read_csv("data/questions.csv")
    question_tags = pd.read_csv("data/question_tags.csv")
    tags = pd.read_csv("data/tags.csv")
    
    # Merge questions with tags
    merged_data = questions.merge(question_tags, left_on="id", right_on="question_id")
    merged_data = merged_data.merge(tags, left_on="tag_id", right_on="id")
    
    # Select and preprocess columns
    merged_data['text'] = merged_data['question_title'] + " " + merged_data['question_body']
    merged_data['category'] = merged_data['tag_name']
    
    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_pandas(merged_data[['text', 'category']])
    
    return dataset

def train_model(dataset):
    # Initialize the tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(dataset.unique('category'))
    )

    # Tokenize the dataset
    def preprocess_data(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    # Define metric for evaluation
    metric = load_metric("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("CARRER_PATH_ADVISOR")
    tokenizer.save_pretrained("CARRER_PATH_ADVISOR")

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    train_model(dataset)