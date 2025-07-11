import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch


file_path = './bert.xlsx'
df = pd.read_excel(file_path, header=None)
df.columns = ['Задание', 'Ответ', 'Оценка']
df['Оценка'] = df['Оценка'].astype(int)


min_label = df['Оценка'].min()
df['Оценка'] = df['Оценка'] - min_label
num_labels = df['Оценка'].nunique()


df['text'] = "Задание №" + df['Задание'].astype(str) + " " + df['Ответ'].astype(str)


train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_data.reset_index(drop=True))


model_name = 'DeepPavlov/rubert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


train_dataset = train_dataset.rename_column("Оценка", "labels")
test_dataset = test_dataset.rename_column("Оценка", "labels")
train_dataset = train_dataset.remove_columns(["Задание", "Ответ", "text"])
test_dataset = test_dataset.remove_columns(["Задание", "Ответ", "text"])

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    do_eval=True,
    do_train=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()
results = trainer.evaluate()

print("Результаты на тестовом наборе:")
print(results)
