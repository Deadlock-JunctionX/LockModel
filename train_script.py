import os
import shutil

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from JXdataset import LABEL_LIST

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if os.path.isdir("/home/viethd/.cache/huggingface/datasets/j_xdataset/Junction/0.1.0/"):
    shutil.rmtree("/home/viethd/.cache/huggingface/datasets/j_xdataset/Junction/0.1.0/")
dataset = load_dataset(os.path.abspath("JXdataset.py"), "Junction")

print(dataset)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {k: v for k, v in enumerate(LABEL_LIST)}
label2id = {v: k for k, v in enumerate(LABEL_LIST)}


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="junction-test",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    save_total_limit=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    do_eval=True,
    do_train=True,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
