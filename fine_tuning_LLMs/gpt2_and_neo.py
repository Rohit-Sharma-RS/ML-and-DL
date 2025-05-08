from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_gpt2(dataset_path, output_dir="./fine_tuned_gpt2", num_train_epochs=3, batch_size=2, learning_rate=2e-5):
    # Load dataset
    dataset = load_dataset("text", data_files={"train": dataset_path})

    # Load GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids"])

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_steps=1000,
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("GPT-2 model fine-tuned and saved to", output_dir)


def fine_tune_gpt_neo(dataset_path, output_dir="./fine_tuned_gpt_neo", num_train_epochs=3, batch_size=2, learning_rate=2e-5):
    # Load dataset
    dataset = load_dataset("text", data_files={"train": dataset_path})

    # Load GPT-Neo model and tokenizer
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids"])

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_steps=1000,
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("GPT-Neo model fine-tuned and saved to", output_dir)


# Fine-tune GPT-2 on a custom dataset
fine_tune_gpt2("path/to/custom_dataset.txt")

# Fine-tune GPT-Neo on a custom dataset
fine_tune_gpt_neo("path/to/custom_dataset.txt")
