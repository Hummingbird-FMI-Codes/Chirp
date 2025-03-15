import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import default_collate
from transformers import TrainingArguments, Trainer, AutoProcessor, AutoModelForCausalLM

from dataset import get_tokenized_dataset


def data_collator(features):
    for feature in features:
        feature["pixel_values"] = torch.tensor(feature["pixel_values"])

    return default_collate(features)

def train_model():
    MODEL_NAME = "microsoft/Florence-2-base"
    TRAINED_MODEL = "trained_florence2"

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cpu")

    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        save_steps=1000,
        evaluation_strategy="epoch"
    )

    tokenized_dataset = get_tokenized_dataset(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()

    model.save_pretrained(TRAINED_MODEL)
    processor.save_pretrained(TRAINED_MODEL)


if __name__ == "__main__":
    train_model()