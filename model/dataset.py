import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def process_data(processor):
    def __process_data(examples):
        images = [Image.open(f"dataset/images/{img}").convert("RGB") for img in examples["image"]]
        inputs = processor(images=images, text=examples["caption"], padding="max_length", truncation=True, return_tensors="pt")
        inputs["decoder_input_ids"] = inputs["input_ids"].clone()

        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "decoder_input_ids": inputs["decoder_input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"].clone(),
        }
    return __process_data


def get_tokenized_dataset(processor):
    df = pd.read_csv("dataset/dataset.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset = DatasetDict({
        "train": train_dataset.map(process_data(processor), batched=True),
        "validation": val_dataset.map(process_data(processor), batched=True)
    })
    print(dataset)
    return dataset