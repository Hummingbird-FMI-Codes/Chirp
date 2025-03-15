import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor

def process_data(processor):
    def __process_data(examples):
        images = [Image.open(f"data/images/{img}").convert("RGB") for img in examples["image"]]
        inputs = processor(images=images, text=examples["caption"], padding="max_length", truncation=True, return_tensors="pt")
        return inputs
    return __process_data


def get_tokenized_dataset(processor):
    df = pd.read_csv("../dataset/dataset.csv")
    dataset = Dataset.from_pandas(df)
    return dataset.map(process_data(processor), batched=True)