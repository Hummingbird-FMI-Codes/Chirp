from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

prompt = "Describe what you see"

class ImageCaptioningLLM:
    MODEL_NAME = "trained_florence2"

    def __init__(self):
        print("Loading trained model...")
        self.allowed_keywords = {"ANTS": [" ants ", " ant ", " anthill "]}
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to("cpu")

    def generate_caption_from_path(self, image_path) -> [str]:
        image = Image.open(image_path).convert("RGB")
        return self.generate_caption(image)

    def generate_caption(self, image: Image) -> [str]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cpu")
        output = self.model.generate(**inputs, max_length=200, num_return_sequences=1)
        return self.processor.decode(output[0], skip_special_tokens=True)

    def get_label(self, image: Image) -> str:
        caption = self.generate_caption(image)
        for key in self.allowed_keywords:
            for allowed in self.allowed_keywords[key]:
                if allowed.lower() in caption.lower():
                    print(f"{key}: {caption}")
                    return key

        return "NOTHING"

    def get_label_from_caption(self, path: str) -> str:
        caption = self.generate_caption_from_path(path)
        for key in self.allowed_keywords:
            for allowed in self.allowed_keywords[key]:
                if allowed in caption:
                    print(f"{allowed}: {caption}")
                    return key

        return "NOTHING"


# Test
if __name__ == "__main__":
    llm = ImageCaptioningLLM()
    print(llm.generate_caption_from_path("data/images/test.jpg"))