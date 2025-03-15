from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class ImageCaptioningLLM:
    MODEL_NAME = "trained_florence2"

    def __init__(self):
        print("Loading trained model...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME).to("cuda")

    def generate_caption_from_path(self, image_path) -> str:
        image = Image.open(image_path)
        return self.generate_caption(image)

    def generate_caption(self, image: Image) -> str:
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)

# Test
if __name__ == "__main__":
    llm = ImageCaptioningLLM()
    print(llm.generate_caption_from_path("data/images/test.jpg"))