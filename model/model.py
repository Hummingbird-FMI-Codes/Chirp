from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

prompt = "NOTHING"

class ImageCaptioningLLM:
    MODEL_NAME = "trained_florence2"

    def __init__(self):
        print("Loading trained model...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to("cpu")

    def generate_caption_from_path(self, image_path) -> str:
        image = Image.open(image_path).convert("RGB")
        return self.generate_caption(image)

    def generate_caption(self, image: Image) -> str:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cpu")
        output = self.model.generate(**inputs, max_length=200, num_return_sequences=1)
        return self.processor.decode(output[0], skip_special_tokens=True)

# Test
if __name__ == "__main__":
    llm = ImageCaptioningLLM()
    print(llm.generate_caption_from_path("data/images/test.jpg"))