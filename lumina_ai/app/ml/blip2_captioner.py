from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import io
import torch

# 모델 로드
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

@torch.inference_mode()
def extract_caption_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        question = "이 이미지에서 어떤 일이 일어나고 있나요?"
        inputs = blip_processor(image, question=question, return_tensors="pt").to("cuda")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"BLIP 이미지 설명 오류: {e}")
        return ""
