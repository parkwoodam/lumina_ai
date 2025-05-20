from PIL import Image
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import io

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

def extract_caption_from_image(image_bytes):
    from transformers import LlavaProcessor, LlavaForConditionalGeneration

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # LLaVA 전용 processor/model 불러오기
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16).to("cuda")

    # LLaVA에서 요구하는 프롬프트 구조
    prompt = "<image>\nUSER: 사진 속에서 어떤 활동이 벌어지고 있나요?\nASSISTANT:"

    # 텍스트 + 이미지 처리
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    output = model.generate(**inputs, max_new_tokens=50)
    response = processor.tokenizer.decode(output[0], skip_special_tokens=True)

    return response.strip()
