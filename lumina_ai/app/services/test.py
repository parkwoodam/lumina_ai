from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import requests
from io import BytesIO

# ✅ 올바른 모델 ID
model_id = "llava-hf/llava-1.5-13b-hf"

# 모델 로드
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 이미지 로딩
image_url = "https://s3-lumina-bucket.s3.ap-northeast-2.amazonaws.com/post/0304_01.jpg"
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

# 프롬프트
prompt = "<image> Describe this image thoroughly and accurately. Include details about objects, people, actions, emotions, colors, background, and any relevant context. Be concise but comprehensive, as if you are explaining this to someone who cannot see the image."

# 추론
inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=150)
print(processor.decode(output[0], skip_special_tokens=True))
