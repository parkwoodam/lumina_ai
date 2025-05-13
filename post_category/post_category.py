from fastapi import FastAPI, UploadFile, File, Form
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import io

app = FastAPI()

# BLIP2 모델 로딩
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

# Gemma 모델 로딩 플래그
model_loaded = False
gemma_model = None
gemma_tokenizer = None

def load_model():
    global model_loaded, gemma_model, gemma_tokenizer
    if not model_loaded:
        model_id = "google/gemma-3-4b-it"
        gemma_tokenizer = AutoTokenizer.from_pretrained(model_id)
        gemma_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model_loaded = True

@torch.inference_mode()
def ask_gemma_for_post(text: str):
    load_model()
    prompt = f"""
당신은 기부 카테고리를 분류하는 전문가입니다.

아래 글은 어떤 기부 카테고리에 가장 적절한지 판단하세요.  
카테고리가 명확하지 않거나 애매하다면 반드시 '기타'로 답변하세요.
아래 글과 이미지 설명을 바탕으로, **피해자 보호**, **인권 침해**, **차별**, **약자 보호**에 해당한다면 반드시 '권익신장'으로 분류하세요.
특히 폭력, 따돌림, 괴롭힘 등은 '아동청소년' 또는 '권익신장'으로 분류해야 합니다.

카테고리:
- 아동청소년
- 노인
- 장애인
- 지구촌
- 권익신장
- 시민사회
- 동물
- 환경
- 재난구휼
- 기타

글:
{text}

정답:"""
    inputs = gemma_tokenizer(prompt.strip(), return_tensors="pt").to("cuda")
    outputs = gemma_model.generate(**inputs, max_new_tokens=30)
    decoded = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"정답[:：]?\s*(아동청소년|노인|장애인|지구촌|권익신장|시민사회|동물|환경|재난구휼|기타)", decoded)
    category = match.group(1) if match else "기타"
    return category, decoded.strip()

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

@app.post("/post-categorize")
def categorize_post(postContent: str = Form(...), postImageFile: UploadFile = File(None)):
    caption_text = ""

    if postImageFile:
        try:
            image_bytes = postImageFile.file.read()
            if image_bytes:
                caption_text = extract_caption_from_image(image_bytes)
            else:
                print("이미지 파일이 비어 있음.")
        except Exception as e:
            print(f"이미지 처리 오류: {e}")

    full_text = postContent
    if caption_text:
        full_text += f"\n이미지 설명: {caption_text}"

    text_cat, raw = ask_gemma_for_post(full_text)

    if text_cat != "기타":
        recommended = text_cat
    elif image_cat != "기타":
        recommended = image_cat
    else:
        recommended = "기타"

    return {
        "image":caption_text,
        "categoryname": recommended,
        "gemma_raw": raw
    }
