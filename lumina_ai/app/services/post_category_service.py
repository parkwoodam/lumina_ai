import requests
from PIL import Image
from io import BytesIO
from app.ml.gemma_predictor_post import ask_gemma
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import io
import re
import pytesseract

# ✅ 전역 초기화: LLaVA 모델 및 프로세서 (한 번만 로드)
llava_model_id = "llava-hf/llava-1.5-7b-hf"
llava_processor = LlavaProcessor.from_pretrained(llava_model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_id, device_map="auto", torch_dtype=torch.float16
).to("cuda")

def clean_ocr_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)  # 한글, 영문, 숫자만 유지
    text = re.sub(r"\s{2,}", " ", text)  # 과도한 띄어쓰기 제거
    return text.strip()

def extract_caption_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # LLaVA 프롬프트
        prompt = """<image>
            USER: 이 이미지는 어떤 기부나 자원봉사 활동과 관련이 있을까요? 사진 속 **도움을 받는 대상(사람 또는 동물, 환경 등)**을 중심으로, 어떤 종류의 활동이 이뤄지고 있는지 간결하게 설명해주세요.
            기부 카테고리는 아동청소년|노인|장애인|지구촌|권익신장|시민사회|동물|환경|재난구휼 중에 하나와 관련되어있어야 합니다. 관련이 없다면 반드시 기타를 반환하세요.
            
            다음 기준을 바탕으로 구체적이고 사실적인 문장으로 답변해주세요:

            1. 도움이 필요한 **대상은 누구 또는 무엇**인가요? (예: 아동, 노인, 장애인, 유기동물, 환경, 재난상황, 외국인 등)
            2. 사진에서 유추되는 **활동의 성격**은 무엇인가요? (예: 돌봄, 교육, 구조, 배식, 청소, 기부 등)

            **중요: 인물 수, 나이, 배경 위치 같은 디테일보다는 '도움의 대상과 활동 내용'을 중심으로 묘사해주세요.**

            ASSISTANT: 응답은 반드시 한국어로 작성해주세요.
            """

        inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to("cuda")

        output = llava_model.generate(**inputs, max_new_tokens=300)
        raw_response = llava_processor.tokenizer.decode(output[0], skip_special_tokens=True)

        if "ASSISTANT:" in raw_response:
            response = raw_response.split("ASSISTANT:")[-1].strip()
        else:
            response = raw_response.strip()

        return response
    except Exception as e:
        print(f"[extract_caption_from_image 오류] {e}")
        return "사진 설명을 생성하지 못했습니다."

def extract_ocr_text(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_text = pytesseract.image_to_string(image, lang="kor+eng")
        return clean_ocr_text(raw_text)
    except Exception as e:
        print(f"[OCR 오류] {e}")
        return ""

def handle_post_categorization(post_text: str, image_url: str = None, hashtags: list[str] = None):
    caption_text = ""
    hashtag_text = ""
    ocr_text = ""

    # ✅ 해시태그 정리
    if hashtags:
        hashtag_text = " ".join(f"#{tag}" for tag in hashtags if tag.strip())

    # ✅ 이미지 설명 및 OCR 텍스트 추출
    if image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_bytes = response.content
                caption_text = extract_caption_from_image(image_bytes)
                ocr_text = extract_ocr_text(image_bytes)
            else:
                print(f"[이미지 다운로드 실패] status_code: {response.status_code}")
        except Exception as e:
            print(f"[이미지 처리 오류] {e}")

    # ✅ 전체 입력 텍스트 구성 (OCR 우선)
    full_text = ""
    if ocr_text:
        full_text += (
            "※ 아래 이미지에는 중요한 문구가 포함되어 있습니다. "
            "가장 우선적으로 이 문구를 기준으로 분류하세요.(이 다음에 오는 이미지 설명이 이미지 내 텍스트보다 무의미하다고 판단되면 무시하고 이미지 내 텍스트만을 기준으로 카테고리를 분류하세요.)\n"
            f"이미지 내 텍스트: {ocr_text}\n"
        )
    if caption_text:
        full_text += f"이미지 설명: {caption_text}\n"
    full_text += f"글 설명: {post_text.strip()}\n"
    if hashtag_text:
        full_text += f"해시태그 설명: {hashtag_text}"

    # ✅ Gemma 모델로 분류 요청
    category, raw_output = ask_gemma(full_text, mode="post")

    return {
        "image": caption_text,
        "ocr_text": ocr_text,
        "categoryName": category,
        "gemma_raw": raw_output
    }
