import requests
from app.ml.blip2_captioner import extract_caption_from_image
from app.ml.gemma_predictor import ask_gemma

def handle_post_categorization(post_text: str, image_url: str = None, hashtags: list[str] = None):
    caption_text = ""
    hashtag_text = ""

    if hashtags:
        hashtag_text = " ".join(f"#{tag}" for tag in hashtags if tag.strip())
        
    # 1. 이미지 설명 추출 (BLIP2 사용)
    if image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_bytes = response.content
                caption_text = extract_caption_from_image(image_bytes)
            else:
                print(f"[이미지 다운로드 실패] status_code: {response.status_code}")
        except Exception as e:
            print(f"[이미지 처리 오류] {e}")

    # 2. 전체 텍스트 구성
    full_text = post_text
    if caption_text:
        full_text += f"\n이미지 설명: {caption_text}"
    if hashtag_text:
        full_text += f"\n해시태그: {hashtag_text}"

    # 3. Gemma 모델에게 전체 문장 전달
    category, raw_output = ask_gemma(full_text, mode="post")

    return {
        "image": caption_text,
        "categoryName": category,
        "gemma_raw": raw_output
    }
