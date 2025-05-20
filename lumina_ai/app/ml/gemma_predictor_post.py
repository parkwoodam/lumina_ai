from app.ml.model_loader import get_gemma
import torch
import re

@torch.inference_mode()
def ask_gemma(text: str, mode: str = "keyword") -> tuple[str, str]:
    tokenizer, model = get_gemma()

    if mode == "keyword":
        task_text = f"키워드: {text}"
    elif mode == "post":
        task_text = f"설명 - {text}"
    else:
        raise ValueError(f"지원하지 않는 모드: {mode}")

    prompt = f"""
    당신은 기부 활동을 분류하는 전문가입니다.

    아래의 글을 읽고, 활동에서 **도움을 받는 대상(수혜자)**이 누구인지 판단하여,
    다음 10가지 카테고리 중 **가장 적절한 하나만** 고르세요.

    도움을 주는 사람(예: 자원봉사자, 청소년 등)은 고려하지 마세요.
    오직 **도움을 받는 대상**만 기준으로 판단하세요.

    반드시 아래 목록 중 하나의 카테고리 이름만 단독으로 출력하세요.
    모호한 경우에는 반드시 '기타'로 답변하세요.

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

    {task_text}

    정답:
    """

    inputs = tokenizer(prompt.strip(), return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=30)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"정답[:：]?\s*(아동청소년|노인|장애인|지구촌|권익신장|시민사회|동물|환경|재난구휼|기타)", decoded)
    category = match.group(1) if match else "기타"
    return category, decoded.strip()
