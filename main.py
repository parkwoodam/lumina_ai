from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 모델 및 토크나이저 로드
model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# FastAPI 인스턴스 생성
app = FastAPI()

# 요청 바디 정의
class PostRequest(BaseModel):
    post: str

# 예측 API 엔드포인트
@app.post("/predict")
async def predict(data: PostRequest):
    # 입력 텍스트 토크나이즈 및 GPU로 이동
    inputs = tokenizer(data.post, return_tensors="pt").to("cuda")

    # 모델 추론
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    # 전체 출력 디코딩
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 입력 텍스트(프롬프트)도 디코딩
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # 프롬프트 부분 제거하여 생성된 응답만 추출
    response_text = full_output.replace(input_text, "").strip()

    # 코드 블록 패턴 제거
    code_block_pattern = r'^```(?:text|markdown)?([\s\S]*?)```$'
    code_match = re.match(code_block_pattern, response_text)
    if code_match:
        response_text = code_match.group(1).strip()
    else:
        # 단순 백틱 패턴도 확인
        simple_code_pattern = r'^```([\s\S]*?)```$'
        simple_match = re.match(simple_code_pattern, response_text)
        if simple_match:
            response_text = simple_match.group(1).strip()

    # 응답 반환
    return {"response": response_text}
