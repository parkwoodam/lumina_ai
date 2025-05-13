from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 디버깅용 (선택)

model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 안정성 ↑
    device_map="auto"
)

prompt = '''다음은 사용자(닉네임: "우담")가 작성한 게시글입니다:
"오늘은 노후된 지역에 가서 벽화를 그렸어요 뿌듯하네요!"

당신은 선한 행동 SNS 플랫폼의 관리자 Luna입니다. 이 게시글에 대한 반응과 적절한 리워드를 제공해야 합니다. 

게시글 분석:
1. 먼저 게시글이 다음 중 어떤 유형인지 판단하세요:
   A) 실제 봉사활동이나 선한 행동을 직접 수행했다는 내용
   B) 선한 생각, 정보 공유, 또는 간접적인 선행 관련 내용

2. 유형에 따른 리워드 책정:
   - A유형: 적절한 리워드 (실제 행동에 대한 가치 인정)
   - B유형: 작은 리워드 (의미 있는 공유에 대한 가치 인정)

3. 응답 작성:
   - 게시글 내용에 직접 반응하는 개인화된 피드백
   - 해당 행동/생각이 사회에 미치는 긍정적 영향 강조
   - 작성자에게 맞춤형 격려와 응원의 메시지
   - 유형에 따른 적절한 리워드 언급

응답에서 작성자를 언급할 때는 반드시 닉네임으로로 부르세요.
게시글 내용 자체를 사람처럼 부르거나 언급하지 마세요.
(예를 들어, 게시글 내용이 "휴지를 주웠어요"라면 "휴지님"이라고 부르지 마세요.)
그리고 답변은 한 가지만 보여주면 됩니다. 정말로 대화를 하는 챗봇처럼 답장 하세요.

최종 응답 형식:
1. 게시글 내용에 대한 직접적인 반응
2. 사회적 영향에 대한 언급
3. 격려 메시지와 함께 리워드 안내 (A유형은 B유형보다 더 높은 리워드, B유형은 작은 리워드)'''

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)  # 샘플링 off
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
