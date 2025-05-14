from pathlib import Path
from main import app
from fastapi import APIRouter
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from konlpy.tag import Okt
from collections import Counter, defaultdict

router = APIRouter()

category_dict = {}
CATEGORY_PATH = Path(__file__).parent / "category_dict.json"
common_words = {"오늘", "뭐해", "싶어", "우리", "나도", "위해서", "그리고", "하지만", "그래서", "때문에", "나", "너"}
postpositions = r"(을|를|이|가|은|는|에|에서|와|과|로|으로|도|만|까지|부터|보다|한테|에게|께)$"

okt = Okt()
bert_tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertModel.from_pretrained("monologg/kobert")
bert_model.eval()

def load_category_dict():
    global category_dict
    with open(CATEGORY_PATH, encoding="utf-8") as f:
        category_dict = json.load(f)

def update_category_dict(keyword: str, category: str):
    global category_dict
    if category in category_dict:
        if keyword not in category_dict[category]:
            category_dict[category].append(keyword)
    else:
        category_dict[category] = [keyword]
    with open(CATEGORY_PATH, "w", encoding="utf-8") as f:
        json.dump(category_dict, f, ensure_ascii=False, indent=2)

def extract_keywords(text: str):
    try:
        words = okt.nouns(text)
        words = list(set(
            re.sub(postpositions, "", w).strip() for w in words if len(w) > 1 and w not in common_words
        ))

        if not words:
            return []

        with torch.no_grad():
            inputs = bert_tokenizer(words, padding=True, truncation=True, return_tensors="pt")
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            norms = F.normalize(embeddings, dim=1)
            similarity_matrix = torch.matmul(norms, norms.T)
            scores = similarity_matrix.sum(dim=1)
            top_indices = scores.topk(min(len(words), 15)).indices
            selected = [words[i] for i in top_indices]

        return selected
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []

@app.on_event("startup")
def startup_event():
    load_category_dict()

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

def check_dict(keyword: str):
    for category, words in category_dict.items():
        if keyword in words:
            return category
    return "기타"

@torch.inference_mode()
def ask_gemma(keyword: str):
    load_model()
    prompt = f"""
당신은 기부 카테고리를 분류하는 전문가입니다.

아래의 키워드를 보고, 반드시 다음 10가지 카테고리 중 **가장 적절한 하나의 카테고리 이름만** 답변하세요.  
단, 키워드의 의미가 명확하지 않거나 어떤 카테고리에도 **명확히 속하지 않는 경우 반드시 '기타'**로 답변해야 합니다.

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

키워드: {keyword}

정답:
"""
    inputs = gemma_tokenizer(prompt.strip(), return_tensors="pt").to("cuda")
    outputs = gemma_model.generate(**inputs, max_new_tokens=20)
    decoded = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"정답[:：]?\s*(아동청소년|노인|장애인|지구촌|권익신장|시민사회|동물|환경|재난구휼|기타)", decoded)
    cat = match.group(1) if match else "기타"
    return cat, decoded.strip()

class FullInput(BaseModel):
    post: list[str]
    comment: list[str]

@router.post("/recommend")
def recommend_category(data: FullInput):
    keyword_categories = {}
    gemma_used = {}
    comment_results = []
    final_votes = []

    for comment in data.comment:
        keywords = extract_keywords(comment)
        cat_counter = Counter()

        for kw in keywords:
            cat = check_dict(kw)
            if cat == "기타":
                cat, raw = ask_gemma(kw)
                gemma_used[kw] = {"predicted": cat, "raw": raw}
                update_category_dict(kw, cat)
            keyword_categories[kw] = cat
            cat_counter[cat] += 1

        # 기타가 최빈값이어도 다른 카테고리가 있다면 그쪽으로
        if cat_counter:
            if len(cat_counter) > 1 and cat_counter.most_common(1)[0][0] == "기타":
                for cat, _ in cat_counter.most_common():
                    if cat != "기타":
                        comment_cat = cat
                        break
                else:
                    comment_cat = "기타"
            else:
                comment_cat = cat_counter.most_common(1)[0][0]
        else:
            comment_cat = "기타"

        comment_results.append({"comment": comment, "category": comment_cat})
        final_votes.append(comment_cat)

    all_votes = final_votes + data.post
    recommended = Counter(all_votes).most_common(1)[0][0] if all_votes else "기타"

    return {
        # "categories": keyword_categories,
        # "gemma_used": gemma_used,
        # "comments": comment_results,
        # "recommended": recommended
        "categoryName":recommended
    }
    
app.include_router(router)