from pathlib import Path
from collections import Counter
import torch
import json
import re
import torch.nn.functional as F
from konlpy.tag import Okt

from app.ml.model_loader import get_kobert
from app.ml.gemma_predictor import ask_gemma

# ✅ 설정
CATEGORY_PATH = Path("lumina_ai/data/category_dict.json")
common_words = {"오늘", "뭐해", "싶어", "우리", "나도", "위해서", "그리고", "하지만", "그래서", "때문에", "나", "너"}
postpositions = r"(을|를|이|가|은|는|에|에서|와|과|로|으로|도|만|까지|부터|보다|한테|에게|께)$"

# ✅ 형태소 분석기 및 모델 불러오기
okt = Okt()
bert_tokenizer, bert_model = get_kobert()

# ✅ 카테고리 사전 로드
with open(CATEGORY_PATH, encoding="utf-8") as f:
    category_dict = json.load(f)

# ✅ 카테고리 사전 업데이트
def update_category_dict(keyword: str, category: str):
    if category in category_dict:
        if keyword not in category_dict[category]:
            category_dict[category].append(keyword)
    else:
        category_dict[category] = [keyword]
    with open(CATEGORY_PATH, "w", encoding="utf-8") as f:
        json.dump(category_dict, f, ensure_ascii=False, indent=2)

# ✅ 키워드 추출
def extract_keywords(text: str):
    try:
        nouns = okt.nouns(text)
        cleaned = list(set(
            re.sub(postpositions, "", w).strip() for w in nouns if len(w) > 1 and w not in common_words
        ))
        if not cleaned:
            return []

        with torch.no_grad():
            inputs = bert_tokenizer(cleaned, padding=True, truncation=True, return_tensors="pt")
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            norms = F.normalize(embeddings, dim=1)
            scores = torch.matmul(norms, norms.T).sum(dim=1)
            top_indices = scores.topk(min(len(cleaned), 15)).indices
            selected = [cleaned[i] for i in top_indices]

        return selected
    except Exception as e:
        print(f"[키워드 추출 오류] {e}")
        return []

# ✅ 사전에서 카테고리 확인
def check_dict(keyword: str):
    for category, words in category_dict.items():
        if keyword in words:
            return category
    return "기타"

# ✅ 카테고리 추천 로직
def recommend_category_logic(data):
    final_votes = []

    for comment in data.comment:
        keywords = extract_keywords(comment)
        cat_counter = Counter()

        for kw in keywords:
            cat = check_dict(kw)
            if cat == "기타":
                cat, _ = ask_gemma(kw, mode="keyword")
                update_category_dict(kw, cat)
            cat_counter[cat] += 1

        # 기타가 가장 많더라도 다른 카테고리가 있다면 우선
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

        final_votes.append(comment_cat)

    # ✅ post 문장은 직접 Gemma 분류
    post_votes = []
    for post in data.post:
        cat, _ = ask_gemma(post, mode="post")
        post_votes.append(cat)

    # ✅ 최종 다수결
    all_votes = final_votes + post_votes
    recommended = Counter(all_votes).most_common(1)[0][0] if all_votes else "기타"

    return {
        "categoryName": recommended
    }
