from pathlib import Path
from collections import Counter
import torch
import json
import re
import torch.nn.functional as F
from konlpy.tag import Okt

from app.ml.model_loader import get_kobert
from app.ml.gemma_predictor import ask_gemma

CATEGORY_PATH = Path("lumina_ai/data/category_dict.json")
common_words = {"오늘", "뭐해", "싶어", "우리", "나도", "위해서", "그리고", "하지만", "그래서", "때문에", "나", "너"}
postpositions = r"(을|를|이|가|은|는|에|에서|와|과|로|으로|도|만|까지|부터|보다|한테|에게|께)$"

okt = Okt()
bert_tokenizer, bert_model = get_kobert()

# JSON 로드
with open(CATEGORY_PATH, encoding="utf-8") as f:
    category_dict = json.load(f)

def update_category_dict(keyword: str, category: str):
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

def check_dict(keyword: str):
    for category, words in category_dict.items():
        if keyword in words:
            return category
    return "기타"

def recommend_category_logic(data):
    keyword_categories = {}
    final_votes = []

    for comment in data.comment:
        keywords = extract_keywords(comment)
        cat_counter = Counter()

        for kw in keywords:
            cat = check_dict(kw)
            if cat == "기타":
                cat = ask_gemma(kw, mode="keyword")
                update_category_dict(kw, cat)
            keyword_categories[kw] = cat
            cat_counter[cat] += 1

        # 기타가 가장 많더라도 다른 카테고리가 있다면 그쪽으로
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

    all_votes = final_votes + data.post
    recommended = Counter(all_votes).most_common(1)[0][0] if all_votes else "기타"

    return {
        "categoryName": recommended
    }
