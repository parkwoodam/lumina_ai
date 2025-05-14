from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import torch

_kobert_tokenizer = None
_kobert_model = None
_gemma_tokenizer = None
_gemma_model = None

def get_kobert():
    global _kobert_tokenizer, _kobert_model
    if _kobert_model is None:
        _kobert_tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        _kobert_model = BertModel.from_pretrained("monologg/kobert")
        _kobert_model.eval()
    return _kobert_tokenizer, _kobert_model

def get_gemma():
    global _gemma_tokenizer, _gemma_model
    if _gemma_model is None:
        model_id = "google/gemma-3-4b-it"
        _gemma_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _gemma_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        torch.cuda.empty_cache()  # 캐시 정리
    return _gemma_tokenizer, _gemma_model
