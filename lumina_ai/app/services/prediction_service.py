from app.ml.model_loader import get_gemma
import torch
import re

@torch.inference_mode()
def generate_prediction(post_text: str):
    tokenizer, model = get_gemma()
    inputs = tokenizer(post_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False, use_cache=False,num_beams=1)

    # 디코딩
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    response_text = full_output.replace(input_text, "").strip()

    # 코드 블록 제거
    code_block_pattern = r'^(?:text|markdown)?([\s\S]*?)$'
    match = re.match(code_block_pattern, response_text)
    if match:
        response_text = match.group(1).strip()
    else:
        simple_code_pattern = r'^([\s\S]*?)$'
        match = re.match(simple_code_pattern, response_text)
        if match:
            response_text = match.group(1).strip()

    return {"response": response_text}