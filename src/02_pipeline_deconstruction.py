# src/02_pipeline_deconstruction.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def manual_inference(text, model_name="klue/bert-base"):
    """
    Hugging Face pipeline() API의 블랙박스를 제거하고, 
    추론 과정을 3단계로 수동 제어하는 엔지니어링 구현체입니다.
    """
    print(f"[{model_name}] 수동 추론 파이프라인 가동...")
    
    # 1. Tokenization (Encoding): 텍스트 -> 정수 인덱스 텐서
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    
    # 2. Forward Pass (Model): 연산 그래프 통과 -> Logits 추출
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval() # 추론 모드 전환
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # 3. Activation (Post-Processing): Logits -> 최종 확률 변환
    probabilities = F.softmax(logits, dim=-1)
    
    return probabilities

if __name__ == "__main__":
    # NSMC 데이터 특성을 반영한 리뷰 텍스트 테스트
    sample_text = "이 영화 감독은 천재인 줄 알았는데, 완전 실망이다. 돈 아까움."
    
    result_probs = manual_inference(sample_text)
    
    neg_prob = result_probs[0][0].item() * 100
    pos_prob = result_probs[0][1].item() * 100
    
    print(f"\n입력 문장: '{sample_text}'")
    print(f"➡️ 부정 확률: {neg_prob:.2f}%")
    print(f"➡️ 긍정 확률: {pos_prob:.2f}%")
