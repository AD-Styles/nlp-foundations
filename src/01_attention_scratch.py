import numpy as np

def softmax(x):
    """안정적인 연산을 위한 소프트맥스 구현"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Scaled Dot-Product Attention 구현
    """
    d_k = Q.shape[-1]
    # Step 1: Score 계산 (유사도 측정 및 스케일링)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Step 2: Weight 계산 (확률 분포 변환)
    weights = softmax(scores)
    
    # Step 3: Context Vector 산출 (가중합)
    context = np.dot(weights, V)
    
    return context, weights

if __name__ == "__main__":
    # 임의의 임베딩 데이터 (3개 단어, 4차원 벡터)
    Q = np.array([[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]])
    K = np.array([[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]])
    V = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]])
    
    context, weights = scaled_dot_product_attention(Q, K, V)
    print("--- Attention Weights ---\n", weights)
    print("\n--- Context Vector ---\n", context)
