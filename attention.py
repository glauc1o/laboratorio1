import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Aplica softmax linha a linha de forma numericamente estável.
    """
    # Subtrai o máximo de cada linha para estabilidade numérica
    x_stable = x - np.max(x, axis=1, keepdims=True)
    
    exp_x = np.exp(x_stable)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    return exp_x / sum_exp_x


def scaled_dot_product_attention(Q: np.ndarray,
                                 K: np.ndarray,
                                 V: np.ndarray) -> np.ndarray:
    """
    Implementa o mecanismo de Scaled Dot-Product Attention.

    Parâmetros:
    Q : matriz de Queries  (n, d_k)
    K : matriz de Keys     (m, d_k)
    V : matriz de Values   (m, d_v)

    Retorna:
    Matriz resultante da atenção (n, d_v)
    """

    # Dimensão das chaves
    d_k = K.shape[1]

    # 1. Produto escalar QK^T
    scores = np.matmul(Q, K.T)

    # 2. Scaling (dividir por sqrt(d_k))
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Aplicar softmax linha a linha
    attention_weights = softmax(scaled_scores)

    # 4. Multiplicar pelos valores V
    output = np.matmul(attention_weights, V)

    return output