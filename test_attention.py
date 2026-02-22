from attention import scaled_dot_product_attention

import numpy as np

if __name__ == "__main__":
    Q = np.array([[1, 0, 1]])
    K = np.array([[1, 0, 1],
                  [0, 1, 0]])
    V = np.array([[1, 2],
                  [3, 4]])

    output = scaled_dot_product_attention(Q, K, V)

    print("Output:")
    print(output)