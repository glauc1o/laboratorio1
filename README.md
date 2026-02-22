# laboratorio1
## Lógica central do mecanismo de Scaled Dot-Product Attention.

Dentro do arquivo *"attention.py"* foram implementadas duas funções.

A função *"softmax()"* aplica Softmax em cada linha da matriz. 

Já a função *"scaled_dot_product_attention()"* é responsável pelo mecanismo de Scaled Dot-Product Attention. Tendo matrizes 'Q' 'K' e 'V' como parâmetros de entrada, onde Q = matriz de Queries, K = Matriz de Keys, V = Matriz de Values. 

Dentro da função, o produto escalar é calculado e então é feita a normalização. Dessa forma busca-se evitar saturação, realizando-se a divisão do resultado do produto escalar pela raíz da dimensão de K. Essa normalização se faz necessária para uma distribuição mais suave e estabilizar o gradiente.

O código pode ser executado através do script *"test_attention.py"*. Como exemplo de input pode ser usado o seguinte exemplo:

Q = np.array([[1, 0, 1]])
    K = np.array([[1, 0, 1],
                  [0, 1, 0]])
    V = np.array([[1, 2],
                  [3, 4]])
 
 Tendo então como output esperado:

 Output:
[[1.47926312 2.47926312]]