# Lista 1

Questões cobrindo as aulas 1 a 5

## O algoritmo *perceptron*

1. Seja $x$ tal que $sign(w^Tx) \neq sign(y)$. Após a atualização do vetor $w \leftarrow w + yx$, não podemos dizer que $sign(w^Tx)= sign(y)$. 

   Se $x^{(t)}$ tinha sido classificado incorretamente como negativo,  então $y^{(t)}=1$. Segue então que o novo produto interno é aumentado por $x^{(i)} \cdot x^{(i)}$ (que é positivo). A fronteira se moveu para a direção correta em relação à $x^{(i)}$.

   No entanto, a magnitude obtida pelo produto interno pode não ser o suficiente caso multiplicarmos $w_0 + w_1 x_1 + w_2 x_2 = 0$ por uma constante $\alpha$ muito grande, na qual ela continua valendo $0$ (estaremos apenas escalando o vetor). Com isso, $\alpha w$ será um vetor enorme, e rotação da reta que separa os pontos será muito pequena. Deste modo, o $x$ que estava classificado incorretamente pode se manter incorreto.

   Uma boa ideia para resolver o problema é utilizar o vetor de pesos $w$ normalizado ($\|w\|=1$) ou utilizar uma *learning rate* $\eta$.

   

2. Variante do perceptron: $w \leftarrow w + \eta yx$, com $\eta>0$. O fator $\eta$ é a taxa de aprendizado, sua utilidade é reescalar os pesos, mas ela nunca muda o sinal da predição. Portanto, ela pode acelerar (se pequena) ou retardar (se muito grande) a convergência do algoritmo, mas sempre dentro do upperbound determinado pelo perceptron.

   

3. Se fosse permitido usar múltiplos perceptrons, seria possível construir partições arbitrárias. Basta adicionar um vetor $w$ para cada *label*.

   ``````python
   yhat = np.argmax(np.dot(w, x))
   
   if yhat != y[i]:
   	w[int(y[i])] += x[i]
   	w[yhat[i]] -= x[i]
   ``````



## Regressão linear

1. Se usarmos uma função de custo baseada no erro absoluto $|w^Tx-y|$, otimizamos a função de custo com o vetor gradiente de $J$:
   $$
   J(w) = \frac{1}{N} \sum^N_{n=1}( \underbrace{h_w(x^{(n)})}_{\hat{y}^{(n)}=w^Tx^{(n)}} - y^{(n)}) \\
   \nabla J(w) = \begin{bmatrix} \frac{\partial J}{\partial w_0}, \frac{\partial J}{\partial w_1}, \dots, \frac{\partial J}{\partial w_d} \end{bmatrix}^T
   $$

   $$
   \begin{align}
   \frac{\partial J}{\partial w_j} & = \frac{\partial}{\partial w_j} \frac{1}{2} \sum^N_{n=1} (\hat{y}^{(n)} - y^{(n)}) \\
   & = \frac{1}{2} \sum^N_{n=1} \frac{\partial}{\partial w_j}  (\hat{y}^{(n)} - y^{(n)}) \\
   & = \frac{1}{2} \sum^N_{n=1} \frac{\partial}{\partial w_j}  ((w_0 + w_1 x_1^{(n)} + \dots + w_j x_j^{(n)} + \dots + w_d x_d^{(n)}) - y_n) \\
   & = \frac{1}{2} \sum^N_{n=1} x_j^{(n)}
   \end{align}
   $$

   Note que a função de custo baseada no erro quadrático é mais suscetível a *outliers* do que aquela baseada no erro absoluto.

   

2. É possível utilizar polinômios em vez de linhas na regressão, e isso pode ser feito com o método de quadrados mínimos. A relação entre $x$ e $y$ é modelada por um polinômio de grau $m$ em $x$.



## Regressão logística

1. Como $e^z \neq 0$:

$$
\frac{1}{1 + e^{-z}} = \frac{e^z}{e^z} \cdot \frac{1}{1 + \frac{1}{e^{z}}} = \frac{e^z}{e^z + \frac{e^z}{e^{z}}} = \frac{e^z}{e^z + 1}
$$

​		

2. $$
   1 - \theta(z) = 1 - \frac{e^z}{e^z + 1} = \frac{e^z + 1}{e^z + 1} - \frac{e^z}{e^z + 1} = \frac{e^z + 1 - e^z}{e^z + 1} = \frac{1}{1 + e^z} = \frac{1}{1 + e^{-(-z)}} = \theta(-z)
   $$

   

3. 

   

4. Quanto mais perto da fronteira de decisão, o *score* se aproxima de $0.5$, ou da fronteira definida pelos pesos do modelo (exemplo: se $P_w (y= +1|x^{(n)}) \geq 0.8$, então quanto mais perto da fronteira, o *score* se aproxima de $0.8$). Ponto no qual a sigmoide está centrada.

   

5. No caso da técnica de gradiente descendente para o problema de regressão linear, $\hat{y}^{(n)}=w^Tx^{(n)}$, ou seja, é a predição ($\in \mathbb{R}$) do valor de $y^{(n)}$ com o atual vetor de pesos $w$. 

   Já na função de custo *cross-entropy*, $\hat{y}^{(n)} = \theta(w^Tx) = P_w (y= +1|x^{(n)})$, que devolve o sinal ($\in [0,1]$ — obtido pela aproximação da sigmoide) da predição do valor de $y^{(n)}$ com o atual vetor de pesos $w$. 

   

6. A porcentagem de classificação correta indica quantos dados do *dataset* foram preditos pelo algoritmo. O dataset 98% corretamente classificado provavelmente apresenta dados sem muitos outliers e linearmente divisíveis.  Já no dataset com 72%, deve haver dados com alta variância e coincidência entre as classes.



## Miscelânea

1. Algoritmos com modelos lineares são aqueles que utilizam hiperplanos (objetos modelados por equações lineares) para realizar classificação ou regressão dos dados.

   

2. Podemos utilizar múltiplos hiperplanos para conseguir classificar pontos em múltiplas ($>2$) classes.

   

3. É possível controlar o conjunto de hipóteses e o algoritmo de aprendizado.



4. Problemas computacionais com os quais já me deparei e poderia ter utilizado *machine learning*:
   - sudoku
   - sugestão de cursos

