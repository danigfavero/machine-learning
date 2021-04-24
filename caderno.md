# MAC0460 - Introdução ao Aprendizado de Máquina

Oferecimento da disciplina no primeiro semestre de 2021, com a professora Nina S. T. Hirata, no Instituto de Matemática e Estatística da USP.

**Bibliografia**

- **Book:** Abu-Mostafa et al. *Learning from Data,* 2012
- [*Machine Learning* online course (MOOC)](https://work.caltech.edu/telecourse.html) — Caltech (video-lectures, slides)

**Avaliação**

- EPs
- Exercícios escritos
- QTs: Tarefas rápidas (~10% da nota final)

## Introdução

### O que é Machine Learning?

- Inteligência artificial
- Data Science
  - The fourth paradigm
  - Big data (5V: volume, velocity, variety, veracity, value)
  - Deep learning

![ia-ml-and-ds](./img/ia-ml-and-ds.png)

Machine learning é parte de inteligência artificial e data science.

### Sob uma perspectiva computacional...

Computadores processam dados de acordo com algoritmos, e algoritmos são soluções para problemas computacionais. Em machine learning lidaremos com relações de *input-output* com dados complexos, casos ambíguos e dados que mudam frequentemente. Em geral **é difícil criar um algoritmo, mas fácil de listar pares *input-output***.

Em vez de escrever um código específico, podemos considerar um **modelo genérico para mapear entradas e saídas**, ajustando seus parâmetros a partir de dados de treinamento disponíveis.

Sob um ponto de vista puramente computacional, machine learning é uma abordagem de meta-programação.

## Problemas de machine learning

### Assumimos...

- Disponibilidade de exemplos de entrada-saída: $(x^{(n)}, y^{(n)}), n=1, \dots, N$
- A existência de uma relação plausível entre $x$ e $y$

### Problemas que gostaríamos de resolver:

Encontrar um mapeamento (programa de computador) que leva $x$ a $y$. Discutiremos **regressão** e **classificação**.

Técnicas de ML utilizadas para resolver problemas típicos de regressão e classificação costumam ser chamados **supervisionadas**. Supervisão se refere ao fato de que a saída $y$ esperada é conhecida para cada observação $x$ no conjunto exemplo de entrada.

A saída esperada $y$ é utilizada em algoritmos de ML para guiar o processo de aprendizado *(por isso se chama "supervisionado")*. 

### Problemas de regressão

$y$ é uma variável contínua $\implies$ o problema é de **regressão**

Encontrando $f$ tal que $y=f(x)$: podemos escolher $f(x; w, b) = wx + b$, com $w, b$ parâmetros do modelo *(para simplificar, vamos tentar uma função afim)*.

#### Formalizando...

Dado um conjunto de treinamento $\{(x^{(n)}, y^{(n)}): n = 1, \dots, N\}$ e a família $\mathcal{F} = \{f(x) = wx + b:w, x, b \in \mathbb{R} \}$, escolha $f^* \in \mathcal{F}$ (ou $w$ e $b$) tal que $f^*(x^{(n)})=y^{(n)},  n=1, ...,N$. Isso é possível? **NÃO!**

Então vamos encontrar lha $f^* \in \mathcal{F}$ (ou $w$ e $b$) tal que $f^*(x^{(n)}) \approx y^{(n)},  n=1, ...,N$.

Portanto, precisamos medir quão perto $f^*(x^{(n)})$ está de $y^{(n)}$.

Definindo o que deve ser minimizado *(a média do erro considerando todas as amostras)*:
$$
J(w,b) = \frac{1}{N} \sum^{N}_{n=1}{l((wx^{(n)}+b), y^{(n)})}
$$
Tal que a função de perda é $l(a,b) = (a-b)^2$ *(este é o erro quadrático, que a gente escolheu agora).*

Então temos o **Mean Squared Error (MSE)**:
$$
J(w,b) = \frac{1}{N} \sum^{N}_{n=1}{((wx^{(n)}+b) - y^{(n)})^2}
$$
É um problema de otimização: queremos minimizar o erro! Quando encontrarmos a função ótima $f^*$, podemos predizer saídas $y$ com novas entradas $x$.

### Problemas de classificação

$y$ é categórica $\implies$ o problema é de **classificação**

É necessário encontrar a relação entre $x$ e $y$. Podemos procurar (estimar) uma **distribuição de probabilidade**. 

#### Abordagem estatística

Teorema de Bayes:
$$
P(y|x) = \frac{P(y)P(x|y)}{P(x)}
$$
Se sabemos as distribuições, temos a *winning rule*:
$$
y^*= \arg\max_{y}{P(y|x)}
$$
E se não tivermos as distribuições? Temos apenas as observações *(nem sempre dá pra fazer a estimação da distribuição de probabilidade)*. Podemos definir um **ponto** que divide a reta de observações e delimita categorias.

E em um input **bidimensional** $x=(x_1, x_2)$? Agora, em vez de um ponto, podemos usar uma **reta** para delimitar as categorias.

Uma reta $w_1x_1 + w_2x_2 + b = 0$ para dividir o espaço no $\mathbb{R}^2$ em duas regiões e declarar:

​	Para um dado $x = (x_1, x_2)$,

​		se $w_1x_1 + w_2x_2 + b > 0 \implies y = $ *categoria 1*

​		se  $w_1x_1 + w_2x_2 + b < 0 \implies y = $ *categoria 2*.

**Fronteira de decisão:** poderíamos polinômios em vez de funções lineares. Novamente, qual família de mapeamentos que deveríamos escolher? Como determinar o valor ótimo desses parâmetros?

### Estatística X Machine Learning

Abordagens:

1. Foco na distribuição de $x$ *(prevalece na estatística)*

   - ABORDAGEM **GERATIVA**
2. Foco na saída esperada $P(y|x)$ *(prevalece em ML)*

   - ABORDAGEM **DISCRIMINATIVA**
- APRENDIZADO LIVRE DE DISTRIBUIÇÃO

### Superfície de decisão

- E se os pontos não forem repartíveis por uma reta?
  - Gaussiana 2D: linhas de contorno (distribuição simétrica)
    - Duas normais formam um sino
    - Medir matriz de covariância
- Tem relação com a distribuição de probabilidade (na prática não conseguimos estimar)

#### Distribuição normal

- Casos em que a fronteira é uma reta
  - $\Sigma_j = \sigma^2I$: ambas as classes têm a mesma matriz de covariância, covariância nula ($\sigma_{ii}=\sigma_{jj}$) -- distribuição simétrica
  - $\Sigma_j=\Sigma$: ambas as classes têm a mesma matriz de covariância

- Casos em que a fronteira não é uma reta
  - $\Sigma_j$ arbitrário: classes têm matrizes de covariância distintas

- Em **classificação binária**, se as duas classes têm distribuição normal com a **mesma matriz de covariância**, então a superfície de decisão ótima é um **hiperplano**

## Perceptron

- Encontra um hiperplano separador quando as classes são linearmente separáveis

INPUT: $x = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$

OUTPUT: $y \in \{-1, +1 \}$

HIPÓTESE: $h(x)= sign((\sum^d_{i=1}{w_ix_i})+b),  b \in \mathbb{R}, w_i \in \mathbb{R}, i = 1, 2, \dots, d$

Componente artificial, apenas para simplificar a notação:
$$
x = (1, x_1, x_2,  \dots, x_d) \in \mathbb{R}^{d+1} \\
w = (w_0, w_1, w_2,  \dots, w_d) \in \mathbb{R}^{d+1}
$$
Portanto 
$$
h(x) = sign(w^Tx) \ \ \ \ \ (w^Tx=0 \text{ define um hiperplano})
$$

### Algoritmo

Seja $w$ o peso "atual"

Seja $\mathcal{D}= \{(x^{(i)}, y^{(i)}), i = 1, \dots, N \}$ o conjunto de treinamento 

Repita:

 1. Seja $(x^{(i)}, y^{(i)}) \in \mathcal{D} $ tal que $sign(w^Tx^{(i)}) \neq y^{(i)}$. Se não houver tal par, então pare.

 2. Atualize o peso atual conforme: 

    $w \leftarrow w + y^{(i)}x^{(i)}, y^{(i)} \in \{ -1, +1 \}$

Devolva $w$

### Intuição

![img](http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/images/perceptron/perceptron_img2.png)

### Convergência do algoritmo perceptron

Assumimos...

- Duas classes são **linearmente separáveis** se existe um hiperplano separador com margem $\gamma$ -- existe um vetor de pesos $w$ com norma 1 ($\|w\|=1$) tal que $y^{(i)}w^Tx^{(i)}>\gamma, \forall i$

- Tome também $R$ como sendo a **máxima norma** dos exemplos $x^{(i)} \in \mathcal{D}$

- Tome $k$ como o **número de iterações** do algoritmo

A prova consiste em mostrar que $k$ é limitado por $\mathcal{O}(R^2/\gamma^2)$

​	**(1)** 			$\|w^{k+1}\| > k \gamma$

​	**(2)** 			$\| w^{k+1} \|^2 \leq kR^2$

​	**(1) + (2)**    $k^2\gamma^2< \| w^{k+1} \|^2 \leq kR^2 \implies k < \frac{R^2}{\gamma^2}$, que é finito

### Observações

- Pocket algorithm