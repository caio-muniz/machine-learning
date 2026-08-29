# Machine Learning

Repositório criado para estudar e praticar conceitos de **Machine Learning**, começando pelos fundamentos e avançando gradualmente para problemas de classificação e processamento de imagens.

A ideia é implementar os conceitos na prática, entender seu funcionamento e registrar a evolução do aprendizado.

## Conteúdos

### 01 — Perceptron

Implementação do Perceptron do zero, sem utilizar bibliotecas de Machine Learning.

* Função de ativação
* Pesos e bias
* Regra de atualização dos pesos
* Treinamento
* Avaliação
* Fronteira de decisão

### 02 — Scikit-learn

Primeiros experimentos utilizando modelos e ferramentas da biblioteca Scikit-learn.

* Dataset Iris
* Separação entre treino e teste
* K-Nearest Neighbors (KNN)
* Escolha do hiperparâmetro `K`
* Cross Validation
* Avaliação de acurácia

### 03 — Image Processing

Estudo dos fundamentos de processamento de imagens utilizando Python.

* Leitura de imagens
* Acesso a pixels
* Canais RGB
* Conversão para escala de cinza
* Redimensionamento
* Normalização
* Manipulação de arrays com NumPy

### 04 — Image Classification

Classificação de imagens utilizando o dataset **Fashion-MNIST** e o algoritmo KNN.

Pipeline utilizado:

```text
Imagem 28 × 28
      ↓
Flatten
      ↓
784 características
      ↓
Normalização
      ↓
KNN
      ↓
Classificação
```

O experimento utiliza:

* Fashion-MNIST
* Flatten das imagens
* Normalização dos pixels
* KNN
* Cross Validation
* Seleção de hiperparâmetro
* Avaliação em conjunto de teste

### Resultado atual

Utilizando 10.000 imagens para treinamento e 2.000 para teste:

```text
Melhor K pela Cross Validation: 6
Cross Validation: 82,46%
Acurácia no teste: 82,45%
```

## Tecnologias

* Python
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* Pillow

## Estrutura

```text
machine-learning/
│
├── 01-perceptron/
│   └── perceptron.py
│
├── 02-sklearn/
│   └── iris.py
│
├── 03-image-processing/
│   └── image_basics.py
│
├── 04-image-classification/
│   └── fashion_mnist.py
│
└── utils/
    ├── __init__.py
    └── image_processing.py
```

## Objetivo

O objetivo deste repositório é construir uma base prática em Machine Learning, partindo dos algoritmos fundamentais e avançando para problemas mais complexos.

Os próximos experimentos irão explorar técnicas de classificação de imagens e redes neurais, buscando compreender não apenas como utilizar os modelos, mas também como avaliar seus resultados e suas limitações.

```
```
