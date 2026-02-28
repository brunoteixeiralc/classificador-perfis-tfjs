# Classificador de Perfis com TensorFlow.js

Este é um projeto prático que utiliza **TensorFlow.js no Node.js** (`@tensorflow/tfjs-node`) para construir, treinar e utilizar uma Rede Neural Artificial simples. O objetivo do modelo é classificar o perfil de um usuário em uma de três categorias: **Premium**, **Medium** ou **Basic**, com base em características demográficas e preferências.

## 🧠 Como Funciona

A rede neural recebe informações de entrada sobre uma pessoa e prevê a qual categoria ela pertence.

### Características de Entrada
Para que o modelo matemático entenda os dados (já que redes neurais só processam números), as características da pessoa são transformadas em um formato numérico através de técnicas como **Normalização** e **One-Hot Encoding**:

1. **Idade**: É normalizada para um valor entre 0 e 1 (ex: `idade / 100`).
2. **Cor Favorita**: Representada de forma binária (One-Hot) nas opções Azul, Vermelho ou Verde (ex: `[1, 0, 0]` para Azul).
3. **Localização**: Representada de forma binária nas opções São Paulo, Rio de Janeiro ou Curitiba (ex: `[0, 1, 0]` para Rio).

O tensor de entrada final obedece a seguinte ordem:
`[idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]`

### A Rede Neural
O modelo possui a seguinte arquitetura:
- **Camada Oculta (Densa)**: 80 neurônios utilizando a função de ativação `ReLU`, que ajuda o modelo a aprender padrões não lineares complexos.
- **Camada de Saída (Densa)**: 3 neurônios (um para cada categoria - Premium, Medium, Basic) utilizando a função de ativação `Softmax`, que converte a saída matemática em probabilidades parecidas com porcentagem.
- **Otimizador**: "Adam" com uma taxa de aprendizado de `0.01`.
- **Função de Perda**: `categoricalCrossentropy`.

### Categorias de Destino (Saída)
O aprendizado tenta classificar predominantemente os seguintes padrões apresentados no dataset de treino:
* **Premium**: Média de idade entre 30 a 50 anos, preferem Azul e geralmente são de São Paulo.
* **Medium**: Mais jovens (0 a 30 anos), preferem Vermelho e geralmente são do Rio de Janeiro.
* **Basic**: Mais velhos (70+ anos), preferem Verde e geralmente são de Curitiba.

## 🚀 Pré-requisitos

- [Node.js](https://nodejs.org/) instalado em seu ambiente.

## 📦 Instalação

1. Clone ou baixe este repositório.
2. Navegue até a pasta do projeto no terminal:
   ```bash
   cd classificador-perfis-tfjs
   ```
3. Instale as dependências executando:
   ```bash
   npm install
   ```

## ⚙️ Como Executar

Para iniciar o treinamento e logo em seguida realizar uma previsão de exemplo (atualmente configurado para prever o perfil do usuário "Bruno", de 28 anos, que gosta da cor vermelho e mora no Rio), basta rodar:

```bash
npm start
```
*(O script `start` foi configurado no `package.json` para rodar `node --watch index.js`, reinciando a rede automaticamente a cada vez que o código for salvo)*

### O que você verá no console:
Durante a execução, você não verá log de cada época pois o `verbose` está desligado no treinamento (apenas exibirá através do callback a queda da taxa de "loss" épocas específicas, se configurado). Ao final das 200 épocas de treinamento, o modelo fará a previsão e exibirá no console:

```
medium 98.45% de certeza
```
*(As porcentagens podem variar minimamente a cada execução devido à natureza probabilística da inicialização e otimização da rede neural)*

## 🛠️ Tecnologias Utilizadas

- **JavaScript (Node.js)**
- **TensorFlow.js API** (`@tensorflow/tfjs-node`)
