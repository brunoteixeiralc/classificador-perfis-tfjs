# Classificador de Perfis com TensorFlow.js

Projeto prático que utiliza **TensorFlow.js no Node.js** (`@tensorflow/tfjs-node`) para construir, treinar e utilizar uma Rede Neural Artificial simples. O objetivo do modelo é classificar o perfil de um usuário em uma de três categorias — **Premium**, **Medium** ou **Basic** — com base em características demográficas e preferências.

## 🎯 Resultado

Após 200 épocas de treinamento, o modelo classifica uma nova pessoa em tempo real:

```
Epoch 0:   loss = 1.0987, acc = 33.3%
Epoch 1:   loss = 1.0820, acc = 44.4%
...
Epoch 198: loss = 0.0041, acc = 100.0%
Epoch 199: loss = 0.0039, acc = 100.0%

medium 98.45% de certeza
```

> As porcentagens podem variar levemente a cada execução devido à natureza probabilística da inicialização da rede neural.

---

## 🧠 Como Funciona

### Pipeline completo

```
Dados brutos → encodePessoa() → Tensor 2D → Rede Neural → Probabilidades → Categoria
{ nome, idade,                  [[0.28, 0,    [input]      [0.02,           "medium"
  cor, localizacao }              1, 0, 0,                  0.97,
                                  0, 1, 0]]                 0.01]]
```

### Pré-processamento com `encodePessoa()`

Para que o modelo entenda os dados, cada pessoa é automaticamente convertida em um vetor numérico de 7 dimensões usando duas técnicas:

| Técnica | Característica | Exemplo |
|---|---|---|
| **Normalização** | Idade → valor entre 0 e 1 | `28 anos → 0.28` |
| **One-Hot Encoding** | Cor → vetor binário | `"vermelho" → [0, 1, 0]` |
| **One-Hot Encoding** | Localização → vetor binário | `"Rio" → [0, 1, 0]` |

**Vetor final:** `[idade_norm, azul, vermelho, verde, São Paulo, Rio, Curitiba]`

```js
// Antes: escrito manualmente (sujeito a erros)
const tensor = [[0.28, 0, 1, 0, 0, 1, 0]]

// Depois: gerado automaticamente
const tensor = [encodePessoa({ nome: "Bruno", idade: 28, cor: "vermelho", localizacao: "Rio" })]
```

### Arquitetura da Rede Neural

```
Entrada (7)  →  Camada Oculta (80 neurônios, ReLU)  →  Saída (3 neurônios, Softmax)
                                                         [premium, medium, basic]
```

- **Otimizador:** Adam (`lr = 0.01`)
- **Loss:** Categorical Cross-Entropy
- **Épocas:** 200 com shuffle

### Categorias aprendidas pelo modelo

| Categoria | Faixa de Idade | Cor Favorita | Localização |
|---|---|---|---|
| 🥇 **Premium** | 30 – 50 anos | Azul | São Paulo |
| 🥈 **Medium** | 0 – 30 anos | Vermelho | Rio de Janeiro |
| 🥉 **Basic** | 70+ anos | Verde | Curitiba |

---

## 🚀 Como Executar

**Pré-requisito:** [Node.js](https://nodejs.org/) instalado (recomendado: LTS v22).

```bash
# 1. Clone o repositório
git clone https://github.com/brunoteixeiralc/classificador-perfis-tfjs.git
cd classificador-perfis-tfjs

# 2. Instale as dependências
npm install

# 3. Execute
npm start
```

> O script `start` usa `node --watch`, reiniciando automaticamente a cada vez que o código for salvo.

---

## 🛠️ Tecnologias

- **JavaScript (Node.js)**
- **TensorFlow.js** (`@tensorflow/tfjs-node`)

## 🤖 Créditos

O desenvolvimento e refinamento do código-fonte deste projeto contaram com o auxílio da inteligência artificial **Claude** (Anthropic).
