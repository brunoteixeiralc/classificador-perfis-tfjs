import tf from '@tensorflow/tfjs-node';

// Função principal para criar e treinar a Rede Neural
async function trainModel(xs, ys) {
    // Cria um modelo sequencial (camadas conectadas em sequência, uma após a outra)
    const model = tf.sequential();

    // Adiciona a primeira camada (oculta/densa) com 80 neurônios, formato de entrada 7 e função de ativação ReLU
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Adiciona a camada de saída com 3 neurônios (um para cada categoria) e ativação Softmax (probabilidades)
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Otimizador Adam com taxa de aprendizado de 0.01
    const otimizador = tf.train.adam(0.01);

    // Compila/Prepara o modelo com o otimizador Adam e função de perda categoricalCrossentropy
    model.compile({ optimizer: otimizador, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Executa o treinamento da rede neural com os tensores de entrada (xs) e saída (ys)
    await model.fit(xs, ys, {
        verbose: 0, // Oculta logs padrão no terminal
        epochs: 200, // Número de épocas (100 vezes lendo todo o dataset) para aprender os padrões
        shuffle: true, // Embaralha a ordem dos dados antes de cada época para evitar "decoreba"
        callbacks:
        {
            // Gatilho que é disparado de forma automática ao final de cada uma das 100 épocas
            onEpochEnd: (epoch, logs) =>
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`) // Imprime o erro atual (loss) para vermos diminuir ao vivo
        }
    });

    // Retorna o modelo treinado para que possamos utilizá-lo nas previsões futuras
    return model;
}

// Função para normalizar a idade dinamicamente usando a fórmula: (Valor - Mínimo) / (Máximo - Mínimo)
function normalizarIdade(idade, min = 0, max = 100) {
    return (idade - min) / (max - min);
}

// Função responsável por usar o modelo treinado para fazer uma nova previsão
async function predict(model, tensorPessoa) {
    // Converte o array do JavaScript (tensorPessoa) em um Tensor 2D compreensível pelo modelo
    const tfInput = tf.tensor2d(tensorPessoa);

    // Pede ao modelo para prever/classificar os dados de entrada
    const previsao = model.predict(tfInput);

    // Extrai os valores matemáticos (probabilidades de cada categoria) do tensor de volta para um array comum JavaScript
    const previsaoArray = await previsao.array();

    // Exibe o array com as probabilidades no console
    //console.log(previsaoArray);

    // Retorna o array com as probabilidades
    return previsaoArray[0].map((prob, index) => ({ categoria: labelsNomes[index], probabilidade: prob }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick - Premium
    [0, 0, 1, 0, 0, 1, 0],    // Ana - Medium
    [1, 0, 0, 1, 0, 0, 1],    // Carlos - Basic

    // PREMIUM: Gostam de Azul, majoritariamente de São Paulo, idade média 30 a 50 (0.3 a 0.5)
    [0.35, 1, 0, 0, 1, 0, 0],
    [0.45, 1, 0, 0, 1, 0, 0],
    [0.50, 1, 0, 0, 0, 1, 0],
    [0.40, 1, 0, 0, 0, 0, 1],
    [0.38, 1, 0, 0, 1, 0, 0],

    // MEDIUM: Gostam de Vermelho, majoritariamente do Rio, mais jovens 0 a 30 (0.0 a 0.3)
    [0.10, 0, 1, 0, 0, 1, 0],
    [0.20, 0, 1, 0, 0, 1, 0],
    [0.25, 0, 1, 0, 1, 0, 0], // Morador de SP também pode ser Medium
    [0.15, 0, 1, 0, 0, 0, 1], // Morador de Curitiba também pode ser Medium
    [0.05, 0, 1, 0, 0, 1, 0],

    // BASIC: Gostam de Verde, majoritariamente de Curitiba, mais velhos 70+ (0.7 a 1.0)
    [0.80, 0, 0, 1, 0, 0, 1],
    [0.90, 0, 0, 1, 0, 0, 1],
    [0.85, 0, 0, 1, 0, 1, 0], // Morador do Rio também pode ser Basic
    [0.75, 0, 0, 1, 1, 0, 0], // Morador de SP também pode ser Basic
    [0.95, 0, 0, 1, 0, 0, 1]
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    // Label original
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1], // basic - Carlos

    // Labels para os novos dados (acompanhando EXATAMENTE a ordem matemática lá de cima)
    [1, 0, 0], // premium
    [1, 0, 0], // premium
    [1, 0, 0], // premium
    [1, 0, 0], // premium
    [1, 0, 0], // premium

    [0, 1, 0], // medium
    [0, 1, 0], // medium
    [0, 1, 0], // medium
    [0, 1, 0], // medium
    [0, 1, 0], // medium

    [0, 0, 1], // basic
    [0, 0, 1], // basic
    [0, 0, 1], // basic
    [0, 0, 1], // basic
    [0, 0, 1]  // basic
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//inputXs.print();
//outputYs.print();

const model = await trainModel(inputXs, outputYs);

// Dados de uma nova pessoa (Bruno) que a inteligência artificial nunca viu antes
const pessoa = { nome: "Bruno", idade: 28, cor: "vermelho", localizacao: "Rio" }

// Normalizando e aplicando one-hot encoding manualmente nos dados do Bruno para o mesmo formato usado no treinamento
// Ordem exigida pelo modelo: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
const tensorPessoa = [[
    normalizarIdade(pessoa.idade), // Idade normalizada dinamicamente
    0,    // Cor azul: 0 (falso)
    1,    // Cor vermelho: 1 (verdadeiro)
    0,    // Cor verde: 0 (falso)
    0,    // Localização São Paulo: 0 (falso)
    1,    // Localização Rio: 1 (verdadeiro)
    0     // Localização Curitiba: 0 (falso)
]]

// Chama a função para tentar prever em qual categoria o Bruno se encaixa (premium, medium ou basic)
const previsao = await predict(model, tensorPessoa);

// Ordena as previsões por probabilidade (do maior para o menor)
const resultado = previsao.sort((a, b) => b.probabilidade - a.probabilidade)[0];

// Exibe a previsão no console
console.log(resultado.categoria, (resultado.probabilidade * 100).toFixed(2) + "% de certeza");

