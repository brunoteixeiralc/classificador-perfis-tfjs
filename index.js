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
        epochs: 200, // Número de épocas (200 vezes lendo todo o dataset) para aprender os padrões
        shuffle: true, // Embaralha a ordem dos dados antes de cada época para evitar "decoreba"
        callbacks: {
            // Gatilho que é disparado de forma automática ao final de cada uma das 200 épocas
            onEpochEnd: (epoch, logs) =>
                console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, acc = ${(logs.acc * 100).toFixed(1)}%`)
        }
    });

    // Retorna o modelo treinado para que possamos utilizá-lo nas previsões futuras
    return model;
}

// Função para normalizar a idade dinamicamente usando a fórmula: (Valor - Mínimo) / (Máximo - Mínimo)
function normalizarIdade(idade, min = 0, max = 100) {
    return (idade - min) / (max - min);
}

// Tabelas de codificação one-hot para cor e localização
// Ordem: [azul, vermelho, verde]
const CORES = {
    azul:     [1, 0, 0],
    vermelho: [0, 1, 0],
    verde:    [0, 0, 1]
};

// Ordem: [São Paulo, Rio, Curitiba]
const LOCALIZACOES = {
    'São Paulo': [1, 0, 0],
    Rio:         [0, 1, 0],
    Curitiba:    [0, 0, 1]
};

// Converte os dados de uma pessoa em um vetor numérico para o modelo
// Ordem final: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
function encodePessoa(pessoa) {
    return [
        normalizarIdade(pessoa.idade),
        ...CORES[pessoa.cor],
        ...LOCALIZACOES[pessoa.localizacao]
    ];
}

// Função responsável por usar o modelo treinado para fazer uma nova previsão
async function predict(model, tensorPessoa) {
    // Converte o array do JavaScript (tensorPessoa) em um Tensor 2D compreensível pelo modelo
    const tfInput = tf.tensor2d(tensorPessoa);

    // Pede ao modelo para prever/classificar os dados de entrada
    const previsao = model.predict(tfInput);

    // Extrai os valores matemáticos (probabilidades de cada categoria) do tensor de volta para um array comum JavaScript
    const previsaoArray = await previsao.array();

    // Libera os tensores da memória após extrair os valores (evita vazamento de memória)
    tfInput.dispose();
    previsao.dispose();

    // Retorna o array com as probabilidades
    return previsaoArray[0].map((prob, index) => ({ categoria: labelsNomes[index], probabilidade: prob }));
}

// Dataset de treino: cada pessoa descrita com seus dados originais
// A rede neural só entende números, por isso usamos encodePessoa() para converter automaticamente
const pessoas = [
    // Dados originais
    { nome: "Erick",  idade: 33,  cor: "azul",    localizacao: "São Paulo", label: [1, 0, 0] }, // Premium
    { nome: "Ana",    idade: 0,   cor: "vermelho", localizacao: "Rio",       label: [0, 1, 0] }, // Medium
    { nome: "Carlos", idade: 100, cor: "verde",    localizacao: "Curitiba",  label: [0, 0, 1] }, // Basic

    // PREMIUM: Gostam de Azul, majoritariamente de São Paulo, idade média 30 a 50 anos
    { nome: "P1", idade: 35, cor: "azul", localizacao: "São Paulo", label: [1, 0, 0] },
    { nome: "P2", idade: 45, cor: "azul", localizacao: "São Paulo", label: [1, 0, 0] },
    { nome: "P3", idade: 50, cor: "azul", localizacao: "Rio",       label: [1, 0, 0] },
    { nome: "P4", idade: 40, cor: "azul", localizacao: "Curitiba",  label: [1, 0, 0] },
    { nome: "P5", idade: 38, cor: "azul", localizacao: "São Paulo", label: [1, 0, 0] },

    // MEDIUM: Gostam de Vermelho, majoritariamente do Rio, mais jovens (0 a 30 anos)
    { nome: "M1", idade: 10, cor: "vermelho", localizacao: "Rio",       label: [0, 1, 0] },
    { nome: "M2", idade: 20, cor: "vermelho", localizacao: "Rio",       label: [0, 1, 0] },
    { nome: "M3", idade: 25, cor: "vermelho", localizacao: "São Paulo", label: [0, 1, 0] }, // Morador de SP também pode ser Medium
    { nome: "M4", idade: 15, cor: "vermelho", localizacao: "Curitiba",  label: [0, 1, 0] }, // Morador de Curitiba também pode ser Medium
    { nome: "M5", idade: 5,  cor: "vermelho", localizacao: "Rio",       label: [0, 1, 0] },

    // BASIC: Gostam de Verde, majoritariamente de Curitiba, mais velhos (70+ anos)
    { nome: "B1", idade: 80, cor: "verde", localizacao: "Curitiba",  label: [0, 0, 1] },
    { nome: "B2", idade: 90, cor: "verde", localizacao: "Curitiba",  label: [0, 0, 1] },
    { nome: "B3", idade: 85, cor: "verde", localizacao: "Rio",       label: [0, 0, 1] }, // Morador do Rio também pode ser Basic
    { nome: "B4", idade: 75, cor: "verde", localizacao: "São Paulo", label: [0, 0, 1] }, // Morador de SP também pode ser Basic
    { nome: "B5", idade: 95, cor: "verde", localizacao: "Curitiba",  label: [0, 0, 1] },
];

// Labels das categorias a serem previstas
const labelsNomes = ["premium", "medium", "basic"];

// Criamos tensores de entrada (xs) e saída (ys) a partir do dataset usando encodePessoa()
const inputXs = tf.tensor2d(pessoas.map(p => encodePessoa(p)));
const outputYs = tf.tensor2d(pessoas.map(p => p.label));

const model = await trainModel(inputXs, outputYs);

// Libera os tensores de treino da memória após o treinamento
inputXs.dispose();
outputYs.dispose();

// Dados de uma nova pessoa (Bruno) que a inteligência artificial nunca viu antes
const pessoa = { nome: "Bruno", idade: 28, cor: "vermelho", localizacao: "Rio" };

// encodePessoa() cuida automaticamente da normalização e do one-hot encoding
const tensorPessoa = [encodePessoa(pessoa)];

// Chama a função para tentar prever em qual categoria o Bruno se encaixa (premium, medium ou basic)
const previsao = await predict(model, tensorPessoa);

// Ordena as previsões por probabilidade (do maior para o menor)
const resultado = previsao.sort((a, b) => b.probabilidade - a.probabilidade)[0];

// Exibe a previsão no console
console.log(resultado.categoria, (resultado.probabilidade * 100).toFixed(2) + "% de certeza");
