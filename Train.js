const NeuralLLM = require('./neural-llm');
const fs = require('fs');

const dataset = JSON.parse(fs.readFileSync('dataset.json'));

const llm = new NeuralLLM({
    vocabSize: 50257,
    hiddenSize: 256,
    numHeads: 8,
    numLayers: 4,
    maxSeqLength: 128
});

console.log('Inizio addestramento...');
llm.train(dataset, {
    epochs: 20,
    batchSize: 64
});

llm.saveModel('trained_model');
console.log('Addestramento completato!');