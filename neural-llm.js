'use strict';

/**
 *© 2025 M.INC. Tutti i diritti riservati. Questo codice non può essere copiato, modificato o distribuito senza permesso."
 * Classe Tensor per operazioni su tensori multidimensionali.
 */
class Tensor {
  constructor(shape, data) {
    this.shape = [...shape];
    this.stride = this.calculateStride();
    const totalSize = this.shape.reduce((a, b) => a * b, 1);
    this.data = data || new Float32Array(totalSize);
  }

  // Calcola lo stride in base alla shape (row-major)
  calculateStride() {
    const stride = new Array(this.shape.length);
    stride[this.shape.length - 1] = 1;
    for (let i = this.shape.length - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * this.shape[i + 1];
    }
    return stride;
  }

  // Converte gli indici multidimensionali in un indice lineare
  getIndex(indices) {
    if (indices.length !== this.shape.length) {
      throw new Error('Numero di indici non corrisponde alle dimensioni del tensore');
    }
    return indices.reduce((acc, idx, i) => acc + idx * this.stride[i], 0);
  }

  // Recupera il valore all'indice specificato
  get(indices) {
    return this.data[this.getIndex(indices)];
  }

  // Imposta il valore all'indice specificato
  set(indices, value) {
    this.data[this.getIndex(indices)] = value;
  }

  // Applica una funzione ad ogni elemento, restituendo un nuovo tensore
  map(fn) {
    const newData = Float32Array.from(this.data, (v, i) => fn(v, i));
    return new Tensor(this.shape, newData);
  }
  
  // Metodo add: somma elemento per elemento con un altro tensore
  add(other) {
    if (this.data.length !== other.data.length) {
      throw new Error('I tensori devono avere lo stesso numero di elementi per l\'addizione');
    }
    const newData = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      newData[i] = this.data[i] + other.data[i];
    }
    return new Tensor(this.shape, newData);
  }

  /**
   * Prodotto matriciale.
   * Se entrambi i tensori sono 2D viene eseguita la moltiplicazione classica;
   * altrimenti si effettua una moltiplicazione "batchata" lungo le ultime due dimensioni.
   */
  dot(other) {
    if (this.shape.length === 2 && other.shape.length === 2) {
      if (this.shape[1] !== other.shape[0]) {
        throw new Error('Dimensioni della matrice non compatibili per il dot');
      }
      const result = new Tensor([this.shape[0], other.shape[1]]);
      for (let i = 0; i < this.shape[0]; i++) {
        for (let j = 0; j < other.shape[1]; j++) {
          let sum = 0;
          for (let k = 0; k < this.shape[1]; k++) {
            sum += this.get([i, k]) * other.get([k, j]);
          }
          result.set([i, j], sum);
        }
      }
      return result;
    } else {
      return this._batchedDot(other);
    }
  }

  // Moltiplicazione batchata per tensori con ≥2 dimensioni.
  _batchedDot(other) {
    const aDims = this.shape;
    const bDims = other.shape;
    const A_M = aDims[aDims.length - 2];
    const A_K = aDims[aDims.length - 1];
    let B_K, B_N, bHasBatch;
    if (bDims.length === 2) {
      B_K = bDims[0];
      B_N = bDims[1];
      bHasBatch = false;
    } else {
      B_K = bDims[bDims.length - 2];
      B_N = bDims[bDims.length - 1];
      bHasBatch = true;
    }
    if (A_K !== B_K) {
      throw new Error('Dimensioni della matrice non compatibili per il dot (batched)');
    }
    const aBatchShape = aDims.slice(0, aDims.length - 2);
    const bBatchShape = bHasBatch ? bDims.slice(0, bDims.length - 2) : [];
    if (bHasBatch && (aBatchShape.length !== bBatchShape.length ||
        !aBatchShape.every((v, i) => v === bBatchShape[i]))) {
      throw new Error('Le dimensioni di batch dei tensori non coincidono');
    }
    const batchShape = aBatchShape;
    const resultShape = (batchShape.length ? batchShape : []).concat([A_M, B_N]);
    const result = new Tensor(resultShape);

    const iterateBatchIndices = (shape, callback, current = []) => {
      if (current.length === shape.length) {
        callback(current);
      } else {
        for (let i = 0; i < shape[current.length]; i++) {
          iterateBatchIndices(shape, callback, current.concat(i));
        }
      }
    };

    if (!batchShape.length) {
      let aRowStride = this.stride[0];
      let bRowStride = bHasBatch ? other.stride[bDims.length - 2] : other.stride[0];
      let resRowStride = result.stride[0];
      for (let i = 0; i < A_M; i++) {
        for (let j = 0; j < B_N; j++) {
          let sum = 0;
          for (let k = 0; k < A_K; k++) {
            const aIndex = i * aRowStride + k;
            const bIndex = k * bRowStride + j;
            sum += this.data[aIndex] * other.data[bIndex];
          }
          result.data[i * resRowStride + j] = sum;
        }
      }
      return result;
    }

    iterateBatchIndices(batchShape, (batchIndices) => {
      let aOffset = 0;
      for (let i = 0; i < batchIndices.length; i++) {
        aOffset += batchIndices[i] * this.stride[i];
      }
      let bOffset = 0;
      if (bHasBatch) {
        for (let i = 0; i < batchIndices.length; i++) {
          bOffset += batchIndices[i] * other.stride[i];
        }
      }
      let resOffset = 0;
      for (let i = 0; i < batchIndices.length; i++) {
        resOffset += batchIndices[i] * result.stride[i];
      }
      const aRowStride = this.stride[batchShape.length];
      const bRowStride = bHasBatch ? other.stride[bDims.length - 2] : other.stride[0];
      const resRowStride = result.stride[batchShape.length];

      for (let i = 0; i < A_M; i++) {
        for (let j = 0; j < B_N; j++) {
          let sum = 0;
          for (let k = 0; k < A_K; k++) {
            const aIndex = aOffset + i * aRowStride + k;
            const bIndex = bOffset + k * bRowStride + j;
            sum += this.data[aIndex] * other.data[bIndex];
          }
          result.data[resOffset + i * resRowStride + j] = sum;
        }
      }
    });
    return result;
  }

  // Permuta gli assi del tensore (se non specificati, inverte l'ordine)
  transpose(axes) {
    if (!axes) {
      axes = [...this.shape.keys()].reverse();
    }
    if (axes.length !== this.shape.length) {
      throw new Error('Il numero di assi deve corrispondere alle dimensioni del tensore');
    }
    const newShape = axes.map(a => this.shape[a]);
    const totalSize = this.data.length;
    const newData = new Float32Array(totalSize);

    const newStride = new Array(newShape.length);
    newStride[newShape.length - 1] = 1;
    for (let i = newShape.length - 2; i >= 0; i--) {
      newStride[i] = newStride[i + 1] * newShape[i + 1];
    }

    const flatToMulti = (flat, shape) => {
      const strides = [];
      let acc = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = acc;
        acc *= shape[i];
      }
      const indices = [];
      for (let i = 0; i < shape.length; i++) {
        indices[i] = Math.floor(flat / strides[i]) % shape[i];
      }
      return indices;
    };

    for (let i = 0; i < totalSize; i++) {
      const oldIndices = flatToMulti(i, this.shape);
      const newIndices = axes.map(a => oldIndices[a]);
      const newIndex = newIndices.reduce((acc, idx, j) => acc + idx * newStride[j], 0);
      newData[newIndex] = this.data[i];
    }
    return new Tensor(newShape, newData);
  }

  /**
   * Calcola la softmax lungo l'ultima dimensione.
   */
  softmax() {
    const lastDim = this.shape[this.shape.length - 1];
    const newData = new Float32Array(this.data.length);
    const outer = this.data.length / lastDim;
    for (let i = 0; i < outer; i++) {
      const start = i * lastDim;
      const end = start + lastDim;
      let maxVal = -Infinity;
      for (let j = start; j < end; j++) {
        if (this.data[j] > maxVal) maxVal = this.data[j];
      }
      let sum = 0;
      for (let j = start; j < end; j++) {
        const expVal = Math.exp(this.data[j] - maxVal);
        newData[j] = expVal;
        sum += expVal;
      }
      for (let j = start; j < end; j++) {
        newData[j] /= sum;
      }
    }
    return new Tensor(this.shape, newData);
  }

  // Metodo gather: per ogni indice in un tensore 2D raccoglie la riga corrispondente
  gather(indicesTensor) {
    const indices = [];
    const [rows, cols] = indicesTensor.shape;
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        row.push(Math.round(indicesTensor.get([i, j])));
      }
      indices.push(row);
    }
    const embeddingSize = this.shape[1];
    const result = new Tensor([rows, cols, embeddingSize]);
    for (let b = 0; b < rows; b++) {
      for (let s = 0; s < cols; s++) {
        const idx = indices[b][s];
        for (let e = 0; e < embeddingSize; e++) {
          result.set([b, s, e], this.get([idx, e]));
        }
      }
    }
    return result;
  }

  // Restituisce una slice lungo la prima dimensione (start incluso, end escluso)
  slice(start, end) {
    const sliceSize = end - start;
    const newShape = [sliceSize, ...this.shape.slice(1)];
    const newData = this.data.slice(start * this.stride[0], end * this.stride[0]);
    return new Tensor(newShape, newData);
  }

  // Cambia la forma del tensore mantenendo lo stesso numero di elementi.
  reshape(newShape) {
    const totalOld = this.shape.reduce((a, b) => a * b, 1);
    const totalNew = newShape.reduce((a, b) => a * b, 1);
    if (totalOld !== totalNew) {
      throw new Error('Il numero totale di elementi deve rimanere invariato');
    }
    return new Tensor(newShape, this.data);
  }
}

/**
 * Configurazione del modello LLM.
 */
class LLMConfig {
  constructor(params = {}) {
    Object.assign(this, {
      vocabSize: 50257,
      hiddenSize: 256,
      numHeads: 8,
      numLayers: 6,
      maxSeqLength: 128,
      dropout: 0.1,
      ...params
    });
  }
}

/**
 * Layer Normalization, normalizza lungo l'ultima dimensione.
 */
class LayerNorm {
  constructor(size) {
    this.gamma = new Tensor([size], new Float32Array(size).fill(1));
    this.beta = new Tensor([size], new Float32Array(size).fill(0));
    this.eps = 1e-5;
  }

  forward(x) {
    const lastDim = this.gamma.shape[0];
    const outer = x.data.length / lastDim;
    const newData = new Float32Array(x.data.length);
    for (let i = 0; i < outer; i++) {
      const offset = i * lastDim;
      let sum = 0;
      for (let j = 0; j < lastDim; j++) {
        sum += x.data[offset + j];
      }
      const mean = sum / lastDim;
      let sumSq = 0;
      for (let j = 0; j < lastDim; j++) {
        const diff = x.data[offset + j] - mean;
        sumSq += diff * diff;
      }
      const variance = sumSq / lastDim;
      for (let j = 0; j < lastDim; j++) {
        newData[offset + j] = ((x.data[offset + j] - mean) / Math.sqrt(variance + this.eps)) * this.gamma.data[j] + this.beta.data[j];
      }
    }
    return new Tensor(x.shape, newData);
  }
}

/**
 * Feed Forward Network con attivazione ReLU.
 */
class FeedForwardNetwork {
  constructor(config) {
    const hiddenSize = config.hiddenSize;
    this.W1 = new Tensor([hiddenSize, 4 * hiddenSize]);
    this.W2 = new Tensor([4 * hiddenSize, hiddenSize]);
    this.initWeights();
  }

  initWeights() {
    const init = (shape) => {
      const std = Math.sqrt(2 / (shape[0] + shape[1]));
      const total = shape[0] * shape[1];
      const data = new Float32Array(total);
      for (let i = 0; i < total; i++) {
        data[i] = (Math.random() * 2 - 1) * std;
      }
      return data;
    };
    this.W1.data = init(this.W1.shape);
    this.W2.data = init(this.W2.shape);
  }

  forward(x) {
    const hidden = x.dot(this.W1).map(v => Math.max(0, v));
    return hidden.dot(this.W2);
  }
}

/**
 * Multi-Head Attention.
 */
class MultiHeadAttention {
  constructor(config) {
    this.numHeads = config.numHeads;
    this.hiddenSize = config.hiddenSize;
    if (this.hiddenSize % this.numHeads !== 0) {
      throw new Error('hiddenSize deve essere multiplo di numHeads');
    }
    this.headSize = this.hiddenSize / this.numHeads;
    this.Wq = this.initWeights([this.hiddenSize, this.hiddenSize]);
    this.Wk = this.initWeights([this.hiddenSize, this.hiddenSize]);
    this.Wv = this.initWeights([this.hiddenSize, this.hiddenSize]);
    this.Wo = this.initWeights([this.hiddenSize, this.hiddenSize]);
  }

  initWeights(shape) {
    const std = Math.sqrt(2 / (shape[0] + shape[1]));
    const total = shape[0] * shape[1];
    const data = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      data[i] = (Math.random() * 2 - 1) * std;
    }
    return new Tensor(shape, data);
  }

  // Divide il tensore in "teste": da [batch, seqLen, hiddenSize] a [batch, numHeads, seqLen, headSize]
  splitHeads(x, batchSize, seqLen) {
    return x.reshape([batchSize, seqLen, this.numHeads, this.headSize])
            .transpose([0, 2, 1, 3]);
  }

  // Combina le teste: da [batch, numHeads, seqLen, headSize] a [batch, seqLen, hiddenSize]
  combineHeads(x) {
    return x.transpose([0, 2, 1, 3])
            .reshape([x.shape[0], x.shape[2], this.hiddenSize]);
  }

  forward(x) {
    const batchSize = x.shape[0];
    const seqLen = x.shape[1];
    const Q = x.dot(this.Wq);
    const K = x.dot(this.Wk);
    const V = x.dot(this.Wv);
    const Q_split = this.splitHeads(Q, batchSize, seqLen);
    const K_split = this.splitHeads(K, batchSize, seqLen);
    const V_split = this.splitHeads(V, batchSize, seqLen);
    const K_t = K_split.transpose([0, 1, 3, 2]);
    let scores = Q_split.dot(K_t).map(v => v / Math.sqrt(this.headSize));
    const probs = scores.softmax();
    const attention = probs.dot(V_split);
    const combined = this.combineHeads(attention);
    return combined.dot(this.Wo);
  }
}

/**
 * Blocco Transformer: include l'attenzione, connessioni skip e FFN.
 */
class TransformerBlock {
  constructor(config) {
    this.config = config;
    this.attention = new MultiHeadAttention(config);
    this.norm1 = new LayerNorm(config.hiddenSize);
    this.ffn = new FeedForwardNetwork(config);
    this.norm2 = new LayerNorm(config.hiddenSize);
  }

  forward(x) {
    const attnOutput = this.attention.forward(x);
    const res1 = x.add(attnOutput);
    const norm1 = this.norm1.forward(res1);
    const ffnOutput = this.ffn.forward(norm1);
    const res2 = norm1.add(ffnOutput);
    return this.norm2.forward(res2);
  }
}

/**
 * Modello NeuralLLM completo.
 * La libreria è utilizzabile sia in ambiente browser che in Node.js.
 */
class NeuralLLM {
  constructor(config = {}) {
    this.config = new LLMConfig(config);
    this.embeddings = {
      token: new Tensor([this.config.vocabSize, this.config.hiddenSize]),
      position: this.createPositionalEncoding()
    };
    this.layers = Array.from({ length: this.config.numLayers }, () => new TransformerBlock(this.config));
    this.head = new Tensor([this.config.hiddenSize, this.config.vocabSize]);
    this.initWeights();
  }

  initWeights() {
    const std = 0.02;
    const init = (tensor) => {
      for (let i = 0; i < tensor.data.length; i++) {
        tensor.data[i] = (Math.random() * 2 - 1) * std;
      }
    };
    init(this.embeddings.token);
    init(this.head);
  }

  createPositionalEncoding() {
    const pe = new Tensor([this.config.maxSeqLength, this.config.hiddenSize]);
    for (let pos = 0; pos < pe.shape[0]; pos++) {
      for (let i = 0; i < pe.shape[1]; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / this.config.hiddenSize);
        pe.set([pos, i], (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle));
      }
    }
    return pe;
  }

  tokenize(text) {
    return [...text].map(c => c.charCodeAt(0) % this.config.vocabSize);
  }

  processBatch(batch) {
    const inputs = [];
    const targets = [];
    
    batch.forEach(({ input, output }) => {
      const text = `${input} <sep> ${output}`;
      const tokens = this.tokenize(text);
      for (let i = 0; i < tokens.length - 1; i++) {
        const padded = tokens.slice(0, i + 1);
        while (padded.length < this.config.maxSeqLength) {
          padded.push(0);
        }
        inputs.push(padded.slice(0, this.config.maxSeqLength));
        targets.push(tokens[i + 1]);
      }
    });
    
    return {
      inputs: new Tensor([inputs.length, this.config.maxSeqLength], Float32Array.from(inputs.flat())),
      targets: new Tensor([targets.length], Float32Array.from(targets))
    };
  }

  computeLoss(outputs, targets) {
    let loss = 0;
    const vocabSize = this.config.vocabSize;
    for (let i = 0; i < outputs.shape[0]; i++) {
      const start = i * vocabSize;
      const probs = outputs.data.slice(start, start + vocabSize);
      loss += -Math.log(probs[targets.data[i]] + 1e-8);
    }
    return loss / outputs.shape[0];
  }

  forward(inputs) {
    const [batchSize, seqLen] = inputs.shape;
    let x = this.embeddings.token.gather(inputs);
    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < seqLen; s++) {
        for (let i = 0; i < this.config.hiddenSize; i++) {
          const idx = b * (seqLen * this.config.hiddenSize) + s * this.config.hiddenSize + i;
          x.data[idx] += this.embeddings.position.get([s, i]);
        }
      }
    }
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    const logits = x.dot(this.head);
    return logits.softmax();
  }

  /**
   * Metodo di training semplificato: esegue solo forward e calcola la loss.
   * Aggiunti log intermedi ogni 10 batch (o all'ultimo batch) per monitorare l'avanzamento.
   * NOTA: Non è implementata la retropropagazione.
   */
  train(dataset, options = {}) {
    const epochs = options.epochs || 10;
    const batchSize = options.batchSize || 32;
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      const batches = [];
      for (let i = 0; i < dataset.length; i += batchSize) {
        batches.push(dataset.slice(i, i + batchSize));
      }
      console.log(`Epoch ${epoch+1}/${epochs} - Inizio... (Total Batches: ${batches.length})`);
      for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
        const batch = batches[batchIndex];
        const { inputs, targets } = this.processBatch(batch);
        const outputs = this.forward(inputs);
        const loss = this.computeLoss(outputs, targets);
        totalLoss += loss;
        if ((batchIndex+1) % 10 === 0 || (batchIndex+1) === batches.length) {
          console.log(`Epoch ${epoch+1}: Batch ${batchIndex+1}/${batches.length} - Loss: ${loss.toFixed(4)}`);
        }
      }
      console.log(`Epoch ${epoch+1}/${epochs} - Average Loss: ${(totalLoss / batches.length).toFixed(4)}`);
    }
  }

  /**
   * Salva il modello in formato JSON.
   * In ambiente Node.js utilizza fs per scrivere il file.
   * In ambiente browser viene lanciato un errore.
   */
  saveModel(filename) {
    const modelData = {
      config: this.config,
      embeddings: {
        token: Array.from(this.embeddings.token.data),
        position: Array.from(this.embeddings.position.data)
      },
      head: Array.from(this.head.data)
      // Per semplicità non salviamo i pesi interni dei layer
    };
    if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
      const fs = require('fs');
      fs.writeFileSync(`${filename}.json`, JSON.stringify(modelData, null, 2));
      console.log(`Modello salvato in ${filename}.json`);
    } else {
      throw new Error("Salvataggio su file non supportato in ambiente browser.");
    }
  }
}

// Esporta la classe solo in ambiente Node.js
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
  module.exports = NeuralLLM;
}
