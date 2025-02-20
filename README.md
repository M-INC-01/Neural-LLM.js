# NeuralLLM - Libreria per LLM in JavaScript

Implementazione di un Large Language Model in JavaScript puro per Node.js e browser.

Realizzata da M.INC.

## 🚀 Installazione

```bash
git clone https://github.com/tuo-utente/neural-llm.git
cd neural-llm
```

## 💻 Uso con Node.js

### Addestramento
Crea un file `train.js`:
```javascript
const NeuralLLM = require('./src/neural-llm');
const fs = require('fs');

// Carica dataset
const dataset = JSON.parse(fs.readFileSync('dataset.json'));

// Configura modello
const llm = new NeuralLLM({
    vocabSize: 50257,
    hiddenSize: 256,
    numHeads: 8,
    numLayers: 4,
    maxSeqLength: 128
});

// Addestramento
console.log('Avvio addestramento...');
llm.train(dataset, {
    epochs: 20,
    batchSize: 64
});

// Salva modello
llm.saveModel('trained_model');
console.log('Addestramento completato!');
```

Esegui:
```bash
node train.js
```

### Formato dataset.json
```json
[
    {"input": "Ciao", "output": "Ciao! Come posso aiutarti?"},
    {"input": "Come stai?", "output": "Sono un'IA sempre pronta ad aiutare!"},
    {"input": "Cosa sai fare?", "output": "Posso rispondere alle tue domande e conversare"}
]
```

## 🌐 Uso nel browser
```html
<!DOCTYPE html>
<html>
<head>
    <script src="src/neural-llm.js"></script>
</head>
<body>
    <script>
        // Carica modello pre-addestrato
        fetch('model.json')
            .then(response => response.json())
            .then(modelData => {
                const llm = new NeuralLLM(modelData.config);
                
                // Carica pesi
                llm.embeddings.token.data = new Float32Array(modelData.embeddings.token);
                llm.head.data = new Float32Array(modelData.head);
                
                // Generazione testo
                const response = llm.generate("Ciao", {maxLength: 50});
                console.log(response);
            });
    </script>
</body>
</html>
```

## 📁 Struttura del progetto
- `src/`: Codice sorgente della libreria
- `examples/`: Esempi d'uso
- `models/`: Modelli pre-addestrati (opzionali)

## 🔧 Configurazione modello
Personalizza i parametri:
```javascript
new NeuralLLM({
    vocabSize: 50257,     # Dimensione vocabolario
    hiddenSize: 256,      # Dimensione layer nascosti
    numHeads: 8,          # Teste per l'attention
    numLayers: 4,         # Numero layer transformer
    maxSeqLength: 128     # Lunghezza massima sequenza
});
```

## 📚 Salvataggio/Caricamento modelli

**Node.js**:
```javascript
// Salva
llm.saveModel('my_model');

// Carica
const modelData = JSON.parse(fs.readFileSync('my_model.json'));
const llm = new NeuralLLM(modelData.config);
llm.embeddings.token.data = new Float32Array(modelData.embeddings.token);
llm.head.data = new Float32Array(modelData.head);
```

**Browser**:
```javascript
// Salva (da implementare)
const modelJson = JSON.stringify(llm.exportModel());
downloadAsFile(modelJson, 'model.json');
```

## ⚠️ Limitazioni
- Training lento in browser per dataset grandi
- Necessario ottimizzazione memoria per modelli grandi
- Supporto GPU limitato

## 📄 Licenza
MIT License
M.INC.©
