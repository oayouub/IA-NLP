        
let modelLoaded = false;
let charToIdx = {};
let idxToChar = {};
let vocabSize = 0;

class SimpleRNN {
    constructor(vocabSize, hiddenSize = 64) {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.hidden = new Array(hiddenSize).fill(0);
        this.weights = this.initializeWeights();
    }

    initializeWeights() {
        return {
            input: Array(this.hiddenSize).fill().map(() => 
                Array(this.vocabSize).fill().map(() => Math.random() * 0.1 - 0.05)
            ),
            hidden: Array(this.hiddenSize).fill().map(() => 
                Array(this.hiddenSize).fill().map(() => Math.random() * 0.1 - 0.05)
            ),
            output: Array(this.vocabSize).fill().map(() => 
                Array(this.hiddenSize).fill().map(() => Math.random() * 0.1 - 0.05)
            )
        };
    }

    forward(input) {
        const oneHot = new Array(this.vocabSize).fill(0);
        oneHot[input] = 1;

        const newHidden = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.vocabSize; j++) {
                newHidden[i] += oneHot[j] * this.weights.input[i][j];
            }
            for (let j = 0; j < this.hiddenSize; j++) {
                newHidden[i] += this.hidden[j] * this.weights.hidden[i][j];
            }
            newHidden[i] = Math.tanh(newHidden[i]);
        }

        const output = new Array(this.vocabSize).fill(0);
        for (let i = 0; i < this.vocabSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                output[i] += newHidden[j] * this.weights.output[i][j];
            }
        }

        const maxVal = Math.max(...output);
        const expOutput = output.map(x => Math.exp(x - maxVal));
        const sumExp = expOutput.reduce((a, b) => a + b, 0);
        const probabilities = expOutput.map(x => x / sumExp);

        this.hidden = newHidden;
        return probabilities;
    }

    reset() {
        this.hidden = new Array(this.hiddenSize).fill(0);
    }
}

const trainingCorpus = `
harry potter was a young wizard who lived in a cupboard under the stairs at number four privet drive. 
the boy who lived had been sleeping in the cupboard under the stairs for ten years. 
a mysterious letter arrived at the dursley household addressed to harry potter. 
magic was real and harry was about to discover his true identity as a wizard. 
dumbledore looked at harry with his twinkling blue eyes and smiled. 
severus snape was the potions master at hogwarts school of witchcraft and wizardry. 
ron weasley was harry's best friend and they shared many adventures together. 
gryffindor was the house that harry potter belonged to at hogwarts. 
voldemort was the dark lord who tried to kill harry potter when he was a baby. 
wizardry was a magical art that harry potter learned at hogwarts school. 
hogwarts was the most magical school in the wizarding world. 
hermione granger was the brightest witch of her age. 
quidditch was the most exciting sport in the magical world. 
the chamber of secrets held ancient magical secrets. 
the sorting hat placed students in their appropriate houses.
the great hall was filled with floating candles and enchanted ceiling. 
the forbidden forest was home to many magical creatures. 
the platform nine and three quarters was where students caught the hogwarts express. 
the golden snitch was the most important ball in quidditch. 
the elder wand was the most powerful wand in existence.
`;

function createVocabulary(text) {
    const chars = [...new Set(text.toLowerCase())].sort();
    const charToIdx = {};
    const idxToChar = {};
    
    chars.forEach((char, idx) => {
        charToIdx[char] = idx;
        idxToChar[idx] = char;
    });
    
    return { charToIdx, idxToChar, vocabSize: chars.length };
}

function trainModel(corpus, epochs = 100) {
    const { charToIdx, idxToChar, vocabSize } = createVocabulary(corpus);
    const model = new SimpleRNN(vocabSize);
    
    const text = corpus.toLowerCase();
    const encoded = text.split('').map(char => charToIdx[char] || 0);
    
    console.log(`Entraînement du modèle avec ${vocabSize} caractères...`);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;
        
        for (let i = 0; i < encoded.length - 1; i++) {
            const input = encoded[i];
            const target = encoded[i + 1];
            
            const output = model.forward(input);
            const loss = -Math.log(output[target] + 1e-8);
            totalLoss += loss;
        }
        
        if (epoch % 20 === 0) {
            console.log(`Epoch ${epoch}: Loss = ${(totalLoss / encoded.length).toFixed(4)}`);
        }
    }
    
    return { model, charToIdx, idxToChar, vocabSize };
}

function generateTextWithModel(model, charToIdx, idxToChar, startPhrase, maxLength) {
    model.reset();
    
    const startChars = startPhrase.toLowerCase().split('');
    for (const char of startChars) {
        if (charToIdx[char] !== undefined) {
            model.forward(charToIdx[char]);
        }
    }
    
    let generatedText = startPhrase;
    
    for (let i = 0; i < maxLength - startPhrase.length; i++) {
        const lastChar = generatedText[generatedText.length - 1].toLowerCase();
        const input = charToIdx[lastChar] || 0;
        
        const probabilities = model.forward(input);
        
        const temperature = 0.8;
        const scaledProbs = probabilities.map(p => Math.pow(p, 1/temperature));
        const sum = scaledProbs.reduce((a, b) => a + b, 0);
        const normalizedProbs = scaledProbs.map(p => p / sum);
        
        const random = Math.random();
        let cumulative = 0;
        let selectedChar = 0;
        
        for (let j = 0; j < normalizedProbs.length; j++) {
            cumulative += normalizedProbs[j];
            if (random <= cumulative) {
                selectedChar = j;
                break;
            }
        }
        
        const nextChar = idxToChar[selectedChar] || ' ';
        generatedText += nextChar;
        
        if (nextChar === '.' || nextChar === '!' || nextChar === '?') {
            break;
        }
    }
    
    return generatedText;
}

let trainedModel = null;

function initializeModel() {
    const result = trainModel(trainingCorpus, 50);
    trainedModel = result.model;
    charToIdx = result.charToIdx;
    idxToChar = result.idxToChar;
    vocabSize = result.vocabSize;
    modelLoaded = true;
}

function generateText(startPhrase, maxLength) {
    if (!modelLoaded || !trainedModel) {
        return "Modèle en cours de chargement...";
    }
    
    try {
        const generatedText = generateTextWithModel(
            trainedModel, 
            charToIdx, 
            idxToChar, 
            startPhrase, 
            maxLength
        );
        
        return generatedText;
    } catch (error) {
        console.error('Erreur lors de la génération:', error);
        return "Erreur lors de la génération du texte.";
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate');
    const clearBtn = document.getElementById('clear');
    const output = document.getElementById('output');
    const startPhraseInput = document.getElementById('startPhrase');
    const maxLengthInput = document.getElementById('maxLength');
    
    output.textContent = 'Initialisation du modèle RNN...';
    initializeModel();
    
    generateBtn.addEventListener('click', function() {
        const startPhrase = startPhraseInput.value;
        const maxLength = parseInt(maxLengthInput.value);
        
        if (!startPhrase) {
            output.textContent = 'Veuillez entrer une phrase de départ.';
            return;
        }
        
        if (!modelLoaded) {
            output.textContent = 'Modèle en cours de chargement, veuillez patienter...';
            return;
        }
        
        output.textContent = 'Génération en cours...';
        
        setTimeout(() => {
            const generatedText = generateText(startPhrase, maxLength);
            
            output.textContent = `Phrase de départ: "${startPhrase}"\n` +
                               `Longueur: ${generatedText.length} caractères\n\n` +
                               `Texte généré:\n"${generatedText}"`;
        }, 500);
    });
    
    clearBtn.addEventListener('click', function() {
        output.textContent = 'Cliquez sur "Générer" pour créer du texte...';
        startPhraseInput.value = 'harry potter';
        maxLengthInput.value = '100';
    });
    
    setTimeout(() => {
        if (modelLoaded) {
            const generatedText = generateText('harry potter', 100);
            output.textContent = `Exemple de génération:\n\n"${generatedText}"`;
        }
    }, 2000);
});