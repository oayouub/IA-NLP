// Générateur de texte simple - Simulation
let modelLoaded = false;

// Échantillons de texte généré (simulation)
const textSamples = {
    'harry potter': 'harry potter was a young wizard who lived in a cupboard under the stairs at number four privet drive.',
    'the boy': 'the boy who lived had been sleeping in the cupboard under the stairs for ten years.',
    'a mysterious': 'a mysterious letter arrived at the dursley household addressed to harry potter.',
    'magic was': 'magic was real and harry was about to discover his true identity as a wizard.',
    'dumbledore': 'dumbledore looked at harry with his twinkling blue eyes and smiled.',
    'severus snape': 'severus snape was the potions master at hogwarts school of witchcraft and wizardry.',
    'ron weasley': 'ron weasley was harry\'s best friend and they shared many adventures together.',
    'gryffindor': 'gryffindor was the house that harry potter belonged to at hogwarts.',
    'voldemort': 'voldemort was the dark lord who tried to kill harry potter when he was a baby.',
    'wizardry': 'wizardry was a magical art that harry potter learned at hogwarts school.',
    'hogwarts': 'hogwarts was the most magical school in the wizarding world.',
    'hermione': 'hermione granger was the brightest witch of her age.',
    'quidditch': 'quidditch was the most exciting sport in the magical world.',
    'the chamber': 'the chamber of secrets held ancient magical secrets.',
    'the sorting': 'the sorting hat placed students in their appropriate houses.'
};

// Fonction de génération de texte (simulation)
function generateText(startPhrase, maxLength) {
    const start = startPhrase.toLowerCase();
    
    // Utiliser un échantillon prédéfini ou générer du texte aléatoire
    let baseText = textSamples[start] || 
        `${startPhrase} discovered that magic was real and the wizarding world was full of wonder and danger.`;
    
    // Ajouter de la variabilité
    const variations = [
        ' the magic was powerful and mysterious.',
        ' hogwarts was the most magical place he had ever seen.',
        ' his friends hermione and ron were always there to help.',
        ' professor dumbledore guided him through his journey.',
        ' the dark arts were a constant threat to the wizarding world.',
        ' quidditch was the most exciting sport in the magical world.',
        ' the sorting hat placed him in gryffindor house.',
        ' voldemort was the greatest threat to the wizarding world.',
        ' the chamber of secrets held ancient magical secrets.',
        ' the triwizard tournament tested the skills of young wizards.'
    ];
    
    // Ajouter des variations aléatoires
    while (baseText.length < maxLength) {
        const randomVariation = variations[Math.floor(Math.random() * variations.length)];
        baseText += randomVariation;
        
        // Arrêter si on dépasse la longueur maximale
        if (baseText.length > maxLength) {
            baseText = baseText.substring(0, maxLength);
            break;
        }
    }
    
    return baseText;
}

// Événements DOM
document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate');
    const clearBtn = document.getElementById('clear');
    const output = document.getElementById('output');
    const startPhraseInput = document.getElementById('startPhrase');
    const maxLengthInput = document.getElementById('maxLength');
    
    // Générer du texte
    generateBtn.addEventListener('click', function() {
        const startPhrase = startPhraseInput.value;
        const maxLength = parseInt(maxLengthInput.value);
        
        if (!startPhrase) {
            output.textContent = 'Veuillez entrer une phrase de départ.';
            return;
        }
        
        // Afficher un message de chargement
        output.textContent = 'Génération en cours...';
        
        // Simuler un délai de traitement
        setTimeout(() => {
            const generatedText = generateText(startPhrase, maxLength);
            
            output.textContent = `Phrase de départ: "${startPhrase}"\n` +
                               `Longueur: ${generatedText.length} caractères\n\n` +
                               `Texte généré:\n"${generatedText}"`;
        }, 1000);
    });
    
    // Effacer l'affichage
    clearBtn.addEventListener('click', function() {
        output.textContent = 'Cliquez sur "Générer" pour créer du texte...';
        startPhraseInput.value = 'harry potter';
        maxLengthInput.value = '100';
    });
    
    // Générer automatiquement au chargement
    setTimeout(() => {
        const generatedText = generateText('harry potter', 100);
        output.textContent = `Exemple de génération:\n\n"${generatedText}"`;
    }, 500);
});

// Informations sur le modèle (simulation)
console.log('🧠 Générateur de Texte - Projet NLP');
console.log('📚 Modèle: RNN/LSTM pour génération caractère par caractère');
console.log('📖 Corpus: Livres Harry Potter (7 volumes)');
console.log('🔤 Encodage: One-hot et embeddings');
console.log('⚡ Technique: Modèles récurrents pour NLP'); 