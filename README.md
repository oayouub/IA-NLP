# Projet 2 : NLP - Générateur de Texte

## 🧠 Approche itérative : Modèles / encodages

Ce projet implémente un générateur de texte utilisant des modèles récurrents (RNN/LSTM) entraînés sur le corpus Harry Potter.

### 📚 Contenu

- **One-hot, embeddings, char-level** : Encodage des caractères
- **Modèles récurrents (RNN, LSTM)** : Génération de texte caractère par caractère
- **Corpus Harry Potter** : 7 volumes pour l'entraînement

### 🚀 Utilisation

1. **Vérifier CUDA** (recommandé) :
   ```bash
   python check_cuda.py
   ```

2. **Entraîner le modèle** :
   ```bash
   python train_text.py
   ```

3. **Générer du texte interactivement** :
   ```bash
   python generate_text.py
   ```

4. **Ouvrir l'interface web** :
   ```bash
   # Ouvrir index.html dans un navigateur
   ```

5. **Explorer avec Jupyter** :
   ```bash
   jupyter notebook projet.ipynb
   ```

### 📁 Structure

```
iaText/
├── data/                 # Corpus Harry Potter
├── train_text.py        # Script d'entraînement
├── generate_text.py     # Interface interactive de génération
├── check_cuda.py        # Vérification CUDA
├── index.html           # Interface web
├── script.js            # JavaScript pour l'interface
├── projet.ipynb         # Notebook d'exploration
└── README.md            # Ce fichier
```

### 🔧 Technologies

- **PyTorch** : Modèles de deep learning
- **CUDA** : Accélération GPU NVIDIA
- **RNN/LSTM** : Modèles récurrents
- **HTML/JavaScript** : Interface web
- **Jupyter** : Exploration et visualisation

### 📊 Fonctionnalités

- ✅ Génération de texte avec phrases d'entrée
- ✅ Interface web interactive
- ✅ Interface console interactive
- ✅ Visualisation des données
- ✅ Entraînement de modèles RNN
- ✅ Analyse du corpus
- ✅ Accélération GPU avec CUDA
- ✅ Optimisations mémoire GPU

### 🎯 Objectifs pédagogiques

- Comprendre les modèles récurrents pour NLP
- Maîtriser l'encodage de texte (one-hot, embeddings)
- Implémenter la génération de texte
- Créer une interface utilisateur simple 