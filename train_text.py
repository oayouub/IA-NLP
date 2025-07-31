import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
from collections import Counter

class TextRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)

def load_corpus():
    text = ""
    for i in range(1, 8):
        file_path = f"data/HPBook{i}.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read() + "\n"
    return text

def preprocess_text(text):
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.lower()

def create_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char

def encode_text(text, char_to_idx):
    return [char_to_idx.get(ch, 0) for ch in text]

def create_sequences(encoded_text, seq_length=50):
    sequences = []
    targets = []
    for i in range(len(encoded_text) - seq_length):
        sequences.append(encoded_text[i:i + seq_length])
        targets.append(encoded_text[i + 1:i + seq_length + 1])
    return torch.tensor(sequences), torch.tensor(targets)

print("Chargement du corpus...")
text = load_corpus()
text = preprocess_text(text)
print(f"Taille du corpus: {len(text):,} caractères")

char_to_idx, idx_to_char = create_vocab(text)
vocab_size = len(char_to_idx)
print(f"Taille du vocabulaire: {vocab_size} caractères")

encoded_text = encode_text(text, char_to_idx)
sequences, targets = create_sequences(encoded_text, seq_length=50)
print(f"Nombre de séquences d'entraînement: {len(sequences)}")

train_size = int(0.8 * len(sequences))
train_sequences = sequences[:train_size]
train_targets = targets[:train_size]
test_sequences = sequences[train_size:]
test_targets = targets[train_size:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextRNN(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
batch_size = 32

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(train_sequences), batch_size):
        batch_sequences = train_sequences[i:i + batch_size].to(device)
        batch_targets = train_targets[i:i + batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_sequences)
        loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}/{num_epochs} — loss: {avg_loss:.4f}")
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(test_sequences), batch_size):
            batch_sequences = test_sequences[i:i + batch_size].to(device)
            batch_targets = test_targets[i:i + batch_size].to(device)
            
            outputs = model(batch_sequences)
            loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))
            test_loss += loss.item()
            
            pred = outputs.argmax(dim=-1)
            correct += (pred == batch_targets).sum().item()
            total += batch_targets.numel()
    
    test_accuracy = correct / total
    print(f" → Test accuracy: {test_accuracy:.4f}\n")

def generate_text(model, char_to_idx, idx_to_char, start_text='h', max_length=100):
    """Génère du texte à partir d'une phrase de départ"""
    model.eval()
    generated_text = start_text
    
    with torch.no_grad():
        context = start_text[-1] if len(start_text) > 0 else 'h'
        
        for _ in range(max_length):
            char_idx = char_to_idx.get(context, 0)
            x = torch.tensor([[char_idx]], dtype=torch.long).to(device)
            
            output = model(x)
            probs = torch.softmax(output[0, -1, :], dim=0)
            char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char.get(char_idx, '?')
            
            generated_text += next_char
            context = next_char
            
            if next_char in '.!?':
                break
    
    return generated_text

print("Génération de texte:")

test_phrases = ['harry potter', 'the boy', 'magic was', 'dumbledore']
for phrase in test_phrases:
    generated = generate_text(model, char_to_idx, idx_to_char, phrase, 100)
    print(f"\nPhrase de départ: '{phrase}'")
    print(f"Texte généré: {generated}")

torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size
}, 'text_model.pth')

print("Modèle sauvegardé sous text_model.pth") 