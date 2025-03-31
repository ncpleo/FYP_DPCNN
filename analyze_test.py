import torch
import torch.nn as nn
from model import DPCNN
from main import config
import json
import os
import re
import numpy as np

def normalize_text(text):
    text = text.lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Keep punctuation
    return text

def analyze_sentence(model, embeds, word_to_index, sentence):
    model.eval()
    normalized_sentence = normalize_text(sentence)
    tokens = normalized_sentence.split()
    indices = [word_to_index.get(word, 0) for word in tokens]
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    if len(indices) == 0:
        print("No valid words found in the input.")
        return
    max_length = 50
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    input_data = embeds(input_tensor).float().permute(0, 2, 1).unsqueeze(1)
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
        threshold = 0.5
        print(f"Raw Output: {output}")
        print(f"Predicted Probabilities: {probabilities}")
        sentiment = "positive" if probabilities[0][1] > threshold else "negative"
        print(f"The sentiment is classified as {sentiment}.")

def load_pretrained_embeddings(embedding_path, word_to_index, embedding_dim):
    vocab_size = len(word_to_index) + 1
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    print(f"Loading pre-trained embeddings from {embedding_path}...")
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: Expected {embedding_dim}, got {len(vector)}")
            if word in word_to_index:
                index = word_to_index[word]
                embedding_matrix[index] = vector
    print("Pre-trained embeddings loaded successfully.")
    return torch.FloatTensor(embedding_matrix)

if __name__ == "__main__":
    checkpoint_path = 'checkpoints/best_model.ckpt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        model = DPCNN(config).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        with open('mappings/word_to_index.json', 'r', encoding='utf-8') as f:
            word_to_index = json.load(f)
        embedding_path = 'data/glove/glove.6B.300d.txt'
        embedding_matrix = load_pretrained_embeddings(embedding_path, word_to_index, config.word_embedding_dimension)
        embeds = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(device)  # Fine-tune embeddings
        while True:
            sentence = input("Enter a sentence to analyze (or type 'exit' to quit): ")
            if sentence.lower() == 'exit':
                break
            analyze_sentence(model, embeds, word_to_index, sentence)