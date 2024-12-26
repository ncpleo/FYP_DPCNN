import torch
import torch.nn as nn
from model import DPCNN
from main import config
import json
import os
import re
import numpy as np

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing punctuation,
    and handling negations.
    """
    text = text.lower()
    
    # Replace Unicode quotation marks with ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # Double quotes

    # Remove other Unicode characters (optional)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    negation_words = {"not", "no", "never", "none", "don't", "doesn't"}
    
    words = text.split()
    new_words = []
    negate_next = False

    for word in words:
        if negate_next:
            new_words.append("NOT_" + word)  # Prefix negated words
            negate_next = False
        elif word in negation_words:
            negate_next = True
        else:
            new_words.append(word)

    return ' '.join(new_words)


# Analyze sentence function with debugging information
def analyze_sentence(model, embeds, word_to_index, sentence):
    model.eval()

    # Normalize and tokenize input sentence
    normalized_sentence = normalize_text(sentence)
    tokens = normalized_sentence.split()

    # Convert tokens to indices
    indices = [word_to_index.get(word, 0) for word in tokens]
    
    # Debugging: Print token-to-index mapping
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")

    if len(indices) == 0:
        print("No valid words found in the input.")
        return

    # Ensure a minimum input size
    max_length = 50  # Set according to your model's requirements
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad with zeros
    else:
        indices = indices[:max_length]  # Truncate if too long

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # Prepare embedding and reshape input data for DPCNN
    input_data = embeds(input_tensor).float().permute(0, 2, 1).unsqueeze(1)

    # Get model prediction
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)

        print(f"Raw Output: {output}")
        print(f"Predicted Probabilities: {probabilities}")

        if probabilities[0][1] > probabilities[0][0]:
            print("The sentiment is classified as positive.")
        else:
            print("The sentiment is classified as negative.")

def load_pretrained_embeddings(embedding_path, word_to_index, embedding_dim):
    """
    Load pre-trained embeddings (e.g., GloVe) and create an embedding matrix.
    Args:
        embedding_path (str): Path to the pre-trained embedding file.
        word_to_index (dict): Mapping of words to indices.
        embedding_dim (int): Dimension of the embeddings.
    Returns:
        torch.FloatTensor: Embedding matrix aligned with word_to_index.
    """
    vocab_size = len(word_to_index) + 1  # +1 for padding index
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))  # Random initialization

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

            # Handle negated terms (e.g., "NOT_good")
            if f"NOT_{word}" in word_to_index:
                index_negated = word_to_index[f"NOT_{word}"]
                embedding_matrix[index_negated] = -vector  # Negate the vector for contrast

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

        # Path to your GloVe file
        embedding_path = 'data/glove/glove.6B.300d.txt'

        # Load pre-trained embeddings
        embedding_matrix = load_pretrained_embeddings(embedding_path, word_to_index, config.word_embedding_dimension)

        # Initialize embedding layer with pre-trained weights
        embeds = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(device)  # Set freeze=True if you don't want to fine-tune


        while True:
            sentence = input("Enter a sentence to analyze (or type 'exit' to quit): ")
            if sentence.lower() == 'exit':
                break

            analyze_sentence(model, embeds, word_to_index, sentence)
