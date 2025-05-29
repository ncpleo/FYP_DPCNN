import torch
import torch.nn as nn
from model import DPCNN
from main import config
import json
import os
import re
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

# Initialize rich console for enhanced UI
console = Console()


def normalize_text(text):
    """
    Normalize input text by converting to lowercase and cleaning special characters.

    Args:
        text (str): Input text

    Returns:
        str: Normalized text
    """
    text = text.lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Keep punctuation
    return text


def analyze_sentence(model, embeds, word_to_index, sentence, history):
    """
    Analyze the sentiment of a sentence and display results in a table.

    Args:
        model: Trained DPCNN model
        embeds: Embedding layer
        word_to_index (dict): Word-to-index mapping
        sentence (str): Input sentence
        history (list): List to store analysis history
    """
    model.eval()
    normalized_sentence = normalize_text(sentence)
    tokens = normalized_sentence.split()

    # Handle empty input
    if not tokens:
        console.print("[red]Error: No valid words found in the input.[/red]")
        return

    # Convert tokens to indices, pad or truncate to max length
    indices = [word_to_index.get(word, 0) for word in tokens]
    max_length = 50
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    # Prepare input tensor
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    input_data = embeds(input_tensor).float().permute(0, 2, 1).unsqueeze(1)

    # Get model predictions
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][1].item() if probabilities[0][1] > 0.5 else probabilities[0][0].item()
        sentiment = "Positive" if probabilities[0][1] > 0.5 else "Negative"

    # Create and display result table
    table = Table(title="Sentiment Analysis Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input Sentence", sentence)
    table.add_row("Normalized Tokens", ", ".join(tokens))
    table.add_row("Sentiment", f"[bold {sentiment.lower()}]{sentiment}[/bold {sentiment.lower()}]")
    table.add_row("Confidence", f"{confidence:.2%}")
    console.print(table)

    # Store result in history
    history.append({
        "sentence": sentence,
        "sentiment": sentiment,
        "confidence": confidence,
        "tokens": tokens
    })


def show_history(history):
    """
    Display analysis history in a table.

    Args:
        history (list): List of past analyses
    """
    if not history:
        console.print("[yellow]No analysis history available.[/yellow]")
        return

    table = Table(title="Analysis History", show_header=True, header_style="bold magenta")
    table.add_column("No.", style="cyan")
    table.add_column("Sentence", style="green")
    tableAe.add_column("Sentiment", style="blue")
    table.add_column("Confidence", style="yellow")

    for idx, entry in enumerate(history, 1):
        table.add_row(
            str(idx),
            entry["sentence"],
            entry["sentiment"],
            f"{entry['confidence']:.2%}"
        )

    console.print(table)


def load_pretrained_embeddings(embedding_path, word_to_index, embedding_dim):
    """
    Load GloVe embeddings for the vocabulary.

    Args:
        embedding_path (str): Path to GloVe file
        word_to_index (dict): Word-to-index mapping
        embedding_dim (int): Embedding dimension

    Returns:
        torch.FloatTensor: Embedding matrix
    """
    vocab_size = len(word_to_index) + 1
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    console.print(f"[blue]Loading pre-trained embeddings from {embedding_path}...[/blue]")
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
    console.print("[green]Pre-trained embeddings loaded successfully.[/green]")
    return torch.FloatTensor(embedding_matrix)


if __name__ == "__main__":
    # Initialize paths and device
    checkpoint_path = 'checkpoints/best_model.ckpt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[blue]Using device: {device}[/blue]")

    # Check if model checkpoint exists
    if not os.path.exists(checkpoint_path):
        console.print("[red]Error: Model checkpoint not found at {checkpoint_path}.[/red]")
        exit(1)

    # Load model
    console.print(f"[blue]Loading model from: {checkpoint_path}[/blue]")
    model = DPCNN(config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Load word-to-index mapping
    with open('mappings/word_to_index.json', 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)

    # Load embeddings
    embedding_path = 'data/glove/glove.6B.300d.txt'
    embedding_matrix = load_pretrained_embeddings(embedding_path, word_to_index, config.word_embedding_dimension)
    embeds = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(device)

    # Initialize history list
    history = []

    # Display welcome message
    console.print(Panel.fit(
        Text("Welcome to the Amazon Review Sentiment Analyzer!\n"
             "Enter a sentence to analyze its sentiment, 'history' to view past analyses, or 'exit' to quit.",
             style="bold green"),
        title="Sentiment Analyzer"
    ))

    # Main interaction loop
    while True:
        user_input = Prompt.ask("[bold cyan]Enter a sentence (or 'history'/'exit')[/bold cyan]")

        if user_input.lower() == 'exit':
            console.print("[green]Thank you for using the Sentiment Analyzer. Goodbye![/green]")
            break
        elif user_input.lower() == 'history':
            show_history(history)
        elif user_input.strip() == "":
            console.print("[red]Error: Input cannot be empty.[/red]")
        else:
            analyze_sentence(model, embeds, word_to_index, user_input, history)
