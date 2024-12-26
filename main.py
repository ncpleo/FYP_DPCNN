# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import Config
from model import DPCNN
from data import TextDataset
import argparse
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Import functional as F
import re
import time
from tqdm import tqdm
import random
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import WeightedRandomSampler

torch.manual_seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
#parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

# Create the configuration
config = Config(batch_size=args.batch_size,
                label_num=args.label_num,
                learning_rate=args.lr,
                epoch=args.epoch)

#Training part
def train_model(config):
    start_time = time.time()  # Start timer
    
    torch.manual_seed(0)

   # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'GPU is available.' if torch.cuda.is_available() else 'Using CPU.'}")

    # Path to data
    train_data_path = 'data/train_split'
    val_data_path = 'data/val_split'
    test_data_path = 'data/test_split'
    raw_data_path = 'data/raw'
    
    # Create directories for checkpoints and graphs
    checkpoint_dir = 'checkpoints'
    graph_dir = os.path.join('pictures', 'graphs')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    # Initialize a list to store loss values
    loss_values = []
    
    # Create a directory for storing the mapping if it doesn't exist
    mapping_dir = 'mappings'
    os.makedirs(mapping_dir, exist_ok=True)
    
    # Save the word_to_index mapping to a JSON file in the mappings directory
    mapping_file_path = os.path.join(mapping_dir, 'word_to_index.json')
    word_to_index = load_word_to_index(mapping_file_path)
    
    # Process your new data to create a new list of words
    new_words = create_word_to_index(raw_data_path).keys()
    word_to_index = update_word_to_index(new_words, word_to_index)
    
    # Save the updated word_to_index mapping back to the JSON file
    with open(mapping_file_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f)
    
    # Initialize the dataset with the word-to-index mapping
    training_set = TextDataset(path=train_data_path, word_to_index=word_to_index, augment=True)
    
    label_counts = Counter(training_set.labels)
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor([total_samples / label_counts[i] for i in range(len(label_counts))]).to(device)
    
        
    validation_set = TextDataset(path=val_data_path, word_to_index=word_to_index, augment=False)  # Load validation set

    # Compute sample weights
    sample_weights = [1 / label_counts[label] for label in training_set.labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(training_set), replacement=True)
    
    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=16,
                                    sampler=sampler)

    validation_iter = data.DataLoader(dataset=validation_set,
                                      batch_size=config.batch_size,
                                      num_workers=16)

    
    # Path to your GloVe file
    embedding_path = 'data/glove/glove.6B.300d.txt'

    # Load pre-trained embeddings
    embedding_matrix = load_pretrained_embeddings(embedding_path, word_to_index, config.word_embedding_dimension)

    # Initialize embedding layer with pre-trained weights
    embeds = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(device)  # Set freeze=True if you don't want to fine-tune

    # Normalize embedding weights
    #embeds.weight.data = F.normalize(embeds.weight.data, p=2, dim=1)
    
    
    model = DPCNN(config).to(device)
    
    
    #criterion = nn.CrossEntropyLoss()
    #L2
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-3)
    #scheduler for learning rate
    #decrease
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    #increase
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=len(training_iter) * config.epoch)
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    count = 0
    loss_sum = 0
    best_val_loss = float('inf')  # For early stopping
    patience = 5  # Number of epochs to wait for improvement
    trigger_times = 0
    # Accumulate gradients over multiple steps
    accumulation_steps = 4
    scaler = torch.amp.GradScaler()
    
    # Train the model
    for epoch in range(config.epoch):
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        for step, (batch_data, batch_label) in enumerate(training_iter):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            #print(f"Batch data shape: {batch_data.shape}")
            
            with torch.amp.autocast(device_type="cuda"):  # Mixed precision training
                # Embed input data using pre-trained embeddings
                input_data = embeds(batch_data).float().permute(0, 2, 1).unsqueeze(1)
                


                # Forward pass through the model
                out = model(input_data)
                
                # Calculate loss
                loss = criterion(out, batch_label) / accumulation_steps  # Scale loss
                
                loss_sum += loss.item()
                count += 1

                if count % 100 == 0:
                    avg_loss = loss_sum / 100
                    
                    print(f"epoch {epoch}, The loss is: {avg_loss:.5f}")
                    loss_values.append(avg_loss)
                    loss_sum = 0
                    count = 0
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:  # Update weights every few steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Normalize embeddings dynamically after optimizer step
                with torch.no_grad():
                    embeds.weight.data = F.normalize(embeds.weight.data, p=2, dim=1)
                    
        # Save the model in every epoch
        model.save(os.path.join(checkpoint_dir, f'epoch{epoch}.ckpt'))

        # Validation step
        val_loss, val_accuracy = validate_model(validation_iter, model, criterion, device, embeds)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.2f}%")
        # Step the scheduler
        scheduler.step(val_loss)
        # Optionally, print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning rate:", param_group['lr'])
            
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model.ckpt'))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

        loss_sum = 0
        count = 0
        
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    plot_loss_graph(loss_values, graph_dir, filename='trainging_loss_graph.png')
    
    #test
    checkpoint_path = f'checkpoints/best_model.ckpt'  # Update this path if needed
    test_model(checkpoint_path, config, device, embeds, subset_size=1000)
    

#Testing part
def test_model(checkpoint_path, config, device, embeds, subset_size=1000):
    print("Starting test model...")
    start_time = time.time()  # Start timer
    
    # Load the model from checkpoint
    model = DPCNN(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))  # Load model parameters
    model.eval()
    
    print("Model loaded. Preparing test dataset...")
    test_data_path = 'data/train_split'
    
    # Load the word_to_index mapping from the mappings directory
    mapping_file_path = os.path.join('mappings', 'word_to_index.json')
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)

    full_testing_set  = TextDataset(path=test_data_path, word_to_index=word_to_index, augment=False)

    # Randomly sample a subset
    if len(full_testing_set) > subset_size:
        indices = random.sample(range(len(full_testing_set)), subset_size)
        subset = torch.utils.data.Subset(full_testing_set, indices)
    else:
        subset = full_testing_set

    testing_iter = data.DataLoader(dataset=subset,
                                    batch_size=config.batch_size,
                                    num_workers=16)
    
    model.to(device)
    print("GPU is available.")
    embeds.to(device)
    print("embeds is available.")
    
    print("Test dataset prepared, beginning evaluation...")
    
    correct = 0
    total = 0
    all_probs = []  # Store probabilities for analysis
    all_labels = []  # Store true labels
    
    with torch.no_grad():
        for batch_data, batch_label in tqdm(testing_iter, desc="Evaluating"):
            print("Processing batch...")

            # Ensure batch_data is a tensor
            batch_data = batch_data.to(device).long()
            batch_label = batch_label.to(device)
            
            if batch_data.max().item() >= len(word_to_index):
                print(f"Error: One or more indices exceed embedding size: {batch_data.max().item()}")
                continue  # Skip this batch

            input_data = embeds(autograd.Variable(batch_data)).to(device)
            
            # Ensure input_data is reshaped correctly
            input_data = input_data.permute(0, 2, 1).unsqueeze(1)  # [64, 1, 50, embedding_dim]

            # Process the model
            outputs = model(input_data)

            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            all_probs.append(probs.cpu().numpy())  # Store probabilities
            all_labels.append(batch_label.cpu().numpy())  # Store true labels

            # Update total and correct
            total += batch_label.size(0)
            _, predicted = torch.max(probs, 1)
            correct += (predicted == batch_label).sum().item()
        
            #print(f"Batch processed. Correct predictions so far: {correct}/{total}")

    print("Testing completed.")
            
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Testing completed in {elapsed_time:.2f} seconds.")

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save probabilities and labels for further analysis (e.g., Cleanlab)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    np.save('test_stats/test_probs.npy', all_probs)  # Save probabilities to a file
    np.save('test_stats/test_labels.npy', all_labels)  # Save true labels to a file

    print("Testing completed.")

def create_word_to_index(path):
    """
    Create a word-to-index mapping from the dataset.
    Args:
        path (str): Path to the dataset directory.
    Returns:
        dict: A dictionary mapping words to indices.
    """
    word_to_index = defaultdict(int)
    UNK_INDEX = 0
    word_to_index[""] = UNK_INDEX  # Assign UNK index

    # Read through the data to build the word-to-index mapping
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                current_data = json.loads(line.strip())
                normalized_text = normalize_text(current_data["text"])

                for word in normalized_text.split():
                    # Add regular words to vocabulary
                    if word not in word_to_index:
                        word_to_index[word] = len(word_to_index)

                    # Add negated terms (e.g., "NOT_good")
                    if word.startswith("NOT_") and word[4:] in word_to_index:
                        if word not in word_to_index:
                            word_to_index[word] = len(word_to_index)

    return dict(word_to_index)


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




def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing punctuation,
    and handling negations.
    """
    text = text.lower() #convert lowercase
    
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


def load_word_to_index(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def update_word_to_index(new_words, existing_mapping):
    """
    Update an existing word-to-index mapping with new words.
    Args:
        new_words (iterable): A list of new words to add.
        existing_mapping (dict): The existing word-to-index mapping.
    Returns:
        dict: The updated word-to-index mapping.
    """
    for word in new_words:
        # Add regular words
        if word not in existing_mapping:
            existing_mapping[word] = len(existing_mapping)

        # Add negated terms (e.g., "NOT_good")
        if word.startswith("NOT_") and word[4:] in existing_mapping:
            if word not in existing_mapping:
                existing_mapping[word] = len(existing_mapping)

    return existing_mapping


def plot_loss_graph(losses, graph_dir, filename='trainging_loss_graph.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='blue', label='Loss')
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iteration (per 100 batches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, filename))  # Save the graph as an image
    plt.close()  # Close the figure to free memory

# Validation function with detailed metrics logging
def validate_model(loader, model, criterion, device, embeds):
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            input_data = embeds(batch_data).float().permute(0, 2, 1).unsqueeze(1)
            outputs = model(input_data)
            
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    print(f"Validation Metrics:\n"
          f"Precision: {report['weighted avg']['precision']:.4f}, "
          f"Recall: {report['weighted avg']['recall']:.4f}, "
          f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    accuracy = report['accuracy'] * 100
    
    return avg_loss, accuracy

if __name__ == '__main__':
    # Train & Test the model
    train_model(config)
    