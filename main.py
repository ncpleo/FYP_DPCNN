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
import torch.nn.functional as F
import re
import time
from tqdm import tqdm
import random
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from torch.utils.data import WeightedRandomSampler

# Set a random seed for PyTorch to make results reproducible (same random numbers each time we run the code).
torch.manual_seed(1)

# Create a parser to handle command-line arguments, so we can customize settings like learning rate when running the script.
parser = argparse.ArgumentParser()

# Add arguments for hyperparameters that we can set when running the script (e.g., python main.py --lr 0.01).
parser.add_argument('--lr', type=float, default=0.001)  # Learning rate: how big a step the model takes when updating weights.
parser.add_argument('--batch_size', type=int, default=256)  # Batch size: number of samples processed before updating weights.
parser.add_argument('--epoch', type=int, default=20)  # Number of epochs: how many times the model sees the entire dataset.
parser.add_argument('--gpu', type=int, default=0)  # GPU ID to use (0 for first GPU); not used directly in this script.
parser.add_argument('--label_num', type=int, default=2)  # Number of classes (2 for binary classification: positive/negative).
parser.add_argument('--seed', type=int, default=1)  # Random seed for reproducibility.
args = parser.parse_args()

# Create a Config object to store our settings, using the values from the command-line arguments.
config = Config(batch_size=args.batch_size,
                label_num=args.label_num,
                learning_rate=args.lr,
                epoch=args.epoch)

# Define paths to our data folders, where the preprocessed training, validation, test, and raw data are stored.
train_data_path = 'data/train_split'  # Folder with training data.
val_data_path = 'data/val_split'  # Folder with validation data.
test_data_path = 'data/test_split'  # Folder with test data.
raw_data_path = 'data/raw'  # Folder with raw (unprocessed) data.

# Define folders to save model checkpoints (best model weights) and graphs (like loss curves).
checkpoint_dir = 'checkpoints'  # Folder to save the best model.
graph_dir = os.path.join('pictures', 'graphs')  # Folder to save plots and graphs.
os.makedirs(checkpoint_dir, exist_ok=True)  # Create the checkpoint folder if it doesn't exist.
os.makedirs(graph_dir, exist_ok=True)  # Create the graphs folder if it doesn't exist.

# Create a folder to store the word-to-index mapping (a dictionary that assigns numbers to words).
mapping_dir = 'mappings'
os.makedirs(mapping_dir, exist_ok=True)

# Path to the GloVe pre-trained word embeddings file, which provides vector representations for words.
embedding_path = 'data/glove/glove.6B.300d.txt'

# Check if a GPU is available for faster training; if not, use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU is available.' if torch.cuda.is_available() else 'Using CPU.'}")


def train_model(config, train_data_path, val_data_path, test_data_path, device, checkpoint_dir, graph_dir):
    """
    Train the DPCNN model, evaluate it, and create plots to visualize the results.

    Args:
        config: A Config object with settings like batch size and learning rate.
        train_data_path (str): Path to the training data folder.
        val_data_path (str): Path to the validation data folder.
        test_data_path (str): Path to the test data folder.
        device: The device (CPU or GPU) to run the model on.
        checkpoint_dir (str): Folder to save the best model weights.
        graph_dir (str): Folder to save the plots and graphs.
    """
    # Load or create a word-to-index mapping, which assigns a unique number to each word in our dataset.
    mapping_file_path = os.path.join(mapping_dir, 'word_to_index.json')
    word_to_index = load_word_to_index(mapping_file_path)  # Load existing mapping if it exists.
    new_words = create_word_to_index(raw_data_path).keys()  # Get all words from the raw data.
    word_to_index = update_word_to_index(new_words, word_to_index)  # Add any new words to the mapping.
    with open(mapping_file_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f)  # Save the updated mapping to a JSON file.
    
    # Load the datasets for training, validation, and testing using our TextDataset class.
    training_set = TextDataset(path=train_data_path, word_to_index=word_to_index, augment=True)  # Training data with augmentation.
    validation_set = TextDataset(path=val_data_path, word_to_index=word_to_index, augment=False)  # Validation data, no augmentation.
    testing_set = TextDataset(path=test_data_path, word_to_index=word_to_index, augment=False)  # Test data, no augmentation.
    
    # Count how many samples belong to each class (e.g., positive, negative) in the training set.
    label_counts = Counter(training_set.labels)
    total_samples = sum(label_counts.values())  # Total number of training samples.
    
    # Calculate class weights to handle class imbalance (give more importance to underrepresented classes).
    class_weights = torch.tensor([total_samples / label_counts[i] for i in range(len(label_counts))]).to(device)
    
    # Create sample weights for balanced sampling, so the model sees underrepresented classes more often.
    sample_weights = [1 / label_counts[label] for label in training_set.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(training_set), replacement=True)
    
    # Define the loss function (CrossEntropyLoss) with class weights to handle imbalance.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create the DPCNN model and move it to the chosen device (CPU or GPU).
    model = DPCNN(config).to(device)
    
    # Load the pre-trained GloVe embeddings and create an embedding matrix for our vocabulary.
    embedding_matrix = load_pretrained_embeddings(embedding_path, word_to_index, config.word_embedding_dimension)

    # Create an embedding layer with the pre-trained weights, allowing fine-tuning (freeze=False).
    embeds = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(device)

    # Set up the optimizer (AdamW) to update the model's weights, with a small weight decay for regularization.
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    
    # Add a learning rate scheduler to reduce the learning rate if validation loss plateaus.
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Use a gradient scaler for mixed precision training, which can speed up training on GPUs.
    scaler = torch.amp.GradScaler()
    
    # Create data loaders to handle batching and shuffling of the datasets.
    training_iter = data.DataLoader(dataset=training_set, batch_size=config.batch_size, num_workers=16, sampler=sampler)
    validation_iter = data.DataLoader(dataset=validation_set, batch_size=config.batch_size, num_workers=16)
    testing_iter = data.DataLoader(dataset=testing_set, batch_size=config.batch_size, num_workers=16)
    
    # Create lists to store the loss and accuracy for each epoch, so we can plot them later.
    train_losses = []  # Training loss per epoch.
    val_losses = []  # Validation loss per epoch.
    train_accuracies = []  # Training accuracy per epoch.
    val_accuracies = []  # Validation accuracy per epoch.
    best_val_loss = float('inf')  # Keep track of the best validation loss to save the best model.
    patience = 10  # Number of epochs to wait for improvement before early stopping.
    patience_counter = 0  # Counter for early stopping.    
    
    # Start the training loop, where the model learns over multiple epochs.
    for epoch in range(config.epoch):
        model.train()  # Set the model to training mode (enables dropout, batch norm updates, etc.).
        epoch_loss = 0.0  # Total loss for this epoch.
        num_batches = 0  # Number of batches processed.
        correct_train = 0  # Number of correct predictions in training.
        total_train = 0  # Total number of training samples.
        
        # Check if gradient accumulation is set in config; if not, default to 1 (no accumulation).
        accumulation_steps = config.accumulation_steps if hasattr(config, 'accumulation_steps') else 1
        
        # Loop through each batch of training data.
        for batch_data, batch_label in training_iter:
            # Move the batch data and labels to the chosen device (CPU or GPU).
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            optimizer.zero_grad()  # Clear any previous gradients.
            
            # Use mixed precision to speed up training and reduce memory usage.
            with torch.amp.autocast(device_type="cuda"):
                # Convert word indices to embeddings, reshape for the DPCNN model, and get predictions.
                input_data = embeds(batch_data).float().permute(0, 2, 1).unsqueeze(1)
                out = model(input_data)  # Forward pass: get model predictions.
                
                # Calculate the loss (how far predictions are from true labels).
                loss = criterion(out, batch_label) / accumulation_steps
            # Backpropagate the loss to compute gradients.
            scaler.scale(loss).backward()
            
            # Add the loss for this batch to the epoch total (adjust for accumulation).
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            # Calculate training accuracy for this batch.
            _, predicted = torch.max(out, 1)  # Get the predicted class (highest score).
            total_train += batch_label.size(0)  # Add the number of samples in this batch.
            correct_train += (predicted == batch_label).sum().item()  # Count correct predictions.
            
            # Update the model weights using the optimizer.
            scaler.step(optimizer)
            scaler.update()
        
        # Calculate the average training loss and accuracy for this epoch.
        avg_train_loss = epoch_loss / num_batches
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{config.epoch}, Training Loss: {avg_train_loss:.5f}, Training Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate the model on the validation set.
        val_loss, val_accuracy, _, _, val_cm = evaluate_model(validation_iter, model, criterion, device, embeds)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Update the learning rate scheduler based on validation loss.
        #scheduler.step(val_loss)
        
        # Save the model if it has the best validation loss so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.ckpt'))
            patience_counter = 0  # Reset the patience counter.
            print(f"Best model saved with validation loss: {best_val_loss:.5f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
        
        # Early stopping: stop training if validation loss doesn't improve for 'patience' epochs.
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Load the best model (based on validation loss) and evaluate it on the test set.
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.ckpt'), weights_only=True))
    
    # Get test metrics, including probabilities for ROC and precision-recall curves.
    test_loss, test_accuracy, test_preds, test_labels, test_cm, test_probs = evaluate_model(testing_iter, model, criterion, device, embeds, return_probs=True)
    print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Print a detailed report with precision, recall, and F1-score for each class.
    print("Classification Report:\n", classification_report(test_labels, test_preds))
    
    # Show the distribution of predicted and actual labels in the test set.
    pred_counter = Counter(test_preds)  # Count predicted labels.
    label_counter = Counter(test_labels)  # Count actual labels.
    print("\nPredicted Distribution:")
    print(f"Negatives (0): {pred_counter[0]}")
    print(f"Positives (1): {pred_counter[1]}")
    print("\nActual Distribution:")
    print(f"Negatives (0): {label_counter[0]}")
    print(f"Positives (1): {label_counter[1]}")
    
    # Create a dictionary of hyperparameters to display on the plots.
    hyperparams = {
        'Learning Rate': args.lr,
        'Batch Size': args.batch_size,
        'Epochs': args.epoch
    }
    
    # Generate and save all the plots with hyperparameters displayed.
    plot_loss_graph(train_losses, val_losses, test_loss, hyperparams, graph_dir)
    plot_accuracy_graph(train_accuracies, val_accuracies, test_accuracy, hyperparams, graph_dir)
    plot_confusion_matrix(test_cm, hyperparams, graph_dir)
    plot_roc_curve(test_labels, test_probs, hyperparams, graph_dir)
    plot_precision_recall_curve(test_labels, test_probs, hyperparams, graph_dir)
    plot_class_distribution(label_counter, hyperparams, graph_dir)

def create_word_to_index(path):
    """
    Build a dictionary that maps each unique word in the dataset to a number (index).

    Args:
        path (str): Path to the folder containing the dataset files.

    Returns:
        dict: A dictionary where keys are words and values are their unique indices.
    """
    # Create a dictionary that automatically assigns 0 to new keys (used for unknown words).
    word_to_index = defaultdict(int)
    UNK_INDEX = 0  # Index for unknown words (UNK).
    word_to_index[""] = UNK_INDEX  # Assign the index 0 to the empty string (for unknown words).

    # Loop through each file in the dataset folder.
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read each line in the file (each line is a JSON object with text data).
            for line in file:
                current_data = json.loads(line.strip())  # Parse the JSON line into a Python dictionary.
                normalized_text = normalize_text(current_data["text"])  # Clean and normalize the text.

                # Split the text into words and add each new word to the dictionary.
                for word in normalized_text.split():
                    if word not in word_to_index:
                        word_to_index[word] = len(word_to_index)  # Assign the next available index.
                    
                    # Handle negated words (e.g., "NOT_good") by adding them as separate entries.
                    if word.startswith("NOT_") and word[4:] in word_to_index:
                        if word not in word_to_index:
                            word_to_index[word] = len(word_to_index)
    return dict(word_to_index)  # Convert defaultdict to a regular dictionary and return.

def load_pretrained_embeddings(embedding_path, word_to_index, embedding_dim):
    """
    Load pre-trained GloVe word embeddings and create a matrix for our vocabulary.

    Args:
        embedding_path (str): Path to the GloVe embeddings file.
        word_to_index (dict): Dictionary mapping words to their indices.
        embedding_dim (int): The size of each word embedding (e.g., 300 for GloVe 300d).

    Returns:
        torch.FloatTensor: A matrix where each row is the embedding vector for a word.
    """
    # Calculate the size of our vocabulary (number of unique words + 1 for padding).
    vocab_size = len(word_to_index) + 1
    # Create a matrix with random values for all words, which we'll overwrite with GloVe embeddings.
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    print(f"Loading pre-trained embeddings from {embedding_path}...")
    
    # Open the GloVe file and read each line.
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()  # Split the line into the word and its embedding vector.
            word = values[0]  # The first value is the word.
            vector = np.asarray(values[1:], dtype='float32')  # The rest are the embedding values.

            # Check if the embedding size matches what we expect.
            if len(vector) != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: Expected {embedding_dim}, got {len(vector)}")
            
            # If the word is in our vocabulary, add its GloVe embedding to the matrix.
            if word in word_to_index:
                index = word_to_index[word]
                embedding_matrix[index] = vector
            
            # Handle negated words (e.g., "NOT_good") by negating the original word's embedding.
            if f"NOT_{word}" in word_to_index:
                index_negated = word_to_index[f"NOT_{word}"]
                embedding_matrix[index_negated] = -vector  # Negate the vector to represent the opposite meaning.
    
    print("Pre-trained embeddings loaded successfully.")
    return torch.FloatTensor(embedding_matrix)  # Convert the matrix to a PyTorch tensor and return.

def normalize_text(text):
    """
    Clean and normalize text by converting it to lowercase, removing punctuation, and handling negations.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The cleaned and normalized text.
    """
    text = text.lower()  # Convert all characters to lowercase for consistency.
    
    # Replace special Unicode quotation marks with standard ASCII quotes.
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Single quotes.
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # Double quotes.

    # Remove any non-ASCII characters (e.g., emojis, special symbols).
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove punctuation, keeping only letters, numbers, and spaces.
    text = re.sub(r'[^\w\s]', '', text)
    
    # Define a set of negation words that indicate the next word should be negated.
    negation_words = {"not", "no", "never", "none", "don't", "doesn't", "isn't", "won't", "didn't"}
    
    # Split the text into words and process each word.
    words = text.split()
    new_words = []
    negate_next = False  # Flag to track if the next word should be negated.

    for word in words:
        if negate_next:
            new_words.append("NOT_" + word)  # Add "NOT_" prefix to the word to indicate negation.
            negate_next = False  # Reset the flag.
        elif word in negation_words:
            negate_next = True  # Set the flag to negate the next word.
        else:
            new_words.append(word)  # Keep the word as is.
    
    return ' '.join(new_words)  # Join the words back into a single string and return.

def load_word_to_index(file_path):
    """
    Load the word-to-index mapping from a JSON file if it exists.

    Args:
        file_path (str): Path to the JSON file with the mapping.

    Returns:
        dict: The word-to-index mapping, or an empty dictionary if the file doesn't exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)  # Load and return the mapping.
    return {}  # Return an empty dictionary if the file doesn't exist.

def update_word_to_index(new_words, existing_mapping):
    """
    Add new words to an existing word-to-index mapping.

    Args:
        new_words (iterable): List of new words to add.
        existing_mapping (dict): The current word-to-index mapping.

    Returns:
        dict: The updated word-to-index mapping.
    """
    for word in new_words:
        # Add the word if it's not already in the mapping.
        if word not in existing_mapping:
            existing_mapping[word] = len(existing_mapping)  # Assign the next available index.
        
        # Handle negated words (e.g., "NOT_good") by adding them as separate entries.
        if word.startswith("NOT_") and word[4:] in existing_mapping:
            if word not in existing_mapping:
                existing_mapping[word] = len(existing_mapping)
    
    return existing_mapping

def plot_loss_graph(train_losses, val_losses, test_loss, hyperparams, graph_dir, filename='loss_graph.png'):
    """
    Create a plot comparing training, validation, and test loss over epochs, with hyperparameters displayed.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        test_loss (float): The final test loss.
        hyperparams (dict): Dictionary of hyperparameters to display on the plot.
        graph_dir (str): Folder to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    epochs = range(1, len(train_losses) + 1)  # Create a range of epoch numbers (1 to number of epochs).
    plt.figure(figsize=(10, 6)) 
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')  # Plot training loss in blue.
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')  # Plot validation loss in orange.
    plt.axhline(y=test_loss, color='red', linestyle='--', label='Test Loss')  # Add a horizontal line for test loss in red.
    best_epoch = val_losses.index(min(val_losses)) + 1  # Find the epoch with the lowest validation loss.
    plt.axvline(x=best_epoch, color='green', linestyle=':', label='Best Model Epoch')  # Mark the best epoch with a green line.
    plt.title('Training, Validation, and Test Loss Comparison')  
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.legend()  
    plt.grid(True) 
    
    # Create a text box with the hyperparameters and add it to the top-left corner of the plot.
    hyperparam_text = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
    plt.text(0.98, 0.98, hyperparam_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(graph_dir, filename))  # Save the plot.
    plt.close()  

def plot_accuracy_graph(train_accuracies, val_accuracies, test_accuracy, hyperparams, graph_dir, filename='accuracy_graph.png'):
    """
    Create a plot comparing training, validation, and test accuracy over epochs, with hyperparameters displayed.

    Args:
        train_accuracies (list): List of training accuracies for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
        test_accuracy (float): The final test accuracy.
        hyperparams (dict): Dictionary of hyperparameters to display on the plot.
        graph_dir (str): Folder to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    epochs = range(1, len(train_accuracies) + 1)  # Create a range of epoch numbers.
    plt.figure(figsize=(10, 6))  
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')  # Plot training accuracy in blue.
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')  # Plot validation accuracy in orange.
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Test Accuracy')  # Add a line for test accuracy in red.
    best_epoch = val_accuracies.index(max(val_accuracies)) + 1  # Find the epoch with the highest validation accuracy.
    plt.axvline(x=best_epoch, color='green', linestyle=':', label='Best Model Epoch')  # Mark the best epoch with a green line.
    plt.title('Training, Validation, and Test Accuracy Comparison')  
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy (%)')  
    plt.legend() 
    plt.grid(True) 
    
    # Add a text box with the hyperparameters in the top-left corner.
    hyperparam_text = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
    plt.text(0.02, 0.98, hyperparam_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(graph_dir, filename))  # Save the plot.
    plt.close()  

def plot_confusion_matrix(cm, hyperparams, graph_dir, filename='confusion_matrix.png'):
    """
    Create a heatmap showing the confusion matrix, with hyperparameters displayed.

    Args:
        cm (array): The confusion matrix (true vs. predicted labels).
        hyperparams (dict): Dictionary of hyperparameters to display on the plot.
        graph_dir (str): Folder to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    plt.figure(figsize=(8, 6))  
    # Create a heatmap of the confusion matrix, with numbers shown in each cell.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')  
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label')  
    
    # Add a text box with the hyperparameters on the right side to avoid overlapping with the heatmap.
    hyperparam_text = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
    plt.text(0.7, 0.98, hyperparam_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight')  # Save the plot, ensuring the text box fits.
    plt.close()  

def plot_roc_curve(labels, probs, hyperparams, graph_dir, filename='roc_curve.png'):
    """
    Create a Receiver Operating Characteristic (ROC) curve to show how well the model distinguishes classes, with hyperparameters displayed.

    Args:
        labels (list): True labels from the test set.
        probs (list): Predicted probabilities for the positive class.
        hyperparams (dict): Dictionary of hyperparameters to display on the plot.
        graph_dir (str): Folder to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    # Calculate the false positive rate (FPR) and true positive rate (TPR) for the ROC curve.
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)  # Calculate the Area Under the Curve (AUC).
    plt.figure(figsize=(8, 6))  
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')  # Plot the ROC curve.
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Add a diagonal line (random guessing baseline).
    plt.xlim([0.0, 1.0])  # Set x-axis limits.
    plt.ylim([0.0, 1.05])  # Set y-axis limits.
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate') 
    plt.title('Receiver Operating Characteristic (ROC) Curve') 
    plt.legend(loc="lower right")  
    plt.grid(True)  
    
    # Add a text box with the hyperparameters in the top-left corner.
    hyperparam_text = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
    plt.text(0.02, 0.98, hyperparam_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(graph_dir, filename))  # Save the plot.
    plt.close() 

def plot_precision_recall_curve(labels, probs, hyperparams, graph_dir, filename='precision_recall_curve.png'):
    """
    Create a Precision-Recall curve to show the trade-off between precision and recall, with hyperparameters displayed.

    Args:
        labels (list): True labels from the test set.
        probs (list): Predicted probabilities for the positive class.
        hyperparams (dict): Dictionary of hyperparameters to display on the plot.
        graph_dir (str): Folder to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    # Calculate precision and recall values for the curve.
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))  
    plt.plot(recall, precision, color='purple', lw=2, label='Precision-Recall curve')  # Plot the curve.
    plt.xlabel('Recall') 
    plt.ylabel('Precision') 
    plt.title('Precision-Recall Curve')  
    plt.legend(loc="lower left")  
    plt.grid(True) 
    
    # Add a text box with the hyperparameters in the top-left corner.
    hyperparam_text = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
    plt.text(0.98, 0.98, hyperparam_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(graph_dir, filename))  # Save the plot.
    plt.close() 

def plot_class_distribution(label_counter, hyperparams, graph_dir, filename='class_distribution.png'):
    """
    Create a bar chart showing the distribution of classes in the test set, with hyperparameters displayed.

    Args:
        label_counter (Counter): Counts of each class in the test set.
        hyperparams (dict): Dictionary of hyperparameters to display on the plot.
        graph_dir (str): Folder to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    labels = ['Negative', 'Positive']  # Names of the classes.
    counts = [label_counter[0], label_counter[1]]  # Counts for each class.
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['blue', 'orange'])  # Create a bar chart with blue and orange bars.
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples') 
    plt.grid(True, axis='y')
    
    # Add a text box with the hyperparameters in the top-left corner.
    hyperparam_text = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
    plt.text(0.02, 0.02, hyperparam_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(graph_dir, filename))  # Save the plot.
    plt.close()

def evaluate_model(loader, model, criterion, device, embeds, return_probs=False):
    """
    Evaluate the model on a dataset (e.g., validation or test set) and return metrics like loss and accuracy.

    Args:
        loader (DataLoader): The data loader for the dataset to evaluate.
        model: The DPCNN model to evaluate.
        criterion: The loss function to calculate the loss.
        device: The device (CPU or GPU) to run the model on.
        embeds: The embedding layer to convert word indices to embeddings.
        return_probs (bool): If True, return the predicted probabilities for the positive class.

    Returns:
        tuple: Average loss, accuracy, predictions, true labels, confusion matrix, and optionally probabilities.
    """
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates, etc.).
    total_loss = 0.0  # Total loss for the dataset.
    all_preds = []  # List to store all predictions.
    all_labels = []  # List to store all true labels.
    all_probs = []  # List to store predicted probabilities (if requested).
    
    with torch.no_grad():  # Disable gradient computation for evaluation (saves memory and speeds up).
        for batch_data, batch_labels in loader:
            # Move the batch data and labels to the chosen device.
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            # Convert word indices to embeddings and reshape for the DPCNN model.
            input_data = embeds(batch_data).float().permute(0, 2, 1).unsqueeze(1)
            outputs = model(input_data)  # Get the model's predictions.
            loss = criterion(outputs, batch_labels)  # Calculate the loss.
            total_loss += loss.item()  # Add the loss for this batch.
            _, preds = torch.max(outputs, dim=1)  # Get the predicted class (highest score).
            
            # Calculate the probability of the positive class for ROC and precision-recall curves.
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(preds.cpu().numpy())  # Store predictions.
            all_labels.extend(batch_labels.cpu().numpy())  # Store true labels.
            all_probs.extend(probs.cpu().numpy())  # Store probabilities.
    
    # Calculate the average loss and accuracy for the dataset.
    avg_loss = total_loss / len(loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    cm = confusion_matrix(all_labels, all_preds)  # Create the confusion matrix.
    
    if return_probs:
        return avg_loss, accuracy, all_preds, all_labels, cm, all_probs
    return avg_loss, accuracy, all_preds, all_labels, cm

if __name__ == '__main__':
    # Start the training and evaluation process.
    train_model(config, train_data_path, val_data_path, test_data_path, device, checkpoint_dir, graph_dir)