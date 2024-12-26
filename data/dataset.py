import re
from torch.utils import data
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet
import random
import os
import json

class TextDataset(data.Dataset):
    def __init__(self, path, word_to_index, max_length=50, augment=False):
        """
        Initialize the dataset.
        Args:
            path (str): Path to the JSONL dataset.
            word_to_index (dict): Mapping of words to indices.
            max_length (int): Maximum sequence length for padding/truncation.
            augment (bool): Whether to apply data augmentation.
        """
        self.file_name = os.listdir(path)
        self.data_set = []
        self.labels = []
        self.word_to_index = word_to_index
        self.max_length = max_length
        self.augment = augment

        # Load preprocessed JSONL files
        for file_name in self.file_name:
            file_path = os.path.join(path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    item = json.loads(line.strip())
                    text = item["text"]
                    label = item['label']
                    
                    # Normalize text
                    text = self.normalize_text(text)

                    # Apply augmentation if enabled
                    if self.augment:
                        text = self.apply_augmentation(text)

                    # Tokenize and convert to indices
                    tokens = text.split()
                    indices = [self.word_to_index.get(word, 0) for word in tokens]
                    
                    # Skip empty texts
                    if not indices:  # Check if the list is empty
                        continue
                    
                    # Pad or truncate to match sentence_max_size
                    if len(indices) < self.max_length:
                        indices += [0] * (self.max_length - len(indices))  # Pad with zeros
                    else:
                        indices = indices[:self.max_length]  # Truncate if too long
                        
                    
                    self.data_set.append(indices)
                    self.labels.append(label)

        print(f"Number of labels loaded: {len(self.labels)}")

    def normalize_text(self, text):
        """Normalize text by converting to lowercase and removing punctuation."""
        text = text.lower() #convert to lowercase
        return re.sub(r'[^\w\s]', '', text) #remove punctuation

    def apply_augmentation(self, text):
        """Randomly apply one of several augmentations."""
        augmentation_methods = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.random_deletion,
        ]
        augmentation_method = random.choice(augmentation_methods)
        return augmentation_method(text)

    def synonym_replacement(self, text):
        """Replace words with their synonyms."""
        words = text.split()
        if not words:  # Check if the list is empty
            return text  # Return original text if no words are present

        new_words = words.copy()
        for i, word in enumerate(words):
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()  # Get a synonym
                new_words[i] = synonym.replace('_', ' ')  # Replace with the synonym
        return ' '.join(new_words)


    def random_insertion(self, text, n=1):
        """Randomly insert synonyms into the sentence."""
        words = text.split()
        if not words:  # Check if the list is empty
            return text  # Return original text if no words are present

        for _ in range(n):
            new_word_position = random.randint(0, len(words))
            random_word = random.choice(words)
            synonyms = wordnet.synsets(random_word)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                words.insert(new_word_position, synonym.replace('_', ' '))
        return ' '.join(words)


    def random_swap(self, text, n=1):
        """Randomly swap two words in the sentence."""
        words = text.split()
        if not words:  # Check if the list is empty
            return text  # Return original text if no words are present
    
        for _ in range(n):
            if len(words) < 2:
                return text
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def random_deletion(self, text, p=0.2):
        """Randomly delete words from a sentence."""
        words = text.split()
        if len(words) <= 1:  # Avoid empty sentences
            return text

        return ' '.join([word for word in words if random.random() > p])
        

    def __getitem__(self, index):
        """
        Return tokenized and padded/truncated sequences.
        Args:
            index (int): Index of the sample.
        Returns:
            tensor_indices (torch.Tensor): Tokenized and padded sequence.
            label (int): Corresponding label (0 or 1).
        """
        tensor_indices = torch.tensor(self.data_set[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return tensor_indices, label

    def __len__(self):
        return len(self.data_set)
