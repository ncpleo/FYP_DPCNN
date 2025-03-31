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
                    text = self.normalize_text(item["text"])
                    label = item['label']

                    # Apply augmentation if enabled
                    if self.augment:
                        text = self.apply_augmentation(text)

                    indices = [self.word_to_index.get(word, 1) for word in text.split()]  # 1 for UNK, 0 for padding
                    if not indices:
                        continue
                    
                    if len(indices) < self.max_length:
                        indices += [0] * (self.max_length - len(indices))
                    else:
                        indices = indices[:self.max_length]
                    
                    self.data_set.append(indices)
                    self.labels.append(label)

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
        text = ' '.join([k for k, v in self.word_to_index.items() if v in self.data_set[index]])
        if self.augment and random.random() < 0.5:  # Apply augmentation dynamically
            text = self.apply_augmentation(text)
            indices = [self.word_to_index.get(word, 1) for word in text.split()]
            if len(indices) < self.max_length:
                indices += [0] * (self.max_length - len(indices))
            else:
                indices = indices[:self.max_length]
        else:
            indices = self.data_set[index]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.data_set)