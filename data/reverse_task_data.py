import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ReverseTaskDataset(Dataset):
    """
    A PyTorch Dataset for generating sequences of integers and their reversed counterparts as targets.

    Attributes:
        num_samples (int): The total number of samples in the dataset.
        seq_length (int): The length of each sequence in the dataset.
        vocab_size (int): The size of the vocabulary (range of integers in sequences).
        data (np.ndarray): The generated input sequences.
        targets (np.ndarray): The reversed sequences corresponding to the input sequences.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the input sequence and corresponding target for a given index.
    """
    def __init__(self, num_samples=1000, seq_length=10, vocab_size=20):
        """
        Initializes the dataset by generating random sequences and their reversed counterparts.

        Args:
            num_samples (int): Number of samples to generate. Default is 1000.
            seq_length (int): Length of each sequence. Default is 10.
            vocab_size (int): Range of integers [1, vocab_size) to use in sequences. Default is 20.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = np.random.randint(1, vocab_size, (num_samples, seq_length))
        self.targets = np.flip(self.data, axis=1).copy()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the input sequence and corresponding reversed target sequence for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (data[idx], targets[idx]) where:
                - data[idx] (np.ndarray): Input sequence at the specified index.
                - targets[idx] (np.ndarray): Reversed target sequence at the specified index.
        """
        return self.data[idx], self.targets[idx]


if __name__ == "__main__":
    """
    test
    """
    # Configuration parameters
    vocab_size = 20
    seq_length = 10
    batch_size = 32

    # Create dataset and split into train, validation, and test sets
    dataset = ReverseTaskDataset(num_samples=1000, seq_length=seq_length, vocab_size=vocab_size)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.9, 0.05, 0.05])

    # Create DataLoader for training and the entire dataset
    train_ds = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    # Iterate over batches in the dataset
    for features, labels in train_ds:
        print("Batch of features has shape: ", features.shape)
        print("Batch of labels has shape: ", labels.shape)
        print(features)
        print(labels)
        break
