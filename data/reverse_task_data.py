import numpy as np
from typing import Tuple, List, Dict
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
    def __init__(self, 
                 num_samples:int=1000, 
                 seq_length:int=10, 
                 vocab_size:int=20):
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

    def __len__(self)->int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx:int)->Tuple[np.ndarray, np.ndarray]:
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


def shift_right(x: np.ndarray) -> np.ndarray:
    """
    Shifts the input tensor to the right by one position along the second axis and pads with zeros.
    
    Args:
        x (np.ndarray): The input array of shape (batch_size, seq_length, num_classes).
    
    Returns:
        np.ndarray: A new array with the same shape as the input, where each sequence is shifted 
                    one position to the right, and the first position is padded with zeros.
    """
    return np.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]

def custom_collate_fn(batch: List[Tuple[np.ndarray, np.ndarray]], num_classes: int) -> Dict[str, np.ndarray]:
    """
    Custom collate function for preparing batch data for a transformer model.
    
    Args:
        batch (List[Tuple[np.ndarray, np.ndarray]]): A list of tuples where each tuple contains:
            - features: A NumPy array of shape (seq_length,).
            - labels: A NumPy array of shape (seq_length,).
        num_classes (int): The number of classes in the dataset.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the following keys:
            - "encoder_inputs": One-hot encoded features of shape (batch_size, seq_length, num_classes+1).
            - "decoder_inputs": One-hot encoded labels shifted to the right, of shape 
                                (batch_size, seq_length, num_classes+1).
            - "targets": One-hot encoded labels of shape (batch_size, seq_length, num_classes+1).
    """
    # Transpose batch to group features and labels separately
    transposed_data = list(zip(*batch))

    # Convert labels and features into NumPy arrays
    labels = np.array(transposed_data[1], dtype=np.int8)
    features = np.array(transposed_data[0], dtype=np.int8)
    
    # One-hot encode features and labels
    one_hot_features = np.eye(num_classes + 1, dtype=np.int8)[features]
    one_hot_labels = np.eye(num_classes + 1, dtype=np.int8)[labels]

    return {
        "encoder_inputs": one_hot_features,
        "decoder_inputs": shift_right(one_hot_labels),
        "targets": one_hot_labels
    }

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