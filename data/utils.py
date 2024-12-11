import numpy as np
from typing import Tuple, List, Dict

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
    def test_shift_right():
        # Create a sample input
        x = np.array([[[1, 0], [0, 1], [1, 0]],
                    [[0, 1], [1, 0], [0, 1]]], dtype=np.int8)
        
        # Expected output (shifted right and padded with zeros)
        expected_output = np.array([[[0, 0], [1, 0], [0, 1]],
                                    [[0, 0], [0, 1], [1, 0]]], dtype=np.int8)

        # Call the shift_right function
        output = shift_right(x)

        # Assert that the output matches the expected result
        np.testing.assert_array_equal(output, expected_output)

    # Test for custom_collate_fn function
    def test_custom_collate_fn():
        # Define a sample batch of tuples (features, labels)
        batch = [
            (np.array([0, 1, 2], dtype=np.int8), np.array([2, 1, 0], dtype=np.int8)),
            (np.array([1, 2, 0], dtype=np.int8), np.array([1, 0, 2], dtype=np.int8)),
        ]
        num_classes = 3

        # Expected output for one-hot encoding the features and labels
        expected_encoder_inputs = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                        [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]], dtype=np.int8)
        
        expected_decoder_inputs = np.array([[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
                                        [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]], dtype=np.int8)
        
        expected_targets = np.array([[[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                                    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]], dtype=np.int8)

        # Call custom_collate_fn
        collated_data = custom_collate_fn(batch, num_classes)

        # Assertions
        np.testing.assert_array_equal(collated_data["encoder_inputs"], expected_encoder_inputs)
        np.testing.assert_array_equal(collated_data["decoder_inputs"], expected_decoder_inputs)
        np.testing.assert_array_equal(collated_data["targets"], expected_targets)

    test_shift_right()
    test_custom_collate_fn()