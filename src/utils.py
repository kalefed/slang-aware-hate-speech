import torch  # Required for TensorDataset, RandomSampler, SequentialSampler, and DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split


def split_dataset(df, seed_value, x, y):
    """Splits the data into a test and training set.

    Args:
        df (DataFrame): Processed and cleaned text and sentiment data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed_value
    )
    return X_train, X_test, y_train, y_test


def create_dataloader(inputs, masks, labels, sampler_type, batch_size=32):
    """Creates the dataloader for a given dataset.

    Args:
        inputs (torch.Tensor): The input data
        masks (torch.Tensor): The attention masks
        labels (torch.Tensor): The data labels
        sampler_type (str): Type of sampler (random or sequential)

    Raises:
        ValueError: If the sampler type is not valid

    Returns:
        DataLoader: Configured DataLoader object.
    """
    data = TensorDataset(inputs, masks, labels)

    match sampler_type:
        case "random":
            sampler = RandomSampler(data)
        case "sequential":
            sampler = SequentialSampler(data)
        case _:
            raise ValueError("Invalid sampler_type. Choose 'random' or 'sequential'.")

    return DataLoader(data, sampler=sampler, batch_size=batch_size)
