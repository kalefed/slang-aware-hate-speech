import torch  # Required for TensorDataset, RandomSampler, SequentialSampler, and DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class Dataloader:
    """
    Class that creates the dataloaders for the given dataset to be used to train BERT model.
    """

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def create_dataloader(self, inputs, masks, labels, sampler_type):
        """
        Creates the dataloader for a given dataset.

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
                raise ValueError(
                    "Invalid sampler_type. Choose 'random' or 'sequential'."
                )

        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        return dataloader
