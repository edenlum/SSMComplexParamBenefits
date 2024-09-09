import torch
from torch.utils.data import Dataset
import unittest
import numpy as np


def delay_l2(lag):
    def delay_loss_func(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((output[:, lag:, :] - target[:, lag:, :]) ** 2)
        return loss

    return delay_loss_func


class NormalNoiseSignalGenerator():
    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("ConstSignalGenerator only supports signal dim 1")

        signals = torch.randn([num_signals, signal_length])
        return signals


class DelayedSignalDatasetRegenerated(torch.utils.data.TensorDataset):

    def __init__(self, samples_num=1, seq_length=10000, lag_length=1000,
                 signal_generator=None, lag_type="zero"):
        assert lag_length < seq_length
        assert lag_type in ["zero"]

        if signal_generator is None:
            signal_generator = NormalNoiseSignalGenerator()
            # raise ValueError("signal_generator must be provided")

        self.signal_generator = signal_generator
        self.samples_num = samples_num
        self.seq_length = seq_length
        self.lag_length = lag_length
        self.lag_type = lag_type
        super().__init__()

    def __getitem__(self, index):
        X = self.signal_generator.generate(num_signals=1,
                                           signal_length=self.seq_length)
        X = X.unsqueeze(-1)

        if self.lag_type == "zero":
            Y = torch.zeros(X.shape)
            Y[:, self.lag_length:, :] = X[:, :-self.lag_length, :]

        return X[0, :, :], Y[0, :, :]

    def __len__(self):
        return self.samples_num


def logits_to_probability(logits):
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)
    # Convert the PyTorch tensor to a NumPy array
    probabilities_numpy = probabilities.detach().numpy()
    return probabilities_numpy


class DynamicCategoricalDataset(Dataset):
    def __init__(self, epoch_size, seq_len, n_categories, lag, auto_regressive=False, copy_token=False):
        """
        Initialize the dataset with the parameters for data generation.
        Args:
            epoch_size (int): The total number of examples (data points).
            seq_len (int): The length of each sequence.
            n_categories (int): The number of categories for each element in the sequence.
            lag (int): The number of steps to shift the data for label generation.
        """
        self.amount_of_examples = epoch_size
        self.seq_len = seq_len
        self.cat_num = n_categories
        self.lag = lag
        self.auto_regressive = auto_regressive
        self.copy_token = copy_token

    def __len__(self):
        """
        Return the total number of examples you want the loader to simulate.
        """
        return self.amount_of_examples

    def __getitem__(self, idx):
        """
        Generates a single data point on demand.
        """
        data = np.random.randint(1, self.cat_num, size=(self.seq_len,))

        if self.copy_token:
            data[0] = 0
            data[self.lag] = 0
        if not self.auto_regressive:
            labels = np.zeros_like(data)
            if self.lag < self.seq_len:
                labels[self.lag:] = data[:-self.lag]
        else:
            data[self.lag:] = data[:-self.lag]
            data, labels = data[:-1], data[1:]
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class InductionHead(Dataset):
    def __init__(self, epoch_size, seq_len, n_categories, num_triggers, induction_length, auto_regressive=True):
        """
        Initialize the Induction Head dataset with the parameters for data generation.
        Args:
            epoch_size (int): The total number of examples (data points).
            seq_len (int): The length of each sequence.
            n_categories (int): The number of categories for each element in the sequence.
            num_triggers (int): The (maximum) number of times the copy token is shown in the sequence.
            induction_length (int): The length of the induction sequence (after the copy token).
        """
        self.amount_of_examples = epoch_size
        self.seq_len = seq_len
        self.cat_num = n_categories
        self.num_triggers = num_triggers
        self.induction_length = induction_length
        self.auto_regressive = auto_regressive
        self.copy_token = 0

    def __len__(self):
        """
        Return the total number of examples you want the loader to simulate.
        """
        return self.amount_of_examples

    def __getitem__(self, idx):
        """
        Generates a single data point on demand.
        """
        data = np.random.randint(1, self.cat_num, size=(self.seq_len,))
        data = np.append(data, 0)
        copy_indices = np.random.choice(np.arange(0, self.seq_len - self.induction_length), size=self.num_triggers,
                                        replace=False)
        copy_indices_filtered = []
        for i, p in enumerate(copy_indices):
            if i == 0:
                copy_indices_filtered.append(p)
            elif p - copy_indices_filtered[-1] > self.induction_length:
                copy_indices_filtered.append(p)
        to_copy = [
            data[copy_indices_filtered[0] + 1 + i]
            for i in range(self.induction_length)
        ]
        for pos in copy_indices_filtered:
            data[pos] = self.copy_token
            for i in range(self.induction_length):
                data[pos + 1 + i] = to_copy[i]
        if self.auto_regressive:
            data = np.concatenate((data, to_copy))
            data, labels = data[:-1], data[1:]
        else:
            labels = np.concatenate((data, to_copy))
            data = np.concatenate((data, np.random.randint(1, self.cat_num, size=(self.induction_length,))))
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


AR = [True, False]


class TestInductionHead(unittest.TestCase):

    def setUp(self):
        # Initialize the dataset with specific parameters
        self.epoch_size = 100
        self.seq_len = 10
        self.n_categories = 5
        self.num_triggers = 1
        self.induction_length = 9
        self.dataset = InductionHead(self.epoch_size, self.seq_len, self.n_categories, self.num_triggers,
                                     self.induction_length, auto_regressive=False)

    def test_initialization(self):
        # Check initialization parameters
        self.assertEqual(self.dataset.amount_of_examples, self.epoch_size)
        self.assertEqual(self.dataset.seq_len, self.seq_len)
        self.assertEqual(self.dataset.cat_num, self.n_categories)
        self.assertEqual(self.dataset.num_triggers, self.num_triggers)
        self.assertEqual(self.dataset.induction_length, self.induction_length)

    def test_dataset_length(self):
        # Test the dataset length
        self.assertEqual(len(self.dataset), self.epoch_size)

    def test_getitem(self):
        # Test properties of generated data points
        data, labels = self.dataset.__getitem__(0)
        self.assertEqual(data.shape[0],
                         self.seq_len + 1 + self.induction_length + (self.dataset.auto_regressive * (- 1)))
        self.assertEqual(labels.shape[0],
                         self.seq_len + 1 + self.induction_length + (self.dataset.auto_regressive * (- 1)))
        self.assertTrue((data >= 0).all() and (data < self.n_categories).all())
        self.assertTrue((labels >= 0).all() and (labels < self.n_categories).all())

        # Check for presence of copy token
        copy_token_positions = (data == self.dataset.copy_token).nonzero()[0]
        self.assertTrue(len(copy_token_positions) <= self.num_triggers)
        for pos in copy_token_positions:
            one = 1 if self.dataset.auto_regressive else 0
            print(data[pos + one:pos + one + self.induction_length], labels[pos:pos + self.induction_length])
            self.assertTrue(np.array_equal(data[pos + one:pos + one + self.induction_length],
                                           labels[pos:pos + self.induction_length]))


class TestDelay(unittest.TestCase):

    def setUp(self):
        # Initialize the dataset with specific parameters
        self.epoch_size = 100
        self.seq_len = 10
        self.n_categories = 5
        self.num_triggers = 1
        self.lag = 5
        self.dataset = DynamicCategoricalDataset(self.epoch_size, self.seq_len, self.n_categories, self.lag,
                                                 auto_regressive=True, copy_token=True)

    def test_getitem(self):
        # Test properties of generated data points
        data, labels = self.dataset.__getitem__(0)
        self.assertTrue((data >= 0).all() and (data < self.n_categories).all())
        self.assertTrue((labels >= 0).all() and (labels < self.n_categories).all())

        self.assertTrue(np.array_equal(data[:-self.lag + 1], labels[self.lag:]))


if __name__ == '__main__':
    unittest.main()
