import numpy as np

import torch
from torch.utils.data import DataLoader
from aeon.datasets import load_classification

def get_blink_data(
    bs_train: int, 
    bs_test: int, 
    whole_train: bool = False, 
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the Blink dataset from time series classification website and return the train, validation and test
    dataloaders.

    Args:
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        whole_train (bool, optional): If True, the whole dataset is used for training.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders.
    """

    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i, :], data_y[i])
            mydata.append(sample)
        return mydata
    

    x, y = load_classification("Blink")
    mapping = {"longblink": 1.0, "shortblink": 0.0}
    y = [mapping[label] for label in y]
    arr_data = np.array(x)
    arr_data = arr_data.transpose(0, 2, 1)
    arr_targets = np.array(y)  
    arr_data = torch.tensor(arr_data, dtype=torch.float32)

    train_series = arr_data[:500]
    train_targets = arr_targets[:500]

    test_series = arr_data[500:]
    test_targets = arr_targets[500:]

    if whole_train:
        valid_len = 0
    else:
        valid_len = 150     # 30% for validation
        train_idx = 500 - valid_len

        indices = np.arange(len(train_series))
        np.random.shuffle(indices)  
        train_series = train_series[indices]
        train_targets = train_targets[indices]

        valid_series = train_series[train_idx:500]
        train_series = train_series[:train_idx]

        valid_targets = train_targets[train_idx:500] 
        train_targets = train_targets[:train_idx]

    train_data, eval_data, test_data = inp_out_pairs(train_series, train_targets), inp_out_pairs(valid_series, valid_targets), inp_out_pairs(test_series, test_targets)
    train_loader = DataLoader(
        train_data, batch_size=bs_train, shuffle=True, drop_last=False
    )
    eval_loader = DataLoader(
        eval_data, batch_size=bs_test, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_data, batch_size=bs_test, shuffle=False, drop_last=False
    )
    return train_loader, eval_loader, test_loader
