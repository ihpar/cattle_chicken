import pickle
import numpy as np

from sensor import get_sensor_tuple_data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class ClfDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class ProportionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        class_to_proportion = {
            0: 0.0,    # 0% chicken
            1: 0.1,    # 10% chicken
            2: 0.2,    # 20% chicken
            3: 0.3,    # 30% chicken
            4: 0.5,    # 50% chicken
            5: 1.0     # 100% chicken
        }

        self.targets = torch.tensor(
            [[class_to_proportion[int(l)]] for l in y_t],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]


class DataOps:
    def __init__(self, random_state=42):
        self.random_state = random_state

        with open("sensor_labels_plain.pkl", "rb") as f:
            self.labels = pickle.load(f)

        with open("interpolation_functions_plain.pkl", "rb") as f:
            self.interp_funcs = pickle.load(f)

    def stratified_train_val_test_split(
        self,
        X,
        y,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        scale=True
    ):

        assert np.isclose(train_size + val_size + test_size, 1.0)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(1 - train_size),
            stratify=y,
            random_state=self.random_state
        )

        val_ratio_adjusted = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio_adjusted),
            stratify=y_temp,
            random_state=self.random_state
        )

        scaler = None

        if scale:
            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler

    def build_dataloaders(
        self,
        s_l,
        s_r,
        batch_size=8,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        type="clf"  # clf or reg
    ):
        assert type in ["clf", "reg"], "type must be either 'clf' or 'reg'"

        X_left, y, _, _ = get_sensor_tuple_data(
            0, s_l, s_r, self.interp_funcs, self.labels
        )

        X_right, _, _, _ = get_sensor_tuple_data(
            1, s_l, s_r, self.interp_funcs, self.labels
        )

        X_net = X_left - X_right

        X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
            self.stratified_train_val_test_split(
                X_net,
                y,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size
            )

        if type == "clf":
            ds_train = ClfDataset(X_train, y_train)
            ds_val = ClfDataset(X_val, y_val)
            ds_test = ClfDataset(X_test, y_test)
        else:
            ds_train = ProportionDataset(X_train, y_train)
            ds_val = ProportionDataset(X_val, y_val)
            ds_test = ProportionDataset(X_test, y_test)

        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True
        )

        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False
        )

        dl_test = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False
        )

        return dl_train, dl_val, dl_test, scaler


def main():
    dops = DataOps()

    print("CLF:\n")
    dl_train, dl_val, dl_test, _ = dops.build_dataloaders(
        0, 1, batch_size=8, type="clf"
    )

    features, labels = next(iter(dl_train))
    print(f"Train feature batch shape: {features.shape}")
    print(f"Train labels batch shape: {labels.shape}")

    features, labels = next(iter(dl_val))
    print(f"Val feature batch shape: {features.shape}")
    print(f"Val labels batch shape: {labels.shape}")

    features, labels = next(iter(dl_test))
    print(f"Test feature batch shape: {features.shape}")
    print(f"Test labels batch shape: {labels.shape}")

    print("\nREG:\n")
    dl_train, dl_val, dl_test, _ = dops.build_dataloaders(
        0, 1, batch_size=8, type="reg"
    )
    features, labels = next(iter(dl_train))
    print(labels)
    print(f"Train feature batch shape: {features.shape}")
    print(f"Train labels batch shape: {labels.shape}")

    features, labels = next(iter(dl_val))
    print(labels)
    print(f"Val feature batch shape: {features.shape}")
    print(f"Val labels batch shape: {labels.shape}")

    features, labels = next(iter(dl_test))
    print(labels)
    print(f"Test feature batch shape: {features.shape}")
    print(f"Test labels batch shape: {labels.shape}")


if __name__ == "__main__":
    main()
