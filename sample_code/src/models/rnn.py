import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_rnn_sequence(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X
    if X.ndim == 5:
        n, r, t, k, d = X.shape
        return X.transpose(0, 2, 1, 3, 4).reshape(n, t, r * k * d)
    raise ValueError(f"[RNN] expected 3D input or radar 5D input, got shape {X.shape}")


class baselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, num_layers=1):
        super(baselineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


class baselineGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


class biLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # hn layout: [num_layers * 2, B, H]
        h_fw = hn[-2]
        h_bw = hn[-1]
        out = torch.cat([h_fw, h_bw], dim=1)
        out = self.dropout(out)
        return self.fc(out)


def _build_rnn_model(arch, input_dim, hidden_dim, output_dim, dropout, num_layers):
    if arch == "gru":
        return baselineGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
        )
    if arch == "bilstm":
        return biLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
        )
    return baselineLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        num_layers=num_layers,
    )

def run_rnn(
    X_path,
    y_path,
    users_path=None,
    device="cpu",
    hidden_dim=64,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    dropout=0.3,
    num_layers=1,
    arch="lstm",
    return_metrics=False,
):
    set_seed(42)
    X = np.load(X_path).astype(np.float32)
    X = _to_rnn_sequence(X)
    y = np.load(y_path)
    users = np.load(users_path) if users_path is not None else None

    bad_vals = np.isnan(X).sum() + np.isinf(X).sum()
    if bad_vals > 0:
        print(f"[RNN] found invalid values in X: {int(bad_vals)} -> replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-8)
    X = np.clip(X, -10.0, 10.0)

    print("\n=== RAW DATA OVERVIEW ===")
    print(f"total samples: {len(y)}")
    if users is not None:
        print(f"unique users: {len(np.unique(users))}")
    unique, counts = np.unique(y, return_counts=True)
    print("class distribution (raw):", dict(zip(unique, counts)))
    if users is not None:
        user_counts = {u: np.sum(users == u) for u in np.unique(users)}
        print(f"avg samples/user: {np.mean(list(user_counts.values())):.1f}, "
              f"min: {np.min(list(user_counts.values()))}, max: {np.max(list(user_counts.values()))}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    if users is not None:
        print("\nsplitting by user_id")
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y_encoded, groups=users))

        train_users = np.unique(users[train_idx])
        test_users = np.unique(users[test_idx])
        print(f"train users: {len(train_users)}, test users: {len(test_users)}")
        overlap = np.intersect1d(train_users, test_users)
        if len(overlap) > 0:
            print("overlap between train/test users", overlap)
        else:
            print("no overlap between train/test users")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        users_train, users_test = users[train_idx], users[test_idx]
    else:
        print("no user info provided - using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        users_train, users_test = None, None

    print("\n=== CLASS DISTRIBUTION ===")
    for split_name, y_split in [("train", y_train), ("test", y_test)]:
        u, c = np.unique(y_split, return_counts=True)
        print(f"{split_name:>5}: {dict(zip(le.inverse_transform(u), c))}")

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=max(batch_size, 64))

    model = _build_rnn_model(
        arch=arch,
        input_dim=X.shape[2],
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n=== TRAINING START ===")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch+1}/{epochs}, loss={epoch_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_true, all_users = [], [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            xb = xb.to(device)
            preds = model(xb)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_true.extend(yb.numpy())
            if users_test is not None:
                start = i * test_loader.batch_size
                end = start + len(yb)
                all_users.extend(users_test[start:end])

    acc = accuracy_score(all_true, all_preds)
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    report_labels = list(range(num_classes))
    print(
        classification_report(
            all_true,
            all_preds,
            labels=report_labels,
            target_names=le.classes_,
            zero_division=0,
        )
    )
    print(f"window-level accuracy: {acc:.4f}")
    print(f"window-level balanced accuracy: {bal_acc:.4f}")

    user_acc = math.nan
    if users_test is not None:
        print("\n=== PER-USER EVALUATION ===")
        per_user_preds = defaultdict(list)
        per_user_true = {}

        for u, true_label, pred_label in zip(all_users, all_true, all_preds):
            per_user_preds[u].append(pred_label)
            per_user_true[u] = true_label

        user_true_labels = []
        user_pred_labels = []

        for u in per_user_preds:
            preds_u = np.array(per_user_preds[u])
            majority_pred = np.bincount(preds_u).argmax()
            user_pred_labels.append(majority_pred)
            true_u = np.bincount([t for uu, t in zip(all_users, all_true) if uu == u]).argmax()
            user_true_labels.append(true_u)

        user_acc = accuracy_score(user_true_labels, user_pred_labels)
        print(f"user-level accuracy (majority vote): {user_acc:.4f}")
        print(
            classification_report(
                user_true_labels,
                user_pred_labels,
                labels=report_labels,
                target_names=le.classes_,
                zero_division=0,
            )
        )

    if return_metrics:
        return {
            "acc": float(acc),
            "user_acc": float(user_acc) if not np.isnan(user_acc) else np.nan,
            "balanced_acc": float(bal_acc),
        }
    return acc


def balance_trainset(X_train, y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))

    max_count = max(class_counts.values())

    X_balanced = []
    y_balanced = []

    for cls in unique:
        idx = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train[idx]

        needed = max_count - len(idx)
        if needed > 0:
            extra_idx = np.random.choice(idx, size=needed, replace=True)
            X_extra = X_train[extra_idx]
            y_extra = y_train[extra_idx]

            X_cls = np.concatenate([X_cls, X_extra], axis=0)
            y_cls = np.concatenate([y_cls, y_extra], axis=0)

        X_balanced.append(X_cls)
        y_balanced.append(y_cls)

    X_out = np.concatenate(X_balanced, axis=0)
    y_out = np.concatenate(y_balanced, axis=0)

    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]

def run_rnn_louo(
    X_path,
    y_path,
    users_path,
    device="cpu",
    hidden_dim=64,
    epochs=6,
    batch_size=32,
    lr=1e-3,
    dropout=0.3,
    num_layers=1,
    arch="lstm",
    return_metrics=False,
):
    set_seed(42)
    X = np.load(X_path).astype(np.float32)
    X = _to_rnn_sequence(X)
    y = np.load(y_path)
    users = np.load(users_path)

    bad_vals = np.isnan(X).sum() + np.isinf(X).sum()
    if bad_vals > 0:
        print(f"[RNN] found invalid values in X: {int(bad_vals)} -> replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-8)
    X = np.clip(X, -10.0, 10.0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    unique_users = np.unique(users)
    print(f"total samples: {len(y)}, unique users: {len(unique_users)}")

    user_accs = []
    per_user_window_acc = {}
    all_true_global = []
    all_pred_global = []

    per_user_true_label = []
    per_user_pred_label = []

    for test_user in unique_users:
        print(f"\n==============================")
        print(f"TEST USER = {test_user}")
        print("==============================")

        test_idx = (users == test_user)
        train_idx = ~test_idx

        X_train, y_train = X[train_idx], y_encoded[train_idx]
        X_train, y_train = balance_trainset(X_train, y_train)
        print("balanced train distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
        X_test,  y_test  = X[test_idx], y_encoded[test_idx]

        print(f"train size={len(y_train)}, test size={len(y_test)}")

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=max(batch_size, 64),
            shuffle=False
        )

        model = _build_rnn_model(
            arch=arch,
            input_dim=X.shape[2],
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            dropout=dropout,
            num_layers=num_layers,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            loss_sum = 0
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_sum += loss.item()
            print(f"epoch {epoch+1}/{epochs}, loss={loss_sum/len(train_loader):.4f}")

        all_preds = []
        all_true = []
        with torch.no_grad():
            model.eval()
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb)
                preds = torch.argmax(preds, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        acc = accuracy_score(all_true, all_preds)
        print(f"acc: {acc:.4f}")
        user_accs.append(acc)
        per_user_window_acc[str(test_user)] = float(acc)

        all_true_global.extend(all_true)
        all_pred_global.extend(all_preds)

        majority_pred = np.bincount(all_preds).argmax()
        majority_true = np.bincount(all_true).argmax()

        per_user_true_label.append(majority_true)
        per_user_pred_label.append(majority_pred)

    print("\n====== LOUO SUMMARY ======")
    print(f"mean window-level accuracy: {np.mean(user_accs):.4f}")
    print(f"std dev: {np.std(user_accs):.4f}")
    bal_acc = balanced_accuracy_score(all_true_global, all_pred_global)
    print(f"window-level balanced accuracy: {bal_acc:.4f}")

    print("\n--- majority vote per user ---")
    report_labels = list(range(num_classes))
    user_report_text = classification_report(
        per_user_true_label,
        per_user_pred_label,
        labels=report_labels,
        target_names=le.classes_,
        zero_division=0,
    )
    user_report = classification_report(
        per_user_true_label,
        per_user_pred_label,
        labels=report_labels,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True,
    )
    print(user_report_text)
    user_level_acc = accuracy_score(per_user_true_label, per_user_pred_label)
    print("user-level accuracy:", user_level_acc)

    mean_window_acc = float(np.mean(user_accs))
    if return_metrics:
        return {
            "acc": mean_window_acc,
            "louo_acc_std": float(np.std(user_accs)),
            "user_acc": float(user_level_acc),
            "balanced_acc": float(bal_acc),
            "user_report": user_report,
            "per_user_window_acc": per_user_window_acc,
        }
    return mean_window_acc
