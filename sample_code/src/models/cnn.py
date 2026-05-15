import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import random
from collections import Counter
from tqdm import tqdm

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class baselineCNN(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        conv1_channels=32,
        conv2_channels=64,
        dropout=0.3,
        fc_hidden=128,
    ):
        super(baselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, conv1_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes

        self.fc1 = None
        self.fc2 = None

    def _init_fc(self, x):
        flatten_dim = x.view(x.size(0), -1).shape[1]
        self.fc1 = nn.Linear(flatten_dim, self.fc_hidden).to(x.device)
        self.fc2 = nn.Linear(self.fc_hidden, self.num_classes).to(x.device)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        if self.fc1 is None:
            self._init_fc(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class residualCNN(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        conv1_channels=32,
        conv2_channels=64,
        dropout=0.3,
        fc_hidden=128,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, conv1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(conv1_channels, conv1_channels, stride=1)
        self.block2 = ResidualBlock(conv1_channels, conv2_channels, stride=2)
        self.block3 = ResidualBlock(conv2_channels, conv2_channels, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv2_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class resnetTinyCNN(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        conv1_channels=32,
        conv2_channels=64,
        dropout=0.3,
        fc_hidden=128,
    ):
        super().__init__()
        c3 = max(conv2_channels, conv1_channels * 2)
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, conv1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(conv1_channels, conv1_channels, stride=1),
            ResidualBlock(conv1_channels, conv1_channels, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(conv1_channels, conv2_channels, stride=2),
            ResidualBlock(conv2_channels, conv2_channels, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(conv2_channels, c3, stride=2),
            ResidualBlock(c3, c3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c3, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class skipConcatCNN(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        conv1_channels=32,
        conv2_channels=64,
        dropout=0.3,
        fc_hidden=128,
    ):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(num_channels, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fuse = nn.Sequential(
            nn.Conv2d(conv1_channels + conv2_channels, conv2_channels, kernel_size=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv2_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x2 = self.enc2(x2)
        x2 = self.pool2(x2)

        x1_ds = F.adaptive_avg_pool2d(x1, output_size=x2.shape[-2:])
        x = torch.cat([x1_ds, x2], dim=1)
        x = self.fuse(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


def _build_cnn_model(
    arch,
    num_channels,
    num_classes,
    conv1_channels,
    conv2_channels,
    dropout,
    fc_hidden,
):
    if arch == "residual":
        return residualCNN(
            num_channels=num_channels,
            num_classes=num_classes,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            dropout=dropout,
            fc_hidden=fc_hidden,
        )
    if arch == "resnet":
        return resnetTinyCNN(
            num_channels=num_channels,
            num_classes=num_classes,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            dropout=dropout,
            fc_hidden=fc_hidden,
        )
    if arch == "skip_concat":
        return skipConcatCNN(
            num_channels=num_channels,
            num_classes=num_classes,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            dropout=dropout,
            fc_hidden=fc_hidden,
        )
    return baselineCNN(
        num_channels=num_channels,
        num_classes=num_classes,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        dropout=dropout,
        fc_hidden=fc_hidden,
    )


def run_cnn(
    x_path,
    y_path,
    users_path,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    conv1_channels=32,
    conv2_channels=64,
    dropout=0.3,
    fc_hidden=128,
    arch="baseline",
    return_metrics=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path, allow_pickle=True)
    users = np.load(users_path, allow_pickle=True)

    print(f"\n[CNN] loaded dataset: {X.shape[0]} samples, shape {X.shape[1:]}")
    print(f"unique users: {len(np.unique(users))}")
    print(f"class distribution (raw): {Counter(y)}")

    X = (X - X.mean(axis=(2, 3), keepdims=True)) / (X.std(axis=(2, 3), keepdims=True) + 1e-8)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_encoded, groups=users))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    users_train, users_test = users[train_idx], users[test_idx]

    print("\n=== DATA SPLIT ===")
    print(f"train users: {len(np.unique(users_train))}, test users: {len(np.unique(users_test))}")
    print(f"train size: {len(X_train)}, test size: {len(X_test)}")
    print(f"class dist train: {Counter(y_train)}, test: {Counter(y_test)}")

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print(f"input channels: {X.shape[1]}, classes: {num_classes}")
    model = _build_cnn_model(
        arch=arch,
        num_channels=X.shape[1],
        num_classes=num_classes,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        dropout=dropout,
        fc_hidden=fc_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    print("\n=== TRAINING START ===")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch {epoch+1}/{epochs}, loss={total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    acc_window = accuracy_score(all_true, all_preds)
    bal_acc_window = balanced_accuracy_score(all_true, all_preds)
    print("\n=== WINDOW-LEVEL PERFORMANCE ===")
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
    print(f"window-level accuracy: {acc_window:.4f}")
    print(f"window-level balanced accuracy: {bal_acc_window:.4f}")

    user_preds, user_labels = [], []
    for u in np.unique(users_test):
        idx = np.where(users_test == u)[0]
        true_labels = y_test[idx]
        pred_labels = np.array(all_preds)[idx]
        if len(pred_labels) > 0:
            maj_pred = np.bincount(pred_labels).argmax()
            maj_true = np.bincount(true_labels).argmax()
            user_preds.append(maj_pred)
            user_labels.append(maj_true)

    acc_user = accuracy_score(user_labels, user_preds)
    print("\n=== PER-USER EVALUATION ===")
    print(f"user-level accuracy (majority vote): {acc_user:.4f}")
    print(
        classification_report(
            user_labels,
            user_preds,
            labels=report_labels,
            target_names=le.classes_,
            zero_division=0,
        )
    )
    
    if return_metrics:
        return {
            "acc": float(acc_window),
            "user_acc": float(acc_user),
            "balanced_acc": float(bal_acc_window),
        }
    return acc_window

def oversample_trainset(X_train, y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_new = []
    y_new = []

    for cls in unique:
        idx = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        y_cls = y_train[idx]

        need = max_count - len(idx)
        if need > 0:
            extra_idx = np.random.choice(idx, size=need, replace=True)
            X_extra = X_train[extra_idx]
            y_extra = y_train[extra_idx]
            X_cls = np.concatenate([X_cls, X_extra])
            y_cls = np.concatenate([y_cls, y_extra])

        X_new.append(X_cls)
        y_new.append(y_cls)

    X_new = np.concatenate(X_new)
    y_new = np.concatenate(y_new)

    perm = np.random.permutation(len(y_new))
    return X_new[perm], y_new[perm]

def run_cnn_louo(
    x_path,
    y_path,
    users_path,
    epochs=6,
    batch_size=32,
    lr=1e-3,
    conv1_channels=32,
    conv2_channels=64,
    dropout=0.3,
    fc_hidden=128,
    arch="baseline",
    return_metrics=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path, allow_pickle=True)
    users = np.load(users_path, allow_pickle=True)

    print(f"\n[LOUO-CNN] dataset: {X.shape[0]} samples, shape {X.shape[1:]}")
    print(f"unique users: {len(np.unique(users))}")
    print(f"class distribution raw: {Counter(y)}")

    X = (X - X.mean(axis=(2, 3), keepdims=True)) / (X.std(axis=(2, 3), keepdims=True) + 1e-8)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    unique_users = np.unique(users)

    window_accs = []
    per_user_window_acc = {}
    window_true_global = []
    window_pred_global = []
    user_preds_global = []
    user_true_global = []

    for test_user in unique_users:
        print("\n==================================")
        print(f"TEST USER = {test_user}")
        print("==================================")

        test_idx = (users == test_user)
        train_idx = ~test_idx

        X_train, y_train = X[train_idx], y_encoded[train_idx]
        X_test,  y_test  = X[test_idx], y_encoded[test_idx]

        print(f"train size={len(y_train)}, test size={len(y_test)}")
        print(f"train dist: {Counter(y_train)}, test dist: {Counter(y_test)}")

        train_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        train_counts[train_counts == 0] = 1.0
        class_weights = train_counts.sum() / (num_classes * train_counts)
        weight_tensor = torch.from_numpy(class_weights).to(device)
        print(f"train dist: {Counter(y_train)}")

        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        y_train = np.ascontiguousarray(y_train, dtype=np.int64)
        y_test = np.ascontiguousarray(y_test, dtype=np.int64)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = _build_cnn_model(
            arch=arch,
            num_channels=X.shape[1],
            num_classes=num_classes,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            dropout=dropout,
            fc_hidden=fc_hidden,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"{test_user} epoch {epoch+1}/{epochs}", leave=False)
            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            print(f"epoch {epoch+1}/{epochs}, loss={total_loss/len(train_loader):.4f}")

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb)
                all_preds.extend(preds.argmax(1).cpu().numpy())
                all_true.extend(yb.numpy())

        acc_window = accuracy_score(all_true, all_preds)
        window_accs.append(acc_window)
        per_user_window_acc[str(test_user)] = float(acc_window)
        window_true_global.extend(all_true)
        window_pred_global.extend(all_preds)
        print(f"[{test_user}] window acc = {acc_window:.4f}")

        maj_pred = np.bincount(all_preds).argmax()
        maj_true = np.bincount(all_true).argmax()

        user_preds_global.append(maj_pred)
        user_true_global.append(maj_true)

    print("\n========== LOUO SUMMARY ==========")
    print(f"mean window accuracy: {np.mean(window_accs):.4f}")
    print(f"std: {np.std(window_accs):.4f}")

    print("\n--- user-level majority vote ---")
    report_labels = list(range(num_classes))
    user_report_text = classification_report(
        user_true_global,
        user_preds_global,
        labels=report_labels,
        target_names=le.classes_,
        zero_division=0,
    )
    user_report = classification_report(
        user_true_global,
        user_preds_global,
        labels=report_labels,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True,
    )
    print(user_report_text)
    user_level_acc = accuracy_score(user_true_global, user_preds_global)
    bal_acc_window = balanced_accuracy_score(window_true_global, window_pred_global)
    print("user-level accuracy:", user_level_acc)
    print("window-level balanced accuracy:", bal_acc_window)

    mean_window_acc = float(np.mean(window_accs))
    if return_metrics:
        return {
            "acc": mean_window_acc,
            "louo_acc_std": float(np.std(window_accs)),
            "user_acc": float(user_level_acc),
            "balanced_acc": float(bal_acc_window),
            "user_report": user_report,
            "per_user_window_acc": per_user_window_acc,
        }
    return mean_window_acc
