import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BiLSTMWithFeatures(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim + feature_dim, eps=1e-5)
        self.fc = nn.Linear(hidden_dim + feature_dim, output_dim)

    def forward(self, x, feats):
        _, (hn, _) = self.lstm(x)
        lstm_out = self.dropout(hn[-1])
        combined = torch.cat([lstm_out, feats], dim=1)
        combined = self.norm(combined)
        return self.fc(combined)


def run_hybrid_rnn(X_feats_path, X_path, y_path, users_path=None, device="cpu"):
    X = np.load(X_path).astype(np.float32)
    X_feats = np.load(X_feats_path).astype(np.float32)
    y = np.load(y_path)
    users = np.load(users_path) if users_path else None

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)

    if X.ndim == 2:
        X = X[:, None, :]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    if users is not None:
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y_encoded, groups=users))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y_encoded)), test_size=0.2, stratify=y_encoded, random_state=42
        )

    X_train, X_test = X[train_idx], X[test_idx]
    Xf_train, Xf_test = X_feats[train_idx], X_feats[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    scaler = StandardScaler()
    Xf_train = scaler.fit_transform(Xf_train)
    Xf_test = scaler.transform(Xf_test)

    X_train_ts = torch.tensor(X_train, dtype=torch.float32)
    X_test_ts  = torch.tensor(X_test,  dtype=torch.float32)
    Xf_train_ts = torch.tensor(Xf_train, dtype=torch.float32)
    Xf_test_ts  = torch.tensor(Xf_test,  dtype=torch.float32)
    y_train_ts = torch.tensor(y_train, dtype=torch.long)
    y_test_ts  = torch.tensor(y_test,  dtype=torch.long)

    print(f"input dims - signal: {X_train.shape[2]}, features: {X_feats.shape[1]}")

    train_loader = DataLoader(TensorDataset(X_train_ts, Xf_train_ts, y_train_ts), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_ts,  Xf_test_ts,  y_test_ts),  batch_size=64)

    model = BiLSTMWithFeatures(
        input_dim=X.shape[2],
        feature_dim=X_feats.shape[1],
        hidden_dim=128,
        output_dim=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for batch_i, (xb, fb, yb) in enumerate(train_loader):
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, fb)
            loss = criterion(out, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/10, loss: {total_loss/max(1, len(train_loader)):.4f}")

    model.eval()
    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, fb, yb in test_loader:
            xb, fb = xb.to(device), fb.to(device)
            logits = model(xb, fb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if num_classes == 2 else np.max(probs.cpu().numpy(), axis=1))

    acc = accuracy_score(all_true, all_preds)
    bal = balanced_accuracy_score(all_true, all_preds)
    print(f"\naccuracy: {acc:.4f}")
    print(classification_report(all_true, all_preds, target_names=le.classes_))
    
    return acc