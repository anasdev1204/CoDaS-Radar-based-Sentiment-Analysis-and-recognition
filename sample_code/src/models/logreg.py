import os
import pickle
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter


def _safe_report(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
    )


def _sanitize_features(X: np.ndarray, pre_clip: float = 1e6) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=pre_clip, neginf=-pre_clip)
    return np.clip(X, -pre_clip, pre_clip)


def _fit_transform_stable(X_train: np.ndarray, X_test: np.ndarray, post_clip: float = 20.0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(_sanitize_features(X_train))
    X_test = scaler.transform(_sanitize_features(X_test))
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=post_clip, neginf=-post_clip)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=post_clip, neginf=-post_clip)
    X_train = np.clip(X_train, -post_clip, post_clip)
    X_test = np.clip(X_test, -post_clip, post_clip)
    return X_train, X_test

def oversample_trainset(X_train, y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_new, y_new = [], []

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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_logreg(
    X_path,
    y_path,
    users_path=None,
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    return_metrics=False,
):
    set_seed(42)
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)
    users = np.load(users_path) if users_path is not None else None

    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    print(f"loaded X: {X.shape}, y: {y.shape}")
    X = _sanitize_features(X)

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
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    min_count = min(np.bincount(y_train))
    balanced_idx = np.hstack([
        np.random.choice(np.where(y_train == cls)[0], min_count, replace=False)
        for cls in np.unique(y_train)
    ])
    X_train, y_train = X_train[balanced_idx], y_train[balanced_idx]

    X_train, X_test = _fit_transform_stable(X_train, X_test)

    print(f"training LogisticRegression on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

    clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if num_classes == 2 else None

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\nevaluation metrics:")
    print(f"accuracy: {acc:.4f}")

    print("\ndetailed classification report:")
    print(_safe_report(y_test, y_pred, le.classes_))

    if return_metrics:
        return {"acc": float(acc), "user_acc": np.nan, "balanced_acc": float(bal_acc)}
    return acc

def run_rf(
    X_path,
    y_path,
    users_path=None,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    return_metrics=False,
):
    set_seed(42)
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)
    users = np.load(users_path) if users_path is not None else None

    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    print(f"loaded X: {X.shape}, y: {y.shape}")
    X = _sanitize_features(X)

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
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    min_count = min(np.bincount(y_train))
    balanced_idx = np.hstack([
        np.random.choice(np.where(y_train == cls)[0], min_count, replace=False)
        for cls in np.unique(y_train)
    ])
    X_train, y_train = X_train[balanced_idx], y_train[balanced_idx]

    X_train, X_test = _fit_transform_stable(X_train, X_test)

    print(f"training random forest on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if num_classes == 2 else None

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\nevaluation metrics:")
    print(f"accuracy: {acc:.4f}")

    print("\ndetailed classification report:")
    print(_safe_report(y_test, y_pred, le.classes_))

    if return_metrics:
        return {"acc": float(acc), "user_acc": np.nan, "balanced_acc": float(bal_acc)}
    return acc

def run_svm(
    X_path,
    y_path,
    users_path=None,
    C=1.0,
    kernel="rbf",
    gamma="scale",
    return_metrics=False,
):
    set_seed(42)
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)
    users = np.load(users_path) if users_path is not None else None

    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    print(f"loaded X: {X.shape}, y: {y.shape}")
    X = _sanitize_features(X)

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
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    min_count = min(np.bincount(y_train))
    balanced_idx = np.hstack([
        np.random.choice(np.where(y_train == cls)[0], min_count, replace=False)
        for cls in np.unique(y_train)
    ])
    X_train, y_train = X_train[balanced_idx], y_train[balanced_idx]

    X_train, X_test = _fit_transform_stable(X_train, X_test)

    print(f"training SVM on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

    clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if num_classes == 2 else None

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\nevaluation metrics:")
    print(f"accuracy: {acc:.4f}")

    print("\ndetailed classification report:")
    print(_safe_report(y_test, y_pred, le.classes_))

    if return_metrics:
        return {"acc": float(acc), "user_acc": np.nan, "balanced_acc": float(bal_acc)}
    return acc

def run_sklearn_louo(
    X_path,
    y_path,
    users_path,
    model_builder,
    model_name="MODEL",
    return_metrics=False,
):
    set_seed(42)

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)
    users = np.load(users_path)

    # flatten
    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    print(f"\n[LOUO-{model_name}] X={X.shape}, y={y.shape}, users={len(np.unique(users))}")

    X = _sanitize_features(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_

    unique_users = np.unique(users)

    window_accs = []
    per_user_window_acc = {}
    window_true_global = []
    window_pred_global = []
    user_preds = []
    user_true = []

    for test_user in unique_users:
        print("\n------------------------------")
        print(f"TEST USER = {test_user}")
        print("------------------------------")

        test_idx = (users == test_user)
        train_idx = ~test_idx

        X_train, y_train = X[train_idx], y_encoded[train_idx]
        X_test,  y_test  = X[test_idx], y_encoded[test_idx]

        print(f"train size={len(y_train)}, test size={len(y_test)}")
        print("train class dist:", Counter(y_train))

        X_train, y_train = oversample_trainset(X_train, y_train)
        print("after oversampling:", Counter(y_train))

        X_train, X_test = _fit_transform_stable(X_train, X_test)

        clf = model_builder()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        window_true_global.extend(y_test)
        window_pred_global.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        window_accs.append(acc)
        per_user_window_acc[str(test_user)] = float(acc)
        print(f"window acc = {acc:.4f}")

        maj_pred = np.bincount(y_pred).argmax()
        maj_true = np.bincount(y_test).argmax()

        user_preds.append(maj_pred)
        user_true.append(maj_true)

    print("\n======== LOUO SUMMARY ========")
    print(f"mean window accuracy: {np.mean(window_accs):.4f}")
    print(f"std: {np.std(window_accs):.4f}")
    bal_acc = balanced_accuracy_score(window_true_global, window_pred_global)
    print(f"window-level balanced accuracy: {bal_acc:.4f}")

    print("\n--- USER-LEVEL MAJORITY ---")
    user_level_acc = accuracy_score(user_true, user_preds)
    user_report_text = _safe_report(user_true, user_preds, classes)
    print(user_report_text)
    user_report = classification_report(
        user_true,
        user_preds,
        labels=classes,
        target_names=[str(c) for c in classes],
        zero_division=0,
        output_dict=True,
    )
    print("user-level accuracy:", user_level_acc)

    mean_window_acc = float(np.mean(window_accs))
    if return_metrics:
        return {
            "acc": mean_window_acc,
            "louo_acc_std": float(np.std(window_accs)),
            "user_acc": float(user_level_acc),
            "balanced_acc": float(bal_acc),
            "user_report": user_report,
            "per_user_window_acc": per_user_window_acc,
        }
    return mean_window_acc


def run_logreg_louo(
    X_path,
    y_path,
    users_path,
    C=1.0,
    solver="lbfgs",
    max_iter=2000,
    return_metrics=False,
):
    return run_sklearn_louo(
        X_path, y_path, users_path,
        model_builder=lambda: LogisticRegression(
            C=C, solver=solver, max_iter=max_iter, class_weight=None
        ),
        model_name="LOGREG",
        return_metrics=return_metrics,
    )

def run_rf_louo(
    X_path,
    y_path,
    users_path,
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    return_metrics=False,
):
    return run_sklearn_louo(
        X_path, y_path, users_path,
        model_builder=lambda: RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        ),
        model_name="RF",
        return_metrics=return_metrics,
    )

def run_svm_louo(
    X_path,
    y_path,
    users_path,
    C=1.0,
    kernel="rbf",
    gamma="scale",
    return_metrics=False,
):
    return run_sklearn_louo(
        X_path, y_path, users_path,
        model_builder=lambda: SVC(C=C, kernel=kernel, gamma=gamma, probability=False),
        model_name="SVM",
        return_metrics=return_metrics,
    )


def train_and_save_sklearn_full(
    X_path,
    y_path,
    save_path,
    model_builder,
    metadata=None,
    oversample=False,
):
    set_seed(42)

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)

    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    X = _sanitize_features(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train = X
    y_train = y_encoded

    if oversample:
        X_train, y_train = oversample_trainset(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(_sanitize_features(X_train))
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=20.0, neginf=-20.0)
    X_train = np.clip(X_train, -20.0, 20.0)

    clf = model_builder()
    clf.fit(X_train, y_train)

    artifact = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "metadata": metadata or {},
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(artifact, f)

    return save_path


def train_and_save_sklearn_with_validation(
    X_path,
    y_path,
    users_path,
    save_path,
    model_builder,
    metadata=None,
    oversample=False,
    val_size=0.2,
):
    set_seed(42)

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)
    users = np.load(users_path) if users_path is not None and os.path.exists(users_path) else None

    if X.ndim == 3:
        X = X.squeeze(1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    X = _sanitize_features(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    all_idx = np.arange(len(y_encoded))
    if users is not None:
        splitter = GroupShuffleSplit(test_size=val_size, n_splits=1, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y_encoded, groups=users))
    else:
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=val_size,
            stratify=y_encoded,
            random_state=42,
        )

    X_train = X[train_idx]
    y_train = y_encoded[train_idx]

    if oversample:
        X_train, y_train = oversample_trainset(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(_sanitize_features(X_train))
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=20.0, neginf=-20.0)
    X_train = np.clip(X_train, -20.0, 20.0)

    clf = model_builder()
    clf.fit(X_train, y_train)

    artifact = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "metadata": metadata or {},
        "split": {
            "type": "validation",
            "val_size": float(val_size),
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(artifact, f)

    return save_path
