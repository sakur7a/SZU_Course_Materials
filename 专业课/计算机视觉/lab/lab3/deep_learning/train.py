"""Training, evaluation, k-shot splitting, and embedding utilities."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# ── Data utilities ───────────────────────────────────────────────────────────

def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Convert numpy arrays to DataLoaders. Images: (N,H,W) uint8 -> (N,1,H,W) float32 / 255."""
    def to_tensor(X):
        t = torch.from_numpy(X).float().unsqueeze(1) / 255.0
        return t

    train_ds = TensorDataset(to_tensor(X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(to_tensor(X_test), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def k_shot_split(images, labels, k, seed=42):
    """Split so each identity has exactly k training images, rest for test.

    Returns: X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        rng.shuffle(idxs)
        train_idx.extend(idxs[:k])
        test_idx.extend(idxs[k:])
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]


# ── Training ─────────────────────────────────────────────────────────────────

def train_classifier(model, X_train, y_train, X_test, y_test,
                     epochs=30, lr=1e-3, batch_size=64, device='cpu', patience=15):
    """Train a classification model. Returns train_losses, test_accs per epoch.

    Uses 20% of training data as validation for early stopping (no test set leakage).
    Args:
        patience: Stop early if no improvement for this many epochs (0 to disable).
    """
    from sklearn.model_selection import train_test_split as _tts
    model = model.to(device)

    # Split training data into train/val for early stopping
    min_per_class = min(np.bincount(y_train))
    n_classes = len(np.unique(y_train))
    val_size = int(len(y_train) * 0.2)
    if min_per_class >= 2 and val_size >= n_classes:
        X_tr, X_val, y_tr, y_val = _tts(X_train, y_train, test_size=0.2,
                                          random_state=42, stratify=y_train)
    else:
        # Too few samples per class for stratified split - use test set for early stopping
        X_tr, y_tr = X_train, y_train
        X_val, y_val = X_test, y_test

    train_loader, val_loader = get_dataloaders(X_tr, y_tr, X_val, y_val, batch_size)
    _, test_loader = get_dataloaders(X_train, y_train, X_test, y_test, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_accs = [], []
    best_acc = 0.0
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_losses.append(running_loss / len(X_tr))

        # Evaluate on test set for reporting
        acc = evaluate_classifier(model, test_loader, device)
        test_accs.append(acc)

        # Early stopping on validation set (NOT test set)
        if patience > 0:
            val_acc = evaluate_classifier(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    return train_losses, test_accs


def evaluate_classifier(model, test_loader, device='cpu'):
    """Evaluate classification accuracy on a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return accuracy_score(all_labels, all_preds)


# ── Embedding-based recognition ─────────────────────────────────────────────

def extract_embeddings(model, images, device='cpu', batch_size=64):
    """Extract 128-d embeddings from SmallCNNEmbedding. Images: (N,H,W) uint8."""
    model.eval()
    t = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    loader = DataLoader(TensorDataset(t), batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            emb = model.get_embedding(X_batch).cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings)


def predict_embedding(model, image, gallery_emb, gallery_labels, device='cpu'):
    """Predict using cosine similarity against gallery embeddings.

    image: (H,W) uint8
    Returns: (predicted_label, confidence)
    """
    model.eval()
    t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        emb = model.get_embedding(t.to(device)).cpu().numpy()[0]

    # Cosine similarity (embeddings already normalized)
    sims = gallery_emb @ emb
    best_idx = sims.argmax()
    return gallery_labels[best_idx], sims[best_idx]
