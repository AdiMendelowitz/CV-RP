import numpy as np
import pytest
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network, CrossEntropyLoss
from train import SGD, train

np.random.seed(42)

_X_TRAIN = np.random.rand(32, 1, 8, 8).astype("float32")
_Y_TRAIN = np.random.randint(0, 3, 32)
_X_TEST = np.random.rand(16, 1, 8, 8).astype("float32")
_Y_TEST = np.random.randint(0, 3, 16)


def _build_model() -> Network:
    return Network(
        [
            Conv2D(1, 4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(64, 3),
            Softmax(),
        ]
    )


def test_train_returns_history() -> None:
    model = _build_model()
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    loss_fn = CrossEntropyLoss()
    history = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        X_train=_X_TRAIN,
        y_train=_Y_TRAIN,
        X_test=_X_TEST,
        y_test=_Y_TEST,
        num_epochs=2,
        batch_size=8,
        verbose=False,
    )
    assert "train_loss" in history
    assert "test_loss" in history
    assert "test_acc" in history
    assert len(history["train_loss"]) == 2


def test_loss_is_finite() -> None:
    model = _build_model()
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    loss_fn = CrossEntropyLoss()
    history = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        X_train=_X_TRAIN,
        y_train=_Y_TRAIN,
        X_test=_X_TEST,
        y_test=_Y_TEST,
        num_epochs=1,
        batch_size=8,
        verbose=False,
    )
    assert np.isfinite(history["test_loss"][0]), "test loss must be finite"
    assert 0.0 <= history["test_acc"][0] <= 1.0, "test accuracy must be in [0, 1]"
