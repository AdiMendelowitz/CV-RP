"""Pytest configuration for the adversarial-ml-toolkit.

Placed at the toolkit root so that "attacks", "models" and "defenses" are importable during collection and so the
shared markers are registered once.
"""


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: needs the trained ResNet-18 checkpoint and the CIFAR-10 test set",
    )