"""painnet -- multimodal pain assessment from EEG + wristband signals.

PSU AI 570 team project. Jonathan Miller, Meng Li, Raj Mamidala.

Typical notebook opening:

    from painnet import config, data, windows, splits, models, evaluate, plots
    print(config.describe())          # where am I, is the data here
    X, y, groups = windows.build_dataset()

The important thing to know before touching anything: every subject in this
dataset has exactly ONE pain label for their whole recording, so windows must
be split by subject. See splits.py -- it enforces that.
"""

__version__ = "0.1.0"

from . import config  # noqa: F401  (cheap, and gives `painnet.config` for free)

__all__ = [
    "config",
    "data",
    "windows",
    "splits",
    "models",
    "evaluate",
    "plots",
]
