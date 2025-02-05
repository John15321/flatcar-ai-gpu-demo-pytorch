"""Package version."""

# Dont touch anything here

import importlib.metadata

__version__ = importlib.metadata.version(__package__)


from .flatcar_ai_gpu_demo_pytorch import (
    main_download_fashion_mnist_samples,
    main_predict,
    main_train,
)
