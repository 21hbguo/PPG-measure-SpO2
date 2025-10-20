# models/__init__.py
"""
Model Factory and Registry.

This file allows for dynamically selecting models based on the
configuration file (config.py).
"""

# 1. Import all model classes
from .cnn_model import CNN
from .lstm_model import LSTMModel

# from .transformer_model import TransformerModel # Example for future models

# 2. Create the Model Registry
# This dictionary maps model names (strings) to their corresponding classes.
MODEL_REGISTRY = {
    "CNN": CNN,
    "LSTMModel": LSTMModel,
    # "TransformerModel": TransformerModel, # Example for future models
}


def get_model_class(model_name):
    """
    Retrieves a model class from the registry based on its name.

    Args:
        model_name (str): The name of the model (e.g., "CNN").

    Returns:
        torch.nn.Module: The uninitialized model class.

    Raises:
        ValueError: If the model_name is not found in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models are: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name]