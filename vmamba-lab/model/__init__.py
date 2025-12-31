from .backbone import VMamba, VSSBlock, SS2D
from .wmamba import WMamba
from .forma import ForMa

__all__ = [
    # Backbone
    "VMamba",
    "VSSBlock",
    "SS2D",
    # Models
    "WMamba",
    "ForMa",
]


def build_model(config: dict):
    """Build model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Model instance
    """
    model_name = config.get("model", {}).get("name", "wmamba")

    if model_name == "wmamba":
        return WMamba(config)
    elif model_name == "forma":
        return ForMa(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
