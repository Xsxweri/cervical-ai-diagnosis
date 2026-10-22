# from .data_loader import create_data_loaders
# from .preprocess import load_config, CombinedCervicalCancerDataset, get_transforms
# from .visualization import plot_training_curves, visualize_prediction, plot_confusion_matrix, plot_heatmap
# from .wsi_processor import WSIProcessor
# from .cell_analyzer import CellAnalyzer

# __all__ = [
#     'create_data_loaders', 'load_config', 'CombinedCervicalCancerDataset', 'get_transforms',
#     'plot_training_curves', 'visualize_prediction', 'plot_confusion_matrix', 'plot_heatmap',
#     'WSIProcessor', 'CellAnalyzer'
# ]

import importlib

_EXPORTS = {
    'create_data_loaders': '.data_loader',
    'load_config': '.preprocess',
    'CombinedCervicalCancerDataset': '.preprocess',
    'get_transforms': '.preprocess',
    'plot_training_curves': '.visualization',
    'visualize_prediction': '.visualization',
    'plot_confusion_matrix': '.visualization',
    'plot_heatmap': '.visualization',
    'WSIProcessor': '.wsi_processor',
    'CellAnalyzer': '.cell_analyzer',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))