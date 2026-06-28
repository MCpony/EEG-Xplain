# --------------------------------------------------------
# EEGMamba Finetuned Model Loader
# Simplified loader for loading finetuned EEGMamba models
# --------------------------------------------------------

import sys
import os
import torch
from collections import OrderedDict

# Add project root to path so model_list can be imported
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_list.eegmamba import (
    model_for_tuev,
    model_for_tuab,
    model_for_faced,
    model_for_seedv,
    model_for_seedvig,
    model_for_isruc,
    model_for_physio,
    model_for_chb,
    model_for_mumtaz,
    model_for_modma,
    model_for_shu,
    model_for_speech,
    model_for_stress,
    model_for_bciciv2a,
)

_DATASET_TO_CLASS = {
    'tuev':     model_for_tuev,
    'tuab':     model_for_tuab,
    'faced':    model_for_faced,
    'seedv':    model_for_seedv,
    'seedvig':  model_for_seedvig,
    'isruc':    model_for_isruc,
    'physio':   model_for_physio,
    'chb':      model_for_chb,
    'mumtaz':   model_for_mumtaz,
    'modma':    model_for_modma,
    'shu':      model_for_shu,
    'speech':   model_for_speech,
    'stress':   model_for_stress,
    'bciciv2a': model_for_bciciv2a,
}


def _get_model_class(dataset_name: str):
    """Return the dataset-specific model class from model_list.eegmamba."""
    key = dataset_name.lower()
    if key not in _DATASET_TO_CLASS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(_DATASET_TO_CLASS.keys())}"
        )
    return _DATASET_TO_CLASS[key]


class EEGMambaLoader:
    """
    EEGMamba finetuned model loader.

    The checkpoint is a raw state_dict (saved via torch.save(model.state_dict(), path)).
    Pass a param-like object or a SimpleNamespace with the fields the target Model.__init__ expects.

    Example
    -------
    >>> from types import SimpleNamespace
    >>> param = SimpleNamespace(
    ...     use_pretrained_weights=False,
    ...     foundation_dir=None,
    ...     cuda=0,
    ...     classifier='avgpooling_patch_reps',
    ...     num_of_classes=6,
    ...     dropout=0.5,
    ... )
    >>> model = EEGMambaLoader.load(
    ...     checkpoint_path='./checkpoints/tuev_best.pth',
    ...     dataset_name='tuev',
    ...     param=param,
    ... )
    """

    @classmethod
    def load(cls, checkpoint_path: str, dataset_name: str, param, device: str = 'cuda'):
        """
        Load a finetuned EEGMamba model.

        Args:
            checkpoint_path: Path to the .pth file saved by finetune_trainer.
            dataset_name:    One of 'tuev', 'tuab', 'faced', 'seedv', ... (case-insensitive).
            param:           Object whose attributes match what Model.__init__ reads
                             (use_pretrained_weights, classifier, num_of_classes, dropout, ...).
                             Set param.use_pretrained_weights=False to skip backbone weight loading
                             inside Model.__init__ (weights come from the checkpoint instead).
            device:          'cuda' or 'cpu'.

        Returns:
            Loaded model in eval mode, ready for inference / attribution.
        """
        ModelClass = _get_model_class(dataset_name)
        print(f"Creating EEGMamba model for dataset: {dataset_name}")

        # Force off so Model.__init__ doesn't try to load foundation weights;
        # the full state_dict (backbone + classifier) is in the checkpoint.
        param.use_pretrained_weights = False
        model = ModelClass(param)

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # The trainer saves state_dict directly, but handle wrapped formats too
        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ['model', 'module', 'state_dict', 'model_state']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"Found model weights under key: '{key}'")
                    break
        if state_dict is None:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")

        state_dict = cls._clean_state_dict(state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        model.float()
        model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully!")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Device: {device}")

        return model

    @staticmethod
    def _clean_state_dict(state_dict):
        """Remove DDP / wrapper prefixes."""
        new_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('model.'):
                key = key[6:]
            new_dict[key] = value
        return new_dict


if __name__ == '__main__':
    from types import SimpleNamespace

    param = SimpleNamespace(
        use_pretrained_weights=False,
        foundation_dir=None,
        cuda=0,
        classifier='avgpooling_patch_reps',
        num_of_classes=6,
        dropout=0.5,
    )

    model = EEGMambaLoader.load(
        checkpoint_path='./checkpoints/tuev_best.pth',
        dataset_name='tuev',
        param=param,
        device='cuda',
    )
    print(model)
