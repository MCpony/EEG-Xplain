# --------------------------------------------------------
# BIOT Finetuned Model Loader
# Simplified loader for loading finetuned BIOT models
# --------------------------------------------------------

import torch
from collections import OrderedDict
from model_list.biot import BIOTClassifier


class BIOTLoader:
    """
    Simplified BIOT finetuned model loader

    Configuration is passed directly as parameters.
    Use YAML files (configs/biot.yaml) for configuration management.
    """

    @classmethod
    def load(cls, checkpoint_path, device='cuda', **config):
        """
        Load a finetuned BIOT model

        Args:
            checkpoint_path: Path to the finetuned model checkpoint
            device: Device to load the model on ('cuda' or 'cpu')
            **config: Model configuration (n_channels, num_classes, emb_size, etc.)

        Returns:
            Loaded model ready for inference

        Example:
            >>> model = BIOTLoader.load(
            ...     './checkpoints/tuab_best.pth',
            ...     n_channels=16, num_classes=1
            ... )
        """
        model_config = {
            'emb_size': config.get('emb_size', 256),
            'heads': config.get('heads', 8),
            'depth': config.get('depth', 4),
            'n_classes': config.get('num_classes', 6),
            'n_channels': config.get('n_channels', 16),
            'n_fft': config.get('n_fft', 200),
            'hop_length': config.get('hop_length', 100),
        }
        print(f"Creating BIOT model with config: {model_config}")

        model = BIOTClassifier(**model_config)

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = None
        for key in ['model', 'module', 'state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Found model weights with key: '{key}'")
                break
        if state_dict is None:
            state_dict = checkpoint
            print("Using checkpoint directly as model weights")

        state_dict = cls._clean_state_dict(state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully!")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Device: {device}")

        return model

    @staticmethod
    def _clean_state_dict(state_dict):
        """Clean state_dict by removing prefixes"""
        new_dict = OrderedDict()

        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('model.'):
                key = key[6:]

            new_dict[key] = value

        return new_dict


if __name__ == '__main__':
    print("""
BIOT Loader Usage:

# Load model with config from YAML
import yaml
with open('configs/biot.yaml') as f:
    configs = yaml.safe_load(f)['CONFIGS']

model = BIOTLoader.load(
    './checkpoints/tuab_best.pth',
    **configs['tuab']
)

# Or load with explicit parameters
model = BIOTLoader.load(
    './checkpoints/tuab_best.pth',
    n_channels=16,
    num_classes=1,
    emb_size=256,
    heads=8,
    depth=4,
    n_fft=200,
    hop_length=100
)
    """)
