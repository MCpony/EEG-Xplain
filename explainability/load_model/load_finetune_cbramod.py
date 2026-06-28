# --------------------------------------------------------
# CBraMod (Channel-Brain Model) Finetuned Model Loader
# Simplified loader for loading finetuned CBraMod models
# --------------------------------------------------------

import torch
import torch.nn as nn
from collections import OrderedDict
from model_list.cbramod import CBraModClassifier


class CBraModLoader:
    """
    Simplified CBraMod finetuned model loader

    Configuration is passed directly as parameters.
    Use YAML files (configs/cbramod.yaml) for configuration management.
    """

    @classmethod
    def load(cls, checkpoint_path, device='cuda', **config):
        """
        Load a finetuned CBraMod model

        Args:
            checkpoint_path: Path to the finetuned model checkpoint
            device: Device to load the model on ('cuda' or 'cpu')
            **config: Model configuration (n_channels, n_patches, num_classes, d_model, etc.)

        Returns:
            Loaded model ready for inference

        Example:
            >>> model = CBraModLoader.load(
            ...     './checkpoints/mumtaz_best.pth',
            ...     n_channels=19, n_patches=5, num_classes=1, d_model=100
            ... )
        """
        print(f"Creating CBraMod model with config: {config}")

        # Create model (will raise error if required params are missing)
        model = CBraModClassifier(**config)

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model weights
        state_dict = None
        for key in ['model', 'module', 'state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Found model weights with key: '{key}'")
                break
        if state_dict is None:
            state_dict = checkpoint
            print("Using checkpoint directly as model weights")

        # Clean state dict
        state_dict = cls._clean_state_dict(state_dict)

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        # Move to device and set to eval mode
        model.to(device)
        model.eval()

        # Print model info
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully!")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Device: {device}")

        return model

    @staticmethod
    def _clean_state_dict(state_dict):
        """Clean state_dict by removing prefixes"""
        new_dict = OrderedDict()

        for key, value in state_dict.items():
            # Remove 'module.' prefix (from DDP)
            if key.startswith('module.'):
                key = key[7:]
            # Remove 'model.' prefix
            if key.startswith('model.'):
                key = key[6:]

            new_dict[key] = value

        return new_dict


if __name__ == '__main__':
    print("")