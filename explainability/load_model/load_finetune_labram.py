# --------------------------------------------------------
# LaBraM Finetuned Model Loader
# Simplified loader for loading finetuned LaBraM models
# --------------------------------------------------------

import torch
import torch.nn as nn
import pickle
from timm.models import create_model
from collections import OrderedDict
import model_list.labram  # Ensure model registration


class _NumpyCoreUnpickler(pickle.Unpickler):
    """Handle checkpoints saved with newer numpy (numpy._core → numpy.core)."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)




class LaBraMLoader:
    """
    Simplified LaBraM finetuned model loader

    Configuration is passed directly as parameters.
    Use YAML files (configs/labram.yaml) for configuration management.
    """

    @classmethod
    def load(cls, checkpoint_path, model_name='labram_base_patch200_200', device='cuda', **config):
        """
        Load a finetuned LaBraM model

        Args:
            checkpoint_path: Path to the finetuned model checkpoint
            model_name: Model architecture ('labram_base_patch200_200', 'labram_large_patch200_200', etc.)
            device: Device to load the model on ('cuda' or 'cpu')
            **config: Model configuration (num_classes, drop_rate, drop_path_rate, etc.)

        Returns:
            Loaded model ready for inference

        Example:
            >>> model = LaBraMLoader.load(
            ...     './checkpoints/tuab_best.pth',
            ...     num_classes=1, drop_path_rate=0.1
            ... )
        """
        print(f"Creating {model_name} with config: {config}")

        # Create model (will raise error if required params are missing)
        model = create_model(
            model_name,
            pretrained=False,
            **config
        )

        # Load checkpoint (compatible with newer numpy checkpoints)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu',
            pickle_module=type('_compat', (), {
                'Unpickler': _NumpyCoreUnpickler,
                'load': pickle.load,
            }),
        )

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
        """Clean state_dict by removing prefixes and unwanted keys"""
        new_dict = OrderedDict()

        for key, value in state_dict.items():
            # Remove 'module.' prefix (from DDP)
            if key.startswith('module.'):
                key = key[7:]
            # Remove 'model.' prefix
            if key.startswith('model.'):
                key = key[6:]

            # Skip relative position index (not needed for inference)
            if 'relative_position_index' in key:
                continue

            new_dict[key] = value

        return new_dict


if __name__ == '__main__':
    print("""
LaBraM Loader Usage:

# Load model with config from YAML
import yaml
with open('configs/labram.yaml') as f:
    configs = yaml.safe_load(f)['CONFIGS']

model = LaBraMLoader.load(
    './checkpoints/tuab_best.pth',
    model_name='labram_base_patch200_200',
    **configs['tuab']
)

# Or load with explicit parameters
model = LaBraMLoader.load(
    './checkpoints/tuab_best.pth',
    model_name='labram_base_patch200_200',
    num_classes=1,
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    use_rel_pos_bias=False,
    use_abs_pos_emb=True,
    qkv_bias=False,
    use_mean_pooling=True
)
    """)
