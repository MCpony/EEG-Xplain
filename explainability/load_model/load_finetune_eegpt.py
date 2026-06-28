from collections import OrderedDict
import torch
from model_list.eegpt import EEGPTLinearProbeClassifier, EEGPTFullFinetuneClassifier


class EEGPTLoader:

    @classmethod
    def load(cls, checkpoint_path, mode='linear_probe', device='cuda', **config):
        """
        Load a fine-tuned EEGPT model.

        Args:
            checkpoint_path : path to .ckpt file
            mode            : 'linear_probe'  — BCIC2A / BCIC2B / KaggleERN / PhysioP300
                              'full_finetune' — TUEV / TUAB
            device          : 'cuda' or 'cpu'
            **config        : model constructor kwargs (see below)

        linear_probe config keys:
            num_classes, img_size, patch_stride, use_channels_names,
            in_channels, use_chan_conv, use_chan_scale, lp1_out, dropout

        full_finetune config keys:
            num_classes, img_size, patch_stride, use_channels_names,
            in_channels, use_chan_conv, head_in_features, head_dropout
        """
        if mode == 'linear_probe':
            model = EEGPTLinearProbeClassifier(**config)
        elif mode == 'full_finetune':
            model = EEGPTFullFinetuneClassifier(**config)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'linear_probe' or 'full_finetune'.")

        print(f"Loading checkpoint: {checkpoint_path}  (mode={mode})")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = None
        for key in ['state_dict', 'model', 'module']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint

        state_dict = cls._clean_state_dict(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  missing keys  ({len(missing)}): {missing}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected}")

        model.to(device).eval()
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}  device: {device}")
        return model

    @staticmethod
    def _clean_state_dict(state_dict):
        new = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('model.'):
                k = k[6:]
            new[k] = v
        return new
