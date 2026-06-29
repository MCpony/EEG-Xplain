"""
EEG 通道名称配置按模型分层索引，每个模型只检索自己的任务通道配置
"""

_CBRAMOD = {

    'mumtaz': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
           'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],

    'tuab': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],

    'tuev': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
    'shu':['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6','T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'PO9', 'O1', 'Oz', 'O2', 'PO10']
}


_LABRAM = {

   'tuab': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'],

   'tuev': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'],
}

_EEGMAMBA = {

    'tuab': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1','Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1','Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],

    'tuev': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1','Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1','Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],

    'mumtaz': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
               'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],

    'physio': ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
               'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
               'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
               'Fp1', 'Fpz', 'Fp2',
               'Af7', 'Af3', 'Afz', 'Af4', 'Af8',
               'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
               'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8',
               'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
               'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
               'O1', 'Oz', 'O2', 'Iz'],
}


# ============================================================
# BIOT - 18 channels (16 TCP bipolar + 2 reference: C3-A2, C4-A1)
# 通道顺序与 BIOT 论文一致（先左后右）
# ============================================================
_BIOT = {
    'tuab': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],

    'tuev': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],

    'chb': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
}


# ============================================================
# EEGPT - 通道名来自 eegpt.yaml 中的 use_channels_names
# 这些是 chan_conv 之后模型实际看到的通道
# ============================================================
_EEGPT = {
    # tuab: in_channels=23, IG等方法归因到23个原始输入电极
    'tuab': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'],

    # tuev: in_channels=23, 同 tuab
    'tuev': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'],

    # bcic2a: img_size[0]=19, in_channels=22, use_channels_names 19个
    'bcic2a': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
               'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'],

    # bcic2b: img_size[0]=7, in_channels=3, use_channels_names 7个
    'bcic2b': ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],

    # physio_p300: img_size[0]=58, in_channels=58, use_channels_names 58个
    'physio_p300': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'],

    # kaggle_ern: img_size[0]=19, in_channels=19, use_channels_names 19个
    'kaggle_ern': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                   'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ',
                   'P4', 'P8', 'O1', 'O2'],
}


# ============================================================
# 模型 -> 通道配置 映射表
# ============================================================
CHANNEL_CONFIGS = {
    'cbramod': _CBRAMOD,
    'labram': _LABRAM,
    'eegmamba': _EEGMAMBA,
    'biot': _BIOT,
    'eegpt': _EEGPT,
}


def get_channel_names(model_type: str, task: str, n_channels: int = None):
    """
    获取指定模型 + task 的通道名称

    Args:
        model_type: 模型名称 ('cbramod', 'labram', 'eegmamba', 'biot', 'eegpt')
        task: task 名称
        n_channels: 通道数（用于验证和 fallback）

    Returns:
        通道名称列表
    """
    # 查找模型对应的配置字典
    model_config = CHANNEL_CONFIGS.get(model_type)
    if model_config is None:
        model_config = {}

    if task and task in model_config:
        names = model_config[task]
        if n_channels is not None and len(names) != n_channels:
            raise ValueError(
                f"model='{model_type}', task='{task}' 配置了 {len(names)} 个通道，"
                f"但模型需要 {n_channels} 个通道，"
                f"请检查 channel_configs.py 中的通道配置是否正确"
            )
        return names

    # fallback: 通用命名
    if n_channels is not None:
        return [f'Ch{i+1}' for i in range(n_channels)]

    return []
 