from .swin_unetr import SwinUNETR


def define_model(configs):
    
    if configs.name == 'swin_unetr':
        model = SwinUNETR(**configs.params)
    else:
        raise NotImplementedError(f'unknown model name {configs.name}')

    return model
