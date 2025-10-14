


def get_model(name,  **kwargs):
    """
    Factory function to get the model based on the model name.
    
    Args:
        model_name (str): Name of the model to retrieve.
        **kwargs: Additional keyword arguments for model initialization.
        
    Returns:
        torch.nn.Module: The initialized model.
    """
    if name == 'unet3d':
        from icecream.models.unet3d import UNet3D 
        return UNet3D(**kwargs)
    if name == 'unet3d_bf':
        from icecream.models.unet3d_bf import UNet3D
        return UNet3D(**kwargs)
    elif name == 'iso_unet3d':
        from icecream.models.iso_unet import Unet3D as iso_Unet3D
        return iso_Unet3D(**kwargs)
    else:
        raise ValueError(f"Model {name} is not recognized.")
