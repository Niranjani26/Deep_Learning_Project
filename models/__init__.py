from .restormer import Restormer
from .unet import UNetDeblur

def get_model(model_name: str = "restormer", **kwargs):
    if model_name.lower() == "unet":
        return UNetDeblur(base_channels=kwargs.get("base_channels", 32))
    return Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
