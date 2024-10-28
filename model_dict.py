import os
from pathlib import Path
from models import FNO_2D, FNO_3D, DeepLag_2D, DeepLag_3D, LSM_2D, LSM_3D, Factformer_2D, Factformer_3D, GNOT_2D, GNOT_3D, UNet_2D, UNet_3D, GkTrm_2D, GkTrm_3D, Vortex_2D


def get_model(args, ckpt_dir=None):
    model_dict = {
        'FNO_2D': FNO_2D,
        'FNO_3D': FNO_3D,
        'LSM_2D': LSM_2D,
        'LSM_3D': LSM_3D,
        'DeepLag_2D': DeepLag_2D,
        'DeepLag_3D': DeepLag_3D,
        'Factformer_2D': Factformer_2D, 
        'Factformer_3D': Factformer_3D, 
        'GNOT_2D': GNOT_2D, 
        'GNOT_3D': GNOT_3D, 
        'UNet_2D': UNet_2D, 
        'UNet_3D': UNet_3D, 
        'GkTrm_2D': GkTrm_2D, 
        'GkTrm_3D': GkTrm_3D, 
        'Vortex_2D': Vortex_2D,
    }
    if ckpt_dir is None:
        return model_dict[args.model].Model(args=args).cuda()
    else:
        os.system(f'cp {str(ckpt_dir)}/{args.model}.py ./models/tmp_test_model.py')
        from models import tmp_test_model
        model = tmp_test_model.Model(args=args).cuda()
        os.system(f'rm -f ./models/tmp_test_model.py')
        return model    
