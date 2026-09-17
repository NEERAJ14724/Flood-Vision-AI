import os
import torch

path = r"c:\Users\ratho\OneDrive\Desktop\nielit project\flood\flood_output_v4_complete\checkpoint_best.pth"
print('exists', os.path.exists(path))
ckpt = torch.load(path, map_location='cpu')
print('keys', list(ckpt.keys()))
print('epoch', ckpt.get('epoch'))
print('best_miou', ckpt.get('best_miou'))
print('layers', len(ckpt['model_state']))
print('sample', list(ckpt['model_state'].keys())[:20])
