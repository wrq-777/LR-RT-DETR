# tools/list_timm_ghost.py
import timm
all_models = timm.list_models('*ghostnet*')
v2 = [m for m in all_models if 'v2' in m.lower()]
v1 = [m for m in all_models if 'v2' not in m.lower()]
print('GhostNetV2:', v2)
print('GhostNetV1:', v1)
