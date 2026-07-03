import sys, pprint
from src.core.yaml_config import YAMLConfig
from src.core.yaml_utils import GLOBAL_CONFIG

cfg = YAMLConfig(sys.argv[1])
print("=== merged cfg['model'] ===")
pprint.pprint(cfg.yaml_cfg.get('model'), width=120)

print("\n=== GLOBAL_CONFIG has GhostNetV2Backbone:", 'GhostNetV2Backbone' in GLOBAL_CONFIG)
if 'GhostNetV2Backbone' in GLOBAL_CONFIG:
    schema = GLOBAL_CONFIG['GhostNetV2Backbone']
    print("_name in schema:", isinstance(schema, dict) and ('_name' in schema))
