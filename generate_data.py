"""Явно пересоздать входные данные из config.json (перезаписывает data)."""
from model import generate
from common import ROOT,write_json
import json
cfg=json.loads((ROOT/'config.json').read_text())
write_json(ROOT/'data/weather.json',generate(cfg['data_seed'],cfg['days']))
