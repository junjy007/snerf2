from config.config import Config
from models.cyclegan import create_model

cfg = Config()

m = create_model(cfg.cyclegan_cfg)