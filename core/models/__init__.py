#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from .GAGAvatar import GAGAvatar
from .GAGAvatar import GAGAvatar_spade32

def build_model(model_cfg, ):
    model_dict = {
        'GAGAvatar': GAGAvatar,
        'GAGAvatar_spade32': GAGAvatar_spade32,
    }
    return model_dict[model_cfg.NAME](model_cfg, )
