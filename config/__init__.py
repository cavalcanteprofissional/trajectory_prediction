"""
Configurações do projeto
"""
from .settings import config
from .beijing_bounds import BEIJING_BOUNDS, clamp_to_beijing, clamp_to_beijing_array, is_in_beijing

__all__ = ['config', 'BEIJING_BOUNDS', 'clamp_to_beijing', 'clamp_to_beijing_array', 'is_in_beijing']