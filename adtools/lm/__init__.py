try:
    import openai
except ImportError:
    raise ImportError('Python package "openai" is not installed.')

from .lm_base import LanguageModel
