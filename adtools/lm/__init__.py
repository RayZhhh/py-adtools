try:
    import openai
except ImportError:
    raise ImportError('Python package "openai" is not installed.')

from adtools.lm.lm_base import LanguageModel
