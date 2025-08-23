"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

from abc import abstractmethod
from typing import List

import openai.types.chat


class LanguageModel:
    """Base class for language model interface."""

    @abstractmethod
    def chat_completion(
            self,
            message: str | List[openai.types.chat.ChatCompletionMessageParam],
            max_tokens: int,
            timeout_seconds: float,
            *args,
            **kwargs
    ):
        """Send a chat completion query with OpenAI format to the vLLM server. Return the response content.
        Args:
            message: The message in str or openai format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
        """
        pass

    def close(self):
        """Release resources (if necessary)."""
        pass

    def __del__(self):
        self.close()
