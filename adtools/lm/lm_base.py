"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

from abc import abstractmethod
from typing import List, Optional

import openai.types.chat


class LanguageModel:
    """Base class for language model interface."""

    def chat_completion(
        self,
        message: str | List[openai.types.chat.ChatCompletionMessageParam],
        max_tokens: int,
        timeout_seconds: float,
        *args,
        **kwargs,
    ):
        """Send a chat completion query with OpenAI format to the vLLM server.
        Return the response content.

        Args:
            message: The message in str or openai format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
        """
        pass

    def embedding(
        self,
        text: str | List[str],
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ) -> List[float] | List[List[float]]:
        """Generate embeddings for the given text(s) using the model specified during initialization.

        Args:
            text: The text or a list of texts to embed.
            dimensions: The number of dimensions for the output embeddings.
            timeout_seconds: The timeout seconds.

        Returns:
            The embedding for the text, or a list of embeddings for the list of texts.
        """
        pass

    def close(self):
        """Release resources (if necessary)."""
        pass

    def reload(self):
        """Reload the language model (if necessary)."""
        pass

    def __del__(self):
        self.close()
