"""
chat_history.py
Manage conversation history for context-aware multi-turn QA.
"""


class ChatHistory:
    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: Maximum number of Q&A pairs to keep in memory.
        """
        self._history: list[dict] = []
        self.max_turns = max_turns

    def add(self, user_msg: str, bot_msg: str) -> None:
        """Add a user/bot turn."""
        self._history.append({"user": user_msg, "bot": bot_msg})
        # Keep only the last N turns
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    def get_history(self) -> list[dict]:
        """Return all turns as a list of dicts."""
        return self._history

    def get_context(self, max_turns: int = 3) -> str:
        """
        Return the last N turns as a formatted string for RAG context injection.

        Args:
            max_turns: How many recent turns to include.

        Returns:
            Formatted conversation string.
        """
        recent = self._history[-max_turns:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['bot']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset history."""
        self._history = []

    def __len__(self) -> int:
        return len(self._history)
