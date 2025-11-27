import os
import json
from typing import Dict, Any, List


class MemoryManager:
    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Load memory from persistent storage (if exists)
        self._load()

    # --------------------------------------------------------------
    # Internal: Save + Load
    # --------------------------------------------------------------

    def _load(self):
        """Load all sessions from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except Exception:
                self.sessions = {}
        else:
            self.sessions = {}

    def _save(self):
        """Persist all memory to disk."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=2)

    # --------------------------------------------------------------
    # Session Management
    # --------------------------------------------------------------

    def create_session(self, session_id: str):
        """Create a fresh session with empty history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": []}
            self._save()

    def delete_session(self, session_id: str):
        """Delete a session completely."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save()

    def list_sessions(self) -> List[str]:
        """Return list of all active session IDs."""
        return list(self.sessions.keys())

    # --------------------------------------------------------------
    # Interaction Handling
    # --------------------------------------------------------------

    def add_interaction(self, session_id: str, user_input: str, system_response: str):
        """
        Add a conversation turn.
        Auto-creates session if not present.
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        self.sessions[session_id]["history"].append({
            "user": user_input,
            "system": system_response
        })

        self._save()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return full history for a session."""
        return self.sessions.get(session_id, {}).get("history", [])

    def clear_history(self, session_id: str):
        """Delete only conversation history."""
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []
            self._save()

    # --------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------

    def last_interaction(self, session_id: str) -> Dict[str, str] | None:
        """Return the most recent (user, system) pair."""
        history = self.get_history(session_id)
        if not history:
            return None
        return history[-1]