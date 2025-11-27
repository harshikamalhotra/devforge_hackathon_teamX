# tests/test_memory_manager.py

import os
from src.session_store.memory_manager import MemoryManager


def test_memory_manager():
    # Use a temporary test memory file
    test_storage = "test_memory_store.json"

    # Cleanup from previous tests
    if os.path.exists(test_storage):
        os.remove(test_storage)

    mm = MemoryManager(storage_path=test_storage)

    # -----------------------------------------------------
    # 1. Create a new session
    # -----------------------------------------------------
    session_id = "test_session_1"
    mm.create_session(session_id)

    assert session_id in mm.sessions
    assert mm.get_history(session_id) == []

    # -----------------------------------------------------
    # 2. Add interactions
    # -----------------------------------------------------
    user_msg = "Hello"
    system_msg = "Hi! How can I help you?"

    mm.add_interaction(session_id, user_msg, system_msg)

    history = mm.get_history(session_id)
    assert len(history) == 1
    assert history[0]["user"] == user_msg
    assert history[0]["system"] == system_msg

    # -----------------------------------------------------
    # 3. Last interaction check
    # -----------------------------------------------------
    last = mm.last_interaction(session_id)
    assert last["user"] == user_msg
    assert last["system"] == system_msg

    # -----------------------------------------------------
    # 4. Persistence check (reload from disk)
    # -----------------------------------------------------
    mm2 = MemoryManager(storage_path=test_storage)
    history2 = mm2.get_history(session_id)

    assert len(history2) == 1
    assert history2[0]["user"] == user_msg

    # -----------------------------------------------------
    # 5. Clear history
    # -----------------------------------------------------
    mm2.clear_history(session_id)
    assert mm2.get_history(session_id) == []

    # -----------------------------------------------------
    # 6. Session deletion
    # -----------------------------------------------------
    mm2.delete_session(session_id)
    assert session_id not in mm2.sessions

    # Cleanup
    if os.path.exists(test_storage):
        os.remove(test_storage)

    print("\n=== MEMORY MANAGER TEST PASSED ===")


if __name__ == "__main__":
    test_memory_manager()
