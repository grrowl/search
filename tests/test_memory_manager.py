import pytest
from datetime import datetime


@pytest.fixture
def memory_manager():
    manager = MemoryManager()
    manager.clear_memories()  # Ensure clean state
    return manager


def test_add_memory(memory_manager):
    content = "Test memory content"
    memory_manager.add_memory(content)
    assert len(memory_manager.memories) == 1
    assert memory_manager.memories[0]["content"] == content


def test_update_memory_importance(memory_manager):
    content = "Test memory content"
    memory_manager.add_memory(content)
    memory_id = memory_manager.memories[0]["id"]

    memory_manager.update_memory_importance(memory_id, 1)
    assert memory_manager.memories[0]["votes"] == 1
    assert memory_manager.memories[0]["importance"] > 1.0


def test_get_relevant_memories(memory_manager):
    memory_manager.add_memory("First memory")
    memory_manager.add_memory("Second memory", importance=2.0)

    memories = memory_manager.get_relevant_memories("test query")
    assert isinstance(memories, str)
    assert "Second memory" in memories


def test_clear_memories(memory_manager):
    memory_manager.add_memory("Test memory 1")
    memory_manager.add_memory("Test memory 2")
    assert len(memory_manager.memories) == 2

    memory_manager.clear_memories()
    assert len(memory_manager.memories) == 0
