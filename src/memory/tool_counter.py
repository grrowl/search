class ToolUsageCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.thought_count = 0
        self.search_count = 0
        self.visit_count = 0
        self.memory_count = 0

    def count_tool(self, tool_name: str):
        if tool_name in ["search", "google_search"]:
            self.search_count += 1
        elif tool_name == "visit":
            self.visit_count += 1
        self.thought_count += 1

    def set_memory_count(self, count: int):
        self.memory_count = count

    def get_summary(self) -> str:
        parts = []
        if self.thought_count > 0:
            parts.append(f"Thought {self.thought_count} times")
        if self.search_count > 0:
            parts.append(
                f"performed {self.search_count} search{'' if self.search_count == 1 else 'es'}"
            )
        if self.visit_count > 0:
            parts.append(
                f"made {self.visit_count} visit{'' if self.visit_count == 1 else 's'}"
            )
        if self.memory_count > 0:
            parts.append(
                f"used {self.memory_count} relevant memor{'y' if self.memory_count == 1 else 'ies'}"
            )

        if not parts:
            return "No tools used"

        return ", ".join(parts)
