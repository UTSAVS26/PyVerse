class BSTNode:
    def __init__(self, value: int) -> None:
        self.value = value
        self.leftPtr = None
        self.rightPtr = None
        self.height = 1  # simple balance factor calculation
        self.count = 1   # For handling duplicate values

    def __str__(self) -> str:
        return f"Node(value={self.value}, height={self.height}, count={self.count})"
