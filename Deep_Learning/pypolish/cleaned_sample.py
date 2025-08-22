import json
import math
import os
import sys
from datetime import datetime


def calc(x: int) -> None:
    """TODO: Add docstring for calc function."""
    if x % 2 == 0:
        print("Even")
    else:
        print("Odd")


def long_function_with_many_lines() -> None:
    """TODO: Add docstring for long_function_with_many_lines function."""
    print("This is line 1")
    print("This is line 2")
    print("This is line 3")
    print("This is line 4")
    print("This is line 5")
    print("This is line 6")
    print("This is line 7")
    print("This is line 8")
    print("This is line 9")
    print("This is line 10")
    print("This is line 11")
    print("This is line 12")
    print("This is line 13")
    print("This is line 14")
    print("This is line 15")
    print("This is line 16")
    print("This is line 17")
    print("This is line 18")
    print("This is line 19")
    print("This is line 20")
    print("This is line 21")
    print("This is line 22")
    print("This is line 23")
    print("This is line 24")
    print("This is line 25")


def infinite_loop() -> None:
    """TODO: Add docstring for infinite_loop function."""
    while True:
        print("This will run forever")


def simple_if_else(x: int) -> str:
    """TODO: Add docstring for simple_if_else function."""
    if x > 0:
        return "positive"
    else:
        return "negative"


def list_append_example():
    """TODO: Add docstring for list_append_example function."""
    result = []
    for i in range(10):
        result.append(i * 2)
    return result


class Calculator:
    """TODO: Add docstring for Calculator class."""

    def __init__(self):
        """TODO: Add docstring for __init__ function."""
        self.value = 0

    def add(self, x, y):
        """TODO: Add docstring for add function."""
        return x + y

    def multiply(self, x, y):
        """TODO: Add docstring for multiply function."""
        return x * y


def main() -> None:
    """TODO: Add docstring for main function."""
    calc = Calculator()
    result = calc.add(5, 3)
    print(result)
    return result
