"""
Find Minimum in Rotated Sorted Array

Approach:
    Compare the middle value with the rightmost value. If the middle value is
    larger, the minimum is to its right; otherwise it is at middle or left.

Time:  O(log n)
Space: O(1)

Practical applications:
    - Finding the oldest entry after a circular, ordered log is rotated.
    - Detecting the starting point of a shifted sequence of measurements.
"""


class RotatedSortedArrayMinimum:
    """Find the smallest value in a rotated sorted list with distinct values."""

    def find_minimum(self, numbers: list[int]) -> int:
        """Return the smallest value; raise ValueError for an empty list."""
        if not numbers:
            raise ValueError("numbers must not be empty")

        left = 0
        right = len(numbers) - 1

        while left < right:
            middle = (left + right) // 2

            if numbers[middle] > numbers[right]:
                left = middle + 1
            else:
                right = middle

        return numbers[left]


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    finder = RotatedSortedArrayMinimum()

    def check_test(actual: int, expected: int) -> None:
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}")

    check_test(finder.find_minimum([3, 4, 5, 1, 2]), 1)
    check_test(finder.find_minimum([4, 5, 6, 7, 0, 1, 2]), 0)
    check_test(finder.find_minimum([11, 13, 15, 17]), 11)
    check_test(finder.find_minimum([2, 1]), 1)
    check_test(finder.find_minimum([1]), 1)

    try:
        finder.find_minimum([])
    except ValueError:
        print("Passed: empty list raises ValueError")
    else:
        raise AssertionError("Expected ValueError for an empty list")


if __name__ == "__main__":
    run_tests()


# Questions to ask yourself before choosing this algorithm:
# - Is the list sorted but rotated at an unknown pivot?
# - Do I need the smallest value rather than a target's location?
# - Can comparing the middle and rightmost values reveal the side with the pivot?
# - Are duplicate values possible, requiring a modified version of this approach?
