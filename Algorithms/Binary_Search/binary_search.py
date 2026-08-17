"""
Binary Search

Approach:
    The list is already sorted, so compare the middle value with the target.
    Each comparison removes half of the remaining search range.

Time:  O(log n)
Space: O(1)

Practical applications:
    - Looking up a word in an alphabetized dictionary.
    - Finding a record in a sorted database index.
    - Locating a timestamp in ordered event data.
"""


class BinarySearch:
    """Find values in a sorted list."""

    def find_index(self, numbers: list[int], target: int) -> int:
        """Return the target's index, or -1 when the target is absent."""
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = (left + right) // 2
            middle_value = numbers[middle]

            if middle_value == target:
                return middle
            if middle_value < target:
                left = middle + 1
            else:
                right = middle - 1

        return -1


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    binary_search = BinarySearch()

    def check_test(actual: int, expected: int) -> None:
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}")

    check_test(binary_search.find_index([-5, -1, 0, 3, 9, 12], 9), 4)
    check_test(binary_search.find_index([-5, -1, 0, 3, 9, 12], -5), 0)
    check_test(binary_search.find_index([-5, -1, 0, 3, 9, 12], 12), 5)
    check_test(binary_search.find_index([1], 1), 0)
    check_test(binary_search.find_index([], 7), -1)
    check_test(binary_search.find_index([1, 3, 5, 7], 4), -1)


if __name__ == "__main__":
    run_tests()


# Questions to ask yourself before choosing this algorithm:
# - Is the data sorted, or can I sort it without losing needed information?
# - Do I need to find one exact value rather than inspect every value?
# - Can comparing the middle value tell me which half to discard?
# - Should a missing target return -1, None, or an insertion position?
