"""
Search in Rotated Sorted Array

Approach:
    A rotated sorted list always has at least one sorted half. At each middle
    value, identify that half and decide whether the target belongs in it.

Time:  O(log n)
Space: O(1)

Practical applications:
    - Searching cyclic data whose original sorted order was shifted.
    - Looking up an item in a rotated ring buffer snapshot.
"""


class RotatedSortedArraySearch:
    """Search a rotated sorted list with distinct values."""

    def find_index(self, numbers: list[int], target: int) -> int:
        """Return the target's index, or -1 when the target is absent."""
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = (left + right) // 2

            if numbers[middle] == target:
                return middle

            # The left half is in normal sorted order.
            if numbers[left] <= numbers[middle]:
                if numbers[left] <= target < numbers[middle]:
                    right = middle - 1
                else:
                    left = middle + 1
            # Otherwise, the right half is in normal sorted order.
            else:
                if numbers[middle] < target <= numbers[right]:
                    left = middle + 1
                else:
                    right = middle - 1

        return -1


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    search = RotatedSortedArraySearch()

    def check_test(actual: int, expected: int) -> None:
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}")

    check_test(search.find_index([4, 5, 6, 7, 0, 1, 2], 0), 4)
    check_test(search.find_index([4, 5, 6, 7, 0, 1, 2], 6), 2)
    check_test(search.find_index([4, 5, 6, 7, 0, 1, 2], 3), -1)
    check_test(search.find_index([1], 1), 0)
    check_test(search.find_index([1], 0), -1)
    check_test(search.find_index([], 5), -1)


if __name__ == "__main__":
    run_tests()


# Questions to ask yourself before choosing this algorithm:
# - Was an originally sorted list rotated or shifted around a pivot?
# - Are the values distinct, so one half is always clearly sorted?
# - Can I identify which half is sorted by comparing the left and middle values?
# - Does the target fall inside the value range of that sorted half?
