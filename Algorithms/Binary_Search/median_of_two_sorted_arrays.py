"""
Median of Two Sorted Arrays

Approach:
    Binary-search a partition in the shorter list. The correct partition puts
    half of all values on each side, with every left-side value <= every
    right-side value.

Time:  O(log(min(m, n)))
Space: O(1)

Practical applications:
    - Combining two ordered streams without fully merging them.
    - Computing the middle value of two separately sorted data sets.
"""


class MedianOfTwoSortedArrays:
    """Find the median of two individually sorted lists."""

    def find_median(self, first: list[int], second: list[int]) -> float:
        """Return the combined median; raise ValueError when both lists are empty."""
        if not first and not second:
            raise ValueError("at least one list must contain a number")

        # Search the shorter list so the partition range stays small.
        if len(first) > len(second):
            first, second = second, first

        first_length = len(first)
        second_length = len(second)
        left = 0
        right = first_length
        left_partition_size = (first_length + second_length + 1) // 2

        while left <= right:
            first_cut = (left + right) // 2
            second_cut = left_partition_size - first_cut

            first_left = float("-inf") if first_cut == 0 else first[first_cut - 1]
            first_right = float("inf") if first_cut == first_length else first[first_cut]
            second_left = float("-inf") if second_cut == 0 else second[second_cut - 1]
            second_right = float("inf") if second_cut == second_length else second[second_cut]

            if first_left <= second_right and second_left <= first_right:
                if (first_length + second_length) % 2 == 1:
                    return float(max(first_left, second_left))
                return (max(first_left, second_left) + min(first_right, second_right)) / 2

            if first_left > second_right:
                right = first_cut - 1
            else:
                left = first_cut + 1

        raise ValueError("input lists must be sorted")


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    median_finder = MedianOfTwoSortedArrays()

    def check_test(actual: float, expected: float) -> None:
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}")

    check_test(median_finder.find_median([1, 3], [2]), 2.0)
    check_test(median_finder.find_median([1, 2], [3, 4]), 2.5)
    check_test(median_finder.find_median([], [1]), 1.0)
    check_test(median_finder.find_median([0, 0], [0, 0]), 0.0)
    check_test(median_finder.find_median([-5, -3], [-2, -1]), -2.5)

    try:
        median_finder.find_median([], [])
    except ValueError:
        print("Passed: two empty lists raise ValueError")
    else:
        raise AssertionError("Expected ValueError for two empty lists")


if __name__ == "__main__":
    run_tests()


# Questions to ask yourself before choosing this algorithm:
# - Are both input lists already sorted?
# - Do I only need the middle value, rather than a fully merged list?
# - Can I partition the two lists so each side contains half of all values?
# - Should I binary-search the shorter list to keep the search efficient?
