"""
Arrays and Strings: Two Sum

Approach:
    We could check every pair, or we could use a hash map / dictionary.
    This solution uses the hash map approach.

    Use a dictionary when you need to quickly answer,
    "Have I seen the matching value already?"

Time:  O(n)
    We visit each number once. Dictionary lookups are usually O(1).

Space: O(n)
    In the worst case, the dictionary stores every number and its index.

Practical applications:
    - Shopping: find two item prices that exactly use a gift card or budget.
    - Banking: find two transactions whose amounts add to a known total.
    - Scheduling: find two task durations that fill an available time slot.
    - Games: find two score values or card values that reach a target score.
"""


class TwoSum:
    """A class groups the Two Sum behavior into one reusable object."""

    def find_indexes(self, nums: list[int], target: int) -> list[int] | None:
        """Return the two matching indexes, or None when no pair exists."""
        seen = {}  # number: its index

        for index, number in enumerate(nums):
            complement = target - number

            # Have we seen the number needed to complete this pair?
            if complement in seen:
                return [seen[complement], index]

            # Store this number so a later number can pair with it.
            seen[number] = index

        return None


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    two_sum = TwoSum()  # Create an object (an instance) of the class.

    def check_test(actual: list[int], expected: list[int] | None):
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}")

    check_test(two_sum.find_indexes([2, 7, 11, 15], 9),[0, 1])
    check_test(two_sum.find_indexes([3, 2, 4], 6),[1, 2])
    check_test(two_sum.find_indexes([3, 3], 6),[0, 1])  # duplicate values
    check_test(two_sum.find_indexes([-3, 4, 3, 90], 0), [0, 2])  # negatives
    check_test(two_sum.find_indexes([1, 2, 3], 10), None)  # no answer


if __name__ == "__main__":
    run_tests()
