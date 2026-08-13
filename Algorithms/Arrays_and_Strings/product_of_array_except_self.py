"""
Arrays and Strings: Product of Array Except Self
LeetCode link: <url>

Problem:
    For every number in nums, return the product of every other number.
    Do not use division.

Example:
    nums = [1, 2, 3, 4]
    answer = [24, 12, 8, 6]

    At index 0, multiply 2 * 3 * 4 = 24.
    At index 1, multiply 1 * 3 * 4 = 12.

Approach:
    First, store the product of all numbers to the LEFT of each index.
    Then move from right to left, multiplying each answer by the product of all
    numbers to the RIGHT of that index.

Time:  O(n)
    We make two passes through nums.

Space: O(1) extra space
    We use the output list for the answers and only one extra suffix variable.
    (The output list itself takes O(n) space.)

Practical applications:
    - Analytics: calculate the combined effect of all factors except one.
    - Reliability: calculate a system's probability using every component except
      the component currently being inspected.
    - Games: calculate a score multiplier while leaving out one item or bonus.
    - Finance: compare a portfolio's combined growth with one investment excluded.
"""


class ProductOfArrayExceptSelf:
    """Build a product list without using division."""

    def find_products(self, nums: list[int]) -> list[int]:
        """Return the product of all other values for every index."""
        answer = [1] * len(nums)

        # First pass: answer[index] becomes the product to its left.
        left_product = 1
        for index in range(len(nums)):
            answer[index] = left_product
            left_product *= nums[index]

        # Second pass: multiply by the product to the right.
        right_product = 1
        for index in range(len(nums) - 1, -1, -1):
            answer[index] *= right_product
            right_product *= nums[index]

        return answer


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    product = ProductOfArrayExceptSelf()

    # Each tuple contains: (input nums, expected answer).
    tests = [
        ([1, 2, 3, 4], [24, 12, 8, 6]),
        ([-1, 1, 0, -3, 3], [0, 0, 9, 0, 0]),
        ([2, 3], [3, 2]),
        ([0, 0], [0, 0]),
        ([], []),
    ]

    passes = 1
    # Run each test. A failing assert stops the loop at that test.
    for nums, expected in tests:
        
        actual = product.find_products(nums)
        assert actual == expected
        print(f"TEST {passes} Passed: nums={nums}, expected={expected}, got={actual}")
        passes += 1


if __name__ == "__main__":
    run_tests()
    print("All Product of Array Except Self tests passed!")
