"""
Koko Eating Bananas

Approach:
    The faster Koko eats, the fewer hours she needs. That monotonic behavior
    lets us binary-search the smallest speed that finishes every pile in time.

Time:  O(n log m), where n is the number of piles and m is the largest pile.
Space: O(1)

Practical applications:
    - Finding the minimum server throughput needed to meet a deadline.
    - Choosing the slowest production rate that still completes a workload.
"""


class KokoEatingBananas:
    """Find the minimum banana-eating speed that meets a deadline."""

    def minimum_speed(self, piles: list[int], hours: int) -> int:
        """Return the minimum bananas per hour needed to finish all piles."""
        if not piles:
            raise ValueError("piles must not be empty")
        if hours < len(piles):
            raise ValueError("hours must be at least the number of piles")
        if any(pile <= 0 for pile in piles):
            raise ValueError("every pile must contain at least one banana")

        left = 1
        right = max(piles)

        while left < right:
            speed = (left + right) // 2
            hours_needed = sum((pile + speed - 1) // speed for pile in piles)

            if hours_needed <= hours:
                right = speed
            else:
                left = speed + 1

        return left


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    koko = KokoEatingBananas()

    def check_test(actual: int, expected: int) -> None:
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}")

    check_test(koko.minimum_speed([3, 6, 7, 11], 8), 4)
    check_test(koko.minimum_speed([30, 11, 23, 4, 20], 5), 30)
    check_test(koko.minimum_speed([30, 11, 23, 4, 20], 6), 23)
    check_test(koko.minimum_speed([1, 1, 1, 1], 4), 1)
    check_test(koko.minimum_speed([312884470], 968709470), 1)

    try:
        koko.minimum_speed([3, 6], 1)
    except ValueError:
        print("Passed: too few hours raises ValueError")
    else:
        raise AssertionError("Expected ValueError when there are too few hours")


if __name__ == "__main__":
    run_tests()


# Questions to ask yourself before choosing this algorithm:
# - Am I looking for the smallest or largest value that satisfies a condition?
# - Does a larger speed always make the task take the same or fewer hours?
# - Can I calculate whether one candidate speed meets the deadline?
# - What are the valid lower and upper bounds for the answer?
