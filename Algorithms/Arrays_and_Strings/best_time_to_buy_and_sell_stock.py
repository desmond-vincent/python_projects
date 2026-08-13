"""
Arrays and Strings: Best Time to Buy and Sell Stock
LeetCode link: <url>

Problem:
    Given daily stock prices, choose one day to buy and a later day to sell.
    Return the largest profit possible. If no profit is possible, return 0.

Approach:
    Keep track of the lowest price seen so far. For each new price, pretend we
    sell today and calculate the profit from buying at that lowest price.

    We must buy before we sell, so we only compare today's price with prices
    from earlier days.

Time:  O(n)
    We visit each price once.

Space: O(1)
    We only store two variables, regardless of how many prices are given.

Practical applications:
    - Shopping: find the biggest discount by buying at a low price and selling
      later at a higher price.
    - Reselling: estimate the best profit from buying an item and reselling it.
    - Data analysis: find the largest increase between an earlier and later value.
    - Travel: find the best time to book when prices later rise or fall.
"""


class BestTimeToBuyAndSellStock:
    """Find the best possible profit from one buy and one later sell."""

    def find_max_profit(self, prices: list[int]) -> int:
        """Return the highest profit from the list of daily prices."""
        lowest_price = float("inf")
        max_profit = 0

        for price in prices:
            # Buying at the lowest earlier price and selling today:
            profit_today = price - lowest_price
            max_profit = max(max_profit, profit_today)

            # Save this price if it is the best price to buy at so far.
            lowest_price = min(lowest_price, price)

        return max_profit


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    stock = BestTimeToBuyAndSellStock()

    def check_test(actual, expected):
        assert actual == expected
        print(f"Passed: expected {expected}, got {actual}!")

    check_test(stock.find_max_profit([7, 1, 5, 3, 6, 4]), 5)
    check_test(stock.find_max_profit([7, 6, 4, 3, 1]), 0)  # prices only decrease
    check_test(stock.find_max_profit([2, 4, 1]), 2)
    check_test(stock.find_max_profit([1, 2]), 1)
    check_test(stock.find_max_profit([]), 0)  # no days means no possible profit


if __name__ == "__main__":
    run_tests()
    print("All Best Time to Buy and Sell Stock tests passed!")
