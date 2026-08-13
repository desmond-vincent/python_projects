"""
Arrays and Strings: Longest Common Prefix
LeetCode link: <url>

Problem:
    Given a list of strings, return the longest beginning part (prefix) shared
    by every string. Return an empty string if they have no common prefix.

Example:
    strs = ["flower", "flow", "flight"]
    answer = "fl"

Approach:
    Begin by treating the first word as the possible common prefix.
    Compare it with each later word. While that word does not start with the
    current prefix, remove the last character from the prefix.

    When the prefix becomes an empty string, no common prefix exists.

Time:  O(n * m)
    n is the number of strings and m is the length of the shortest string.
    In the worst case, we compare up to m characters for each string.

Space: O(1) extra space
    Apart from the returned prefix, we only use a few variables.

Practical applications:
    - Search suggestions: find a shared starting phrase among search terms.
    - File paths: identify the shared beginning of related paths or URLs.
    - Autocomplete: determine text multiple user entries begin with.
    - Data cleaning: group labels or codes that share a starting pattern.
"""


class LongestCommonPrefix:
    """Find the longest prefix shared by every string in a list."""

    def find_prefix(self, strs: list[str]) -> str:
        """Return the longest common prefix, or an empty string."""
        if not strs:
            return ""

        prefix = strs[0]

        for word in strs[1:]:
            # Remove letters from the end until word starts with prefix.
            while not word.startswith(prefix):
                prefix = prefix[:-1]

                if prefix == "":
                    return ""

        return prefix


def run_tests() -> None:
    """Run small checks. An AssertionError means a test failed."""
    prefix_finder = LongestCommonPrefix()

    # Each tuple contains: (input strings, expected prefix).
    tests = [
        (["flower", "flow", "flight"], "fl"),
        (["dog", "racecar", "car"], ""),
        (["interspecies", "interstellar", "interstate"], "inters"),
        (["same", "same", "same"], "same"),
        ([], ""),
    ]

    passes = 1

    for strs, expected in tests:
        actual = prefix_finder.find_prefix(strs)
        assert actual == expected
        print(f"TEST {passes} \tPassed: strs={strs}, expected={expected!r}, got={actual!r}")
        passes += 1


if __name__ == "__main__":
    run_tests()
    print("\nAll Longest Common Prefix tests passed!\n")
