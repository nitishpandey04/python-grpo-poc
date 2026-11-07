# References rejected by stronger tests

All five references passed the previous supplied tests. They are excluded, with original instructions preserved; no instruction constraint was silently removed.

## opencoder/31646688236 — html_tag

Write a function that checks if a given string is a valid HTML tag. A tag is considered valid if it starts with '<', ends with '>', and the content between the brackets does not contain any '<' characters. The function should return True if the tag is valid, and False otherwise.

Failure: ['IndexError: string index out of range']

```python
def isValidHTMLTag(s):
    if s[0] != "<" or s[-1] != ">" or "<" in s[1:-1]:
        return False
    return True
```

## opencoder/73819648209 — hex_color

Create a function that checks if a given string is a valid hexadecimal color code. A valid hexadecimal color code starts with a '#' followed by exactly six hexadecimal characters (0-9, a-f, A-F).

Failure: ['AssertionError: independent_0008']

```python
import re

def is_valid_hex_color(color):
    if re.match(r'^#[0-9A-Fa-f]{6}$', color):
        return True
    else:
        return False
```

## opencoder/69878197227 — search_insert

Write a function to find the index at which a target value should be inserted into a sorted list to maintain the sorted order.

Failure: ['IndexError: list index out of range']

```python
def search_insert(nums, target):
  i = 0
  if target < nums[0]:
    return 0
  while i < len(nums):
    if nums[i] == target:
      return i
    elif i < len(nums) - 1 and nums[i] < target < nums[i + 1]:
      return i + 1
    i += 1
  return i
```

## opencoder/74866389783 — crossword

Write a function that takes a list of strings as input, where each string represents a row in a crossword puzzle. The function should return the length of the longest consecutive sequence of non-empty cells (i.e., cells that are not empty spaces) in any row of the crossword.

Failure: ['AssertionError: independent_0002']

```python
def longest_row_length(crossword):
    max_length = 0
    for row in crossword:
        row_length = len(row)
        if row_length > max_length:
            max_length = row_length
    return max_length
```

## opencoder/74960684985 — count_digit

Write a function to count the number of times a digit appears in a number.

Failure: ['AssertionError: independent_0000']

```python
def count_digit(n, d):
    count = 0
    while n > 0:
        if n % 10 == d:
            count += 1
        n //= 10
    return count
```
