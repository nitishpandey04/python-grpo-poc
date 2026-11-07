# Reviewed contracts and tested domains

Domains below bound the evidence; they do not restrict the original problem to these cases or prove correctness outside them.

| Task | Contract | Independent cases | Fresh passes / 8 | Selected | Tested domain |
|---|---|---:|---:|---|---|
| opencoder/10813843864 | pin | 61 | 5 | yes | ASCII strings, including invalid lengths, leading zero, non-digits and newline. |
| opencoder/36205923144 | digital_root | 105 | 6 | yes | Nonnegative base-10 integers up to 999999. |
| opencoder/25487510878 | moving_average | 214 | 2 | yes | Nonempty bounded integer lists; valid window 1 <= k <= len(list). |
| opencoder/40250997547 | permutation_1n | 50 | 3 | yes | Nonempty bounded integer arrays; compare with each value 1..n exactly once. |
| opencoder/2851535477 | roman | 88 | 0 | no | Canonical uppercase Roman numerals encoding integers 1..3999. |
| opencoder/1186130284 | attendance | 1093 | 0 | no | Exhaustive attendance records over A/L/P with lengths 0..6. |
| opencoder/26913407897 | swap_ends | 46 | 4 | yes | Integer lists of length 0..8, negative values and duplicates. |
| opencoder/22363512647 | consecutive_duplicates | 46 | 6 | yes | Integer lists of length 0..8; equality of adjacent items only. |
| opencoder/70848189818 | next_permutation | 50 | 0 | no | Integer lists of length 0..6; brute-force unique permutations and cyclic lexicographic successor. |
| opencoder/71982260207 | convert_ints | 38 | 4 | yes | Lists of ASCII signed decimal strings or explicitly invalid literals; no whitespace/underscore interpretation. |
| opencoder/9855304006 | matrix_transpose | 16 | 3 | yes | Existing reviewed narrow contract; fresh bounded edge/random cases using a new seed. |
| opencoder/34478518852 | count_vowels | 31 | 8 | no | Existing reviewed narrow contract; fresh bounded edge/random cases using a new seed. |
| opencoder/45176070003 | repeated_string | 256 | 0 | no | Nonempty strings: exhaustive a/b strings through length 7 and selected longer periods. |
| opencoder/3579853496 | sorted_absolute_values | 29 | 6 | yes | Existing reviewed narrow contract; fresh bounded edge/random cases using a new seed. |
| opencoder/2803810271 | unique_substrings | 135 | 3 | yes | Nonempty substrings of bounded strings, empty input has zero distinct nonempty substrings. |
| opencoder/11092861362 | smallest_char | 57 | 4 | yes | Nonempty bounded ASCII strings; Python character ordering. |
| opencoder/4997252410 | sliding_max | 214 | 4 | yes | Nonempty integer lists; valid windows 1..len(list), ties, negatives and both endpoints. |
| opencoder/737468861 | perfect_cube | 303 | 0 | no | Signed integers of magnitude at most 9262, integer cubes and immediate neighbors. |
| opencoder/71057583086 | first_max | 46 | 7 | yes | Integer lists including empty lists and tied maxima; first maximum index. |
| opencoder/2561817343 | harshad | 302 | 7 | yes | Positive base-10 integers; zero excluded because digit sum is zero. |
| opencoder/4138982841 | reverse_alpha | 65 | 5 | yes | ASCII strings; preserve letter case, discard digits/punctuation/space, then reverse. |
| opencoder/25875407409 | ipv4 | 90 | 0 | no | Dotted ASCII decimal octets; format/range errors, no leading-zero policy asserted. |
| opencoder/71942347628 | abc_string | 343 | 3 | yes | Bounded ASCII strings; alphabet restriction and at least one a. |
| opencoder/6948017274 | binary_five | 510 | 3 | yes | Exhaustive nonempty binary strings of lengths 1..8, including leading zeroes. |
| opencoder/18405278989 | bst_count | 10 | 3 | yes | Integers n=1..10; independent Catalan closed form rather than reference dynamic programming. |
| opencoder/33785290243 | longest_consecutive | 49 | 2 | yes | Integer arrays length at most 8; order independent, duplicates not counted twice. |
| opencoder/22578728363 | longest_unique_substring_length | 31 | 3 | yes | Existing reviewed narrow contract; fresh bounded edge/random cases using a new seed. |
| opencoder/30688868052 | replace_chars | 45 | 6 | yes | ASCII source strings; character keys to string values, empty/multichar replacements, simultaneous single pass. |
| opencoder/29398324817 | pangram | 31 | 2 | yes | English ASCII alphabet, case insensitive; punctuation and digits do not substitute for letters. |
