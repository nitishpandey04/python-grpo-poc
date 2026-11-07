"""Bounded, independently specified contracts; never execute source references here."""
import calendar
import itertools
import math
import random
import re
import string
from educational_oracles import independent_cases
from opencoder_data import literal_examples

# Deliberately conservative related-family exclusions, applied to protected text only.
PROTECTED_PATTERNS={
 'powers':r'power of (two|three|2|3|a given|the second|base)|power.*base|first number.*power',
 'perfect_square':r'perfect square','digital_root':r'digital root|until.*single.digit',
 'word_occurrences':r'count.*word|word.*occurrence','moving_average':r'moving average',
 'permutation_1n':r'valid permutation|every integer from 1 to','roman':r'roman',
 'attendance':r'attendance','swap_ends':r'swap.*first.*last',
 'top_k_frequent':r'most frequent.*element|top.?k.*frequen',
 'consecutive_dedup':r'remove.*consecutive.*duplicate|consecutive duplicate.*removed',
 'next_permutation':r'next permutation','convert_ints':r'list of strings.*integer|converts.*strings.*integer',
 'calendar':r'leap year|days.*month|month.*days','repeated_string':r'constructed by repeating|repeated substring',
 'consecutive_duplicates':r'contains.*consecutive duplicate','lucas':r'lucas',
 'html_tag':r'html tag','hex_color':r'hexadecimal color','unique_substrings':r'number of unique substrings',
 'count_digits':r'count.*digits|number of digits','smallest_char':r'smallest character',
 'sliding_max':r'maximum.*sliding window|maximum values.*window','perfect_cube':r'perfect cube',
 'first_max':r'index.*first occurrence.*maximum|first.*maximum.*index',
 'perfect_number':r'perfect number|proper divisors','harshad':r'harshad',
 'reverse_alpha':r'non.alphabetic.*reverse|reverse.*non.alphabetic',
 'third_largest':r'third largest|third maximum','ipv4':r'ipv4',
 'search_insert':r'index.*insert.*sorted|insert.*sorted order',
 'abc_string':r"only the characters 'a', 'b', and 'c'",'binary_five':r'binary.*divisible by 5',
 'bst_count':r'unique binary search tree','longest_consecutive':r'longest consecutive sequence',
 'crossword':r'crossword','longest_prefix':r'longest common prefix',
 'replace_chars':r'each character.*replaced|character.*corresponding value',
 'count_digit':r'number of times a digit','pangram':r'pangram',
 'adjacent_duplicates':r'consecutive duplicate','order_statistic':r'(second|third|kth).*(largest|smallest|highest|maximum)',
 'pin':r'valid pin|atm.*pin','matrix_transpose':r'transpose.*matrix|matrix.*transpose',
 'longest_unique_substring_length':r'longest.*substring.*without repeating|longest.*non.repeating.*substring',
 'count_vowels':r'count.*vowel|number of vowels','sorted_absolute_values':r'absolute.*sort|sort.*absolute',
 'strictly_increasing':r'strictly increasing'}
ALIASES={'consecutive_dedup':'adjacent_duplicates','consecutive_duplicates':'adjacent_duplicates','third_largest':'order_statistic','power_two':'powers','power_three':'powers','power_base':'powers','leap_year':'calendar','month_days':'calendar'}


def family(kind):return ALIASES.get(kind,kind)


def protected_hits(kind,protected):
    pattern=PROTECTED_PATTERNS[family(kind)]
    return [row['task_id'] for row in protected if re.search(pattern,row['instruction'].lower())]


def decimal_root(n):
    return 0 if n==0 else 1+(n-1)%9


def roman_encode(n):
    result=''
    for value,token in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        count,n=divmod(n,value);result+=token*count
    return result


def next_perm(a):
    possibilities=sorted(set(itertools.permutations(a)))
    index=possibilities.index(tuple(a))
    return list(possibilities[(index+1)%len(possibilities)])


def third_unique(a):
    return next((v for v in a if len({x for x in a if x>v})==2),None)


def longest_run(a):
    return max((length for start in a for length in range(1,len(a)+1) if all(start+j in a for j in range(length))),default=0)


def crossword(rows):
    return max((len(part) for row in rows for part in row.split(' ')),default=0)


def longest_prefix(values):
    if not values:return ''
    result=''
    for chars in zip(*values):
        if any(c!=chars[0] for c in chars):break
        result+=chars[0]
    return result


# Each is a valid, nonconstant faulty implementation. All must be rejected.
# Bodies execute exclusively in Docker, just like canonical references.
CONTROLS={
 'pin':[('ignores_leading_zero','s',"s.isdigit() and 4<=len(s)<=6"),('rejects_five_digits','s',"s.isdigit() and len(s) in (4,6) and s[0]!='0'")],
 'digital_root':[('only_one_sum','n',"sum(map(int,str(n)))"),('zero_for_multiples_of_nine','n','n%9')],
 'moving_average':[('integer_division','a,k','[sum(a[i:i+k])//k for i in range(len(a)-k+1)]'),('omits_last_window','a,k','[sum(a[i:i+k])/k for i in range(len(a)-k)]')],
 'permutation_1n':[('ignores_duplicates','a','min(a)==1 and max(a)==len(a)'),('checks_uniqueness_only','a','len(set(a))==len(a)')],
 'roman':[('additive_only','s',"sum(dict(I=1,V=5,X=10,L=50,C=100,D=500,M=1000)[c] for c in s)"),('single_character_only','s',"dict(I=1,V=5,X=10,L=50,C=100,D=500,M=1000)[s[0]]")],
 'attendance':[('counts_total_lates','s',"s.count('A')<=1 and s.count('L')<=2"),('allows_two_absences','s',"s.count('A')<=2 and 'LLL' not in s")],
 'swap_ends':[('reverses_everything','a','a[::-1]'),('rotates_left','a','a[1:]+a[:1]')],
 'consecutive_dedup':[('removes_all_duplicates','a','list(dict.fromkeys(a))'),('drops_last_item','a','[x for i,x in enumerate(a[:-1]) if i==0 or a[i-1]!=x]')],
 'next_permutation':[('always_sorts','a','sorted(a)'),('reverses_list','a','a[::-1]')],
 'convert_ints':[('truncates_decimals','a',"[int(float(s)) if s not in ('','abc','+','-','12x') else None for s in a]"),('fails_on_bad_entry','a','[int(s) for s in a]')],
 'repeated_string':[('any_duplicate_character','s','len(set(s))<len(s)'),('only_even_half_repeats','s','len(s)%2==0 and s[:len(s)//2]==s[len(s)//2:]')],
 'consecutive_duplicates':[('any_duplicate_anywhere','a','len(a)!=len(set(a))'),('misses_first_pair','a','any(a[i]==a[i-1] for i in range(2,len(a)))')],
 'html_tag':[('checks_endpoints_only','s',"s.startswith('<') and s.endswith('>')"),('rejects_interior_closing_bracket','s',"s.startswith('<') and s.endswith('>') and '<' not in s[1:-1] and '>' not in s[1:-1]")],
 'hex_color':[('accepts_three_digits','s',"len(s) in (4,7) and s[0]=='#' and all(c in '0123456789abcdefABCDEF' for c in s[1:])"),('lowercase_only','s',"len(s)==7 and s[0]=='#' and all(c in '0123456789abcdef' for c in s[1:])")],
 'unique_substrings':[('counts_all_positions','s','len(s)*(len(s)+1)//2'),('counts_characters_only','s','len(set(s))')],
 'smallest_char':[('counts_largest','s','s.count(max(s))'),('counts_first','s','s.count(s[0])')],
 'sliding_max':[('minimum_instead_of_maximum','a,k','[min(a[i:i+k]) for i in range(len(a)-k+1)]'),('omits_last_window','a,k','[max(a[i:i+k]) for i in range(len(a)-k)]')],
 'perfect_cube':[('positive_cubes_only','n','n>=0 and any(i**3==n for i in range(23))'),('checks_squares','n','any(i*i==abs(n) for i in range(101))')],
 'first_max':[('returns_last_tie','a','len(a)-1-a[::-1].index(max(a)) if a else -1'),('returns_value_not_index','a','max(a) if a else -1')],
 'harshad':[('divisible_by_nine_only','n','n%9==0'),('divisible_by_last_digit','n','n%10!=0 and n%(n%10)==0')],
 'reverse_alpha':[('preserves_digits','s',"''.join(c for c in s[::-1] if c.isalnum())"),('filters_without_reversing','s',"''.join(c for c in s if c.isalpha())")],
 'third_largest':[('does_not_deduplicate','a','sorted(a)[-3] if len(a)>=3 else None'),('second_instead_of_third','a','sorted(set(a))[-2] if len(set(a))>=2 else None')],
 'ipv4':[('ignores_range','s',"len(s.split('.'))==4 and all(p.isdigit() for p in s.split('.'))"),('accepts_missing_octet','s',"len(s.split('.')) in (3,4) and all(p.isdigit() and int(p)<=255 for p in s.split('.'))")],
 'search_insert':[('exact_matches_only','a,t','a.index(t) if t in a else len(a)'),('off_by_one','a,t','sum(x<t for x in a)+1')],
 'abc_string':[('ignores_required_a','s',"all(c in 'abc' for c in s)"),('ignores_invalid_characters','s',"'a' in s")],
 'binary_five':[('interprets_decimal','s','int(s)%5==0'),('checks_parity','s','int(s,2)%2==0')],
 'bst_count':[('factorial','n',"__import__('math').factorial(n)"),('missing_catalan_denominator','n',"__import__('math').comb(2*n,n)")],
 'longest_consecutive':[('counts_distinct_values','a','len(set(a))'),('counts_duplicates_in_sorted_run','a','len(a) if a and max(a)-min(a)+1==len(set(a)) else 0')],
 'crossword':[('counts_row_length','rows','max(map(len,rows),default=0)'),('counts_total_filled_cells','rows',"max((len(s.replace(' ','')) for s in rows),default=0)")],
 'longest_prefix':[('returns_shortest_word','a',"min(a,key=len) if a else ''"),('checks_first_two_only','a',"''.join(x for x,y in zip(a[0],a[1]) if x==y) if len(a)>1 else (a[0] if a else '')")],
 'replace_chars':[('drops_unmapped_characters','s,m',"''.join(m.get(c,'') for c in s)"),('ignores_replacements','s,m','s')],
 'count_digit':[('zero_has_no_digits','n,d','str(n).count(str(d)) if n else 0'),('counts_presence_only','n,d','int(str(d) in str(n))')],
 'pangram':[('counts_nonletters','s','len(set(s.lower()))>=26'),('case_sensitive','s',"all(c in s for c in 'abcdefghijklmnopqrstuvwxyz')")],
 'matrix_transpose':[('returns_original','a','a'),('reverses_row_order','a','[list(t) for t in zip(*a[::-1])]')],
 'longest_unique_substring_length':[('counts_global_distinct','s','len(set(s))'),('off_by_one_nonempty','s','max(0,len(s)-1)')],
 'count_vowels':[('case_sensitive','s',"sum(c in 'aeiou' for c in s)"),('counts_unique_vowels','s',"len(set(s.lower())&set('aeiou'))")],
 'sorted_absolute_values':[('sorts_signed_values','a','sorted(a)'),('preserves_absolute_order','a','list(map(abs,a))')],
 'strictly_increasing':[('allows_equal_values','a','all(a[i]<=a[i+1] for i in range(len(a)-1))'),('checks_endpoints_only','a','len(a)<2 or a[0]<a[-1]')],
}


def cases_for(kind,row,seed=20261108):
    """Return independent (literal argument tuple, expected value) pairs and domain."""
    rng=random.Random(seed)
    one=lambda values:[(v,) for v in values]
    words=['','a','aa','ab','aba','abab','abc','abcabc','ABC','A a','bca','abac','abcabcabc','aabaaab']
    arrays=[[],[0],[1],[1,1],[2,1],[1,2,1],[4,1,3,2],[3,3,2,1],[-3,-1,0,-1],[1,2,2,4]]
    arrays += [[rng.randint(-4,5) for _ in range(rng.randint(0,8))] for _ in range(40)]
    domain='';fn=None;inputs=[];pairs=[]
    if kind in {'matrix_transpose','longest_unique_substring_length','count_vowels','sorted_absolute_values','strictly_increasing'}:
        _,pairs=independent_cases(row['instruction'],literal_examples(row['tests'][:row['provided_assertions']],row['entry_point']),seed)
        domain='Existing reviewed narrow contract; fresh bounded edge/random cases using a new seed.'
    elif kind=='pin':
        inputs=one(['','0','123','1234','12345','123456','1234567','0123','123a','12 4','+123','1234\n']+[''.join(rng.choice('0123456789ab') for _ in range(rng.randint(0,8))) for _ in range(60)])
        fn=lambda s:4<=len(s)<=6 and s[0]!='0' and all(c in string.digits for c in s);domain='ASCII strings, including invalid lengths, leading zero, non-digits and newline.'
    elif kind=='digital_root':
        inputs=one(list(range(101))+[999,1000,98765,999999]);fn=decimal_root;domain='Nonnegative base-10 integers up to 999999.'
    elif kind=='moving_average':
        inputs=[(a,k) for a in arrays if a for k in range(1,len(a)+1)];fn=lambda a,k:[sum(a[i+j] for j in range(k))/k for i in range(len(a)-k+1)];domain='Nonempty bounded integer lists; valid window 1 <= k <= len(list).'
    elif kind=='permutation_1n':
        inputs=one([a for a in arrays if a]+[[1,2,3],[3,1,2],[1,1,4,4],[0,1],[2,3]])
        fn=lambda a:all(a.count(x)==1 for x in range(1,len(a)+1));domain='Nonempty bounded integer arrays; compare with each value 1..n exactly once.'
    elif kind=='roman':
        pairs=[((roman_encode(n),),n) for n in sorted(set([1,4,9,40,90,400,900,3999]+[rng.randint(1,3999) for _ in range(80)]))];domain='Canonical uppercase Roman numerals encoding integers 1..3999.'
    elif kind=='attendance':
        inputs=one([''.join(p) for n in range(7) for p in itertools.product('ALP',repeat=n)])
        fn=lambda s:s.count('A')<=1 and all(s[i:i+3]!='LLL' for i in range(max(0,len(s)-2)));domain='Exhaustive attendance records over A/L/P with lengths 0..6.'
    elif kind=='swap_ends':
        inputs=one(arrays);fn=lambda a:([a[-1]]+a[1:-1]+[a[0]]) if len(a)>1 else a[:];domain='Integer lists of length 0..8, negative values and duplicates.'
    elif kind=='consecutive_dedup':
        inputs=one(arrays);fn=lambda a:[k for k,_ in itertools.groupby(a)];domain='Integer lists of length 0..8, including separated and adjacent duplicates.'
    elif kind=='next_permutation':
        inputs=one([a[:6] for a in arrays]+[[1,2,3],[3,2,1],[1,1,5],[1,3,2]])
        fn=next_perm;domain='Integer lists of length 0..6; brute-force unique permutations and cyclic lexicographic successor.'
    elif kind=='convert_ints':
        values=['0','1','-1','+2','0002','999','abc','1.2','','+','-','12x']
        inputs=one([[],values]+[[rng.choice(values) for _ in range(rng.randint(1,8))] for _ in range(40)])
        fn=lambda a:[int(s) if re.fullmatch(r'[+-]?[0-9]+',s) else None for s in a];domain='Lists of ASCII signed decimal strings or explicitly invalid literals; no whitespace/underscore interpretation.'
    elif kind=='repeated_string':
        inputs=one([''.join(p) for n in range(1,8) for p in itertools.product('ab',repeat=n)]+['abcabcabc','abcd'])
        fn=lambda s:any(s==s[:k]*(len(s)//k) for k in range(1,len(s)) if len(s)%k==0);domain='Nonempty strings: exhaustive a/b strings through length 7 and selected longer periods.'
    elif kind=='consecutive_duplicates':
        inputs=one(arrays);fn=lambda a:any(len(list(g))>1 for _,g in itertools.groupby(a));domain='Integer lists of length 0..8; equality of adjacent items only.'
    elif kind=='html_tag':
        inputs=one(['','<','>','<>','<a>','<a>>','<<a>','<a<b>','x<a>','<a>x','<a\n>']+[''.join(p) for n in range(5) for p in itertools.product('<>a',repeat=n)])
        fn=lambda s:len(s)>=2 and s[:1]=='<' and s[-1:]=='>' and '<' not in s[1:-1];domain='Exact simplified tag rules in instruction, not HTML standard validation; empty and malformed strings included.'
    elif kind=='hex_color':
        inputs=one(['','#','#123','#abcdef','#ABCDEF','#012345','#aBcD09','#000000','#123456\n','#12345g','123456','#1234567']+['#'+''.join(rng.choice('0123456789abcdefABCDEFg') for _ in range(rng.randint(3,8))) for _ in range(80)])
        fn=lambda s:len(s)==7 and s[0]=='#' and all(c in string.hexdigits for c in s[1:]);domain='ASCII strings; exact six hex characters after #, including trailing-newline rejection.'
    elif kind=='unique_substrings':
        inputs=one(words+[''.join(p) for n in range(7) for p in itertools.product('ab',repeat=n)])
        fn=lambda s:len(set(s[i:j] for i in range(len(s)) for j in range(i+1,len(s)+1)));domain='Nonempty substrings of bounded strings, empty input has zero distinct nonempty substrings.'
    elif kind=='smallest_char':
        inputs=one([s for s in words if s]+['bbaaa','zaaz','bBaa','  a']+[''.join(rng.choice('abcABC 09') for _ in range(rng.randint(1,12))) for _ in range(40)])
        fn=lambda s:sum(c==sorted(set(s))[0] for c in s);domain='Nonempty bounded ASCII strings; Python character ordering.'
    elif kind=='sliding_max':
        inputs=[(a,k) for a in arrays if a for k in range(1,len(a)+1)];fn=lambda a,k:[sorted(a[i:i+k])[-1] for i in range(len(a)-k+1)];domain='Nonempty integer lists; valid windows 1..len(list), ties, negatives and both endpoints.'
    elif kind=='perfect_cube':
        inputs=one(list(range(-100,101))+[n**3+d for n in range(-21,22) for d in [-1,0,1]])
        fn=lambda n:any(i**3==n for i in range(-22,23));domain='Signed integers of magnitude at most 9262, integer cubes and immediate neighbors.'
    elif kind=='first_max':
        inputs=one(arrays);fn=lambda a:next((i for i,v in enumerate(a) if all(v>=x for x in a)),-1);domain='Integer lists including empty lists and tied maxima; first maximum index.'
    elif kind=='harshad':
        inputs=one(list(range(1,301))+[1729,9999]);fn=lambda n:n%sum(ord(c)-ord('0') for c in str(n))==0;domain='Positive base-10 integers; zero excluded because digit sum is zero.'
    elif kind=='reverse_alpha':
        inputs=one(words+['a1B!','Hello, World!','123','x y z','AbC09!']+[''.join(rng.choice(string.ascii_letters+'09 !?') for _ in range(rng.randint(0,15))) for _ in range(50)])
        fn=lambda s:''.join(s[i] for i in range(len(s)-1,-1,-1) if s[i] in string.ascii_letters);domain='ASCII strings; preserve letter case, discard digits/punctuation/space, then reverse.'
    elif kind=='third_largest':
        inputs=one(arrays);fn=third_unique;domain='Bounded integer lists including empty, fewer than three distinct values and repeated maxima.'
    elif kind=='ipv4':
        # Leading-zero policy is unresolved by the short source prompt: do not invent it.
        inputs=one(['','1.2.3','1.2.3.4.5','1..2.3','1.2.3.a','1.2.3.-1','1.2.3.256','0.0.0.0','255.255.255.255','1.2.3.4\n']+['.'.join(str(rng.randint(0,300)) for _ in range(4)) for _ in range(80)])
        fn=lambda s:len(s.split('.'))==4 and all(p and all(c in string.digits for c in p) and 0<=int(p)<=255 for p in s.split('.'));domain='Dotted ASCII decimal octets; format/range errors, no leading-zero policy asserted.'
    elif kind=='search_insert':
        inputs=[(sorted(set(a)),t) for a in arrays for t in [-6,0,3,8]];fn=lambda a,t:sum(x<t for x in a);domain='Strictly increasing integer lists including empty; target any bounded integer, so insertion position is unique.'
    elif kind=='abc_string':
        inputs=one([''.join(p) for n in range(5) for p in itertools.product('abcx',repeat=n)]+['abc\n','Aabc'])
        fn=lambda s:'a' in s and set(s)<=set('abc');domain='Bounded ASCII strings; alphabet restriction and at least one a.'
    elif kind=='binary_five':
        inputs=one([''.join(p) for n in range(1,9) for p in itertools.product('01',repeat=n)])
        fn=lambda s:int(s,2)%5==0;domain='Exhaustive nonempty binary strings of lengths 1..8, including leading zeroes.'
    elif kind=='bst_count':
        inputs=one(range(1,11));fn=lambda n:math.comb(2*n,n)//(n+1);domain='Integers n=1..10; independent Catalan closed form rather than reference dynamic programming.'
    elif kind=='longest_consecutive':
        inputs=one(arrays+[[100,4,200,1,3,2],[1,2,0,1],[-2,-1,0,2,2]])
        fn=longest_run;domain='Integer arrays length at most 8; order independent, duplicates not counted twice.'
    elif kind=='crossword':
        inputs=one([[],[''],['A B'],['ABC DE'],[' A  B '],['AB','C DE'],['   ']]+[[''.join(rng.choice('AB ') for _ in range(rng.randint(0,12))) for _ in range(rng.randint(0,5))] for _ in range(40)])
        fn=crossword;domain='Rows of ASCII letters and literal spaces; count longest run of cells without spaces.'
    elif kind=='longest_prefix':
        inputs=one([[],[''],['abc'],['flower','flow','flight'],['ab','ab','a'],['aa','ab'],['ab','ac','ax'],['ab','ab','xy']]);fn=longest_prefix;domain='Lists of bounded ASCII strings, including empty lists and empty strings.'
    elif kind=='replace_chars':
        inputs=[('ab',{'a':'b','b':'c'}),('axb',{'a':'','b':'YY'}),('',{}),('Aa!',{'A':'a'}),('aba',{})]+[(''.join(rng.choice('abx ') for _ in range(rng.randint(0,12))),{k:rng.choice(['','X','ab']) for k in rng.sample(list('abx'),rng.randint(0,3))}) for _ in range(40)]
        fn=lambda s,m:''.join(m[c] if c in m else c for c in s);domain='ASCII source strings; character keys to string values, empty/multichar replacements, simultaneous single pass.'
    elif kind=='count_digit':
        inputs=[(n,d) for n in [0,1,9,10,11,101,1000,999,12345] for d in range(10)];fn=lambda n,d:sum(c==str(d) for c in str(n));domain='Nonnegative decimal integers, digit d=0..9; number zero contains one zero digit.'
    elif kind=='pangram':
        inputs=one(['',string.ascii_lowercase,string.ascii_uppercase,string.ascii_lowercase[:-1],string.digits+string.punctuation,'The quick brown fox jumps over the lazy dog']+[string.ascii_lowercase[:i]+string.ascii_lowercase[i+1:] for i in range(26)])
        fn=lambda s:set(string.ascii_lowercase)<=set(s.lower());domain='English ASCII alphabet, case insensitive; punctuation and digits do not substitute for letters.'
    else:raise ValueError('No reviewed generator: '+kind)
    if fn:pairs=[(args,fn(*args)) for args in inputs]
    unique={repr(args):(args,expected) for args,expected in pairs}
    if not unique:raise ValueError('Independent contract generated no cases')
    return list(unique.values()),domain


def faulty_controls(kind,entry):
    return [{'name':name,'code':f'def {entry}({args}):\n return {expression}'} for name,args,expression in CONTROLS[kind]]
