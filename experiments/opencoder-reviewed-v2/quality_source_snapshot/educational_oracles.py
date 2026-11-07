"""Reviewed narrow task contracts; reference code is never executed on the host."""
import collections
import math
import random
import re


def longest_unique_length(s):
    return max((j-i for i in range(len(s)+1) for j in range(i,len(s)+1) if len(set(s[i:j]))==j-i),default=0)


def min_parentheses_removal(s):
    opens=removed=0
    for ch in s:
        if ch=='(':opens+=1
        elif ch==')':
            if opens:opens-=1
            else:removed+=1
    return opens+removed


def is_prime(n):
    return n>=2 and all(n%d for d in range(2,math.isqrt(n)+1))


def matching_oracle(instruction,examples):
    """Return an explicit contract only when its shape and supplied examples agree."""
    text=instruction.lower();candidates=[]
    if not examples or any(kwargs for _,kwargs,_ in examples):return None
    args0=examples[0][0]
    arity=len(args0)
    strings=arity==1 and all(len(a)==1 and isinstance(a[0],str) for a,_,_ in examples)
    ints=all(len(a)==arity and all(type(x) is int for x in a) for a,_,_ in examples)
    lists=arity==1 and all(len(a)==1 and isinstance(a[0],list) and all(type(x) is int for x in a[0]) for a,_,_ in examples)
    if strings:
        if 'palindrome' in text and not any(w in text for w in ['substring','permutation','rearrang','longest','number of','count','remove','delete','partition']):
            if 'alphanumeric' in text or ('punctuation' in text and 'ignor' in text):
                fn=lambda s:(lambda t:t==t[::-1])(''.join(c.lower() for c in s if c.isalnum()))
                candidates.append(('palindrome_alphanumeric_casefold',fn,'string'))
            elif any(w in text for w in ['case-insensitive','ignoring case','ignore case','ignoring capitalization']):
                candidates.append(('palindrome_ignore_case',lambda s:s.lower()==s.lower()[::-1],'string'))
            elif not any(w in text for w in ['ignore','ignoring','spaces','punctuation','capitalization']):
                candidates.append(('palindrome_exact',lambda s:s==s[::-1],'string'))
        if 'reverse' in text and 'string' in text and not any(w in text for w in ['word','vowel','each','parenthes','substring','integer','number','recursive']):
            candidates.append(('reverse_string',lambda s:''.join(reversed(s)),'string'))
        if 'vowel' in text and any(w in text for w in ['count','number of']) and not any(w in text for w in ['each vowel','consonant','substring','consecutive']):
            candidates.append(('count_vowels',lambda s:sum(c.lower() in 'aeiou' for c in s),'string'))
        if 'unique' in text and 'character' in text and 'list' in text and any(w in text for w in ['order they appear','order of appearance','original order','preserving the order']):
            candidates.append(('unique_characters_in_order',lambda s:list(dict.fromkeys(s)),'string'))
        if any(w in text for w in ['frequency','frequencies','occurrence']) and 'character' in text and any(w in text for w in ['dictionary','dict']) and not any(w in text for w in ['most frequent','least frequent','first','last','alphabetical','sort']):
            candidates.append(('character_counts',lambda s:dict(collections.Counter(s)),'string'))
        if 'longest' in text and 'substring' in text and 'length' in text and any(w in text for w in ['without repeating','without duplicate','non-repeating','distinct characters']):
            candidates.append(('longest_unique_substring_length',longest_unique_length,'string'))
        if 'parenthes' in text and 'minimum' in text and any(w in text for w in ['remove','removed','removal']) and not any(w in text for w in ['return all','list of','valid strings']):
            candidates.append(('minimum_parentheses_removal',min_parentheses_removal,'parentheses'))
    if lists:
        if 'absolute' in text and ('sort' in text) and not any(w in text for w in ['descending','decreasing','difference','closest','maximum','minimum']):
            candidates.append(('sorted_absolute_values',lambda a:sorted(abs(x) for x in a),'integer_list'))
        if 'sum' in text and not any(w in text for w in ['index','indices','position','subarray','maximum','minimum','consecutive','square','cube']):
            if 'even' in text and 'odd' not in text:candidates.append(('sum_even_values',lambda a:sum(x for x in a if x%2==0),'integer_list'))
            elif 'odd' in text and 'even' not in text:candidates.append(('sum_odd_values',lambda a:sum(x for x in a if x%2),'integer_list'))
            elif any(w in text for w in ['sum of all elements','sum of the elements','sum of all the elements']) and not any(w in text for w in ['unique','distinct','positive','negative','absolute']):
                candidates.append(('sum_list',lambda a:sum(a),'integer_list'))
        if any(w in text for w in ['running sum','cumulative sum','prefix sum']):
            candidates.append(('prefix_sums',lambda a:[sum(a[:i+1]) for i in range(len(a))],'integer_list'))
        if 'duplicate' in text and any(w in text for w in ['original order','preserving order','preserving the order','maintaining the order']) and any(w in text for w in ['remove','unique']):
            candidates.append(('deduplicate_in_order',lambda a:list(dict.fromkeys(a)),'integer_list'))
        if any(w in text for w in ['strictly increasing','sorted in ascending order and all elements are unique']):
            candidates.append(('strictly_increasing',lambda a:all(a[i]<a[i+1] for i in range(len(a)-1)),'integer_list'))
    matrices=arity==1 and all(len(a)==1 and isinstance(a[0],list) and a[0] and all(isinstance(row,list) for row in a[0]) for a,_,_ in examples)
    if matrices and 'transpose' in text and 'matrix' in text:
        candidates.append(('matrix_transpose',lambda a:[list(row) for row in zip(*a)],'matrix'))
    if ints and arity==1:
        bool_outputs=all(type(expected) is bool for _,_,expected in examples)
        if 'prime' in text and bool_outputs and not any(w in text for w in ['factor','next','previous','nth','sum','count','digit']):
            if any(w in text for w in ['not prime','non-prime','not a prime']):candidates.append(('not_prime',lambda n:not is_prime(n),'integer'))
            else:candidates.append(('is_prime',is_prime,'integer'))
        if 'factorial' in text and not any(w in text for w in ['sum','digit','trailing','double','modulo','prime']):
            candidates.append(('factorial',math.factorial,'factorial'))
        if any(w in text for w in ['sum of digits','sum of the digits']) and not any(w in text for w in ['power','factorial','square','until','single digit','repeated']):
            candidates.append(('digit_sum',lambda n:sum(int(c) for c in str(abs(n))),'integer'))
    if ints and arity==2 and any(w in text for w in ['greatest common divisor','greatest common factor','gcd']) and not any(w in text for w in ['list','array','extended','lcm']):
        candidates.append(('gcd',math.gcd,'integer_pair'))
    matching=[]
    for name,fn,domain in candidates:
        # Do not let large source literals turn trusted host-side oracle checks into
        # unbounded work. Unsupported examples simply receive no independent tier.
        if domain in {'string','parentheses'} and any(len(a[0])>(24 if name=='longest_unique_substring_length' else 256) for a,_,_ in examples):continue
        if domain=='integer_list' and any(len(a[0])>100 for a,_,_ in examples):continue
        if domain=='matrix' and any(sum(len(row) for row in a[0])>100 for a,_,_ in examples):continue
        if domain=='factorial' and any(not 0<=a[0]<=20 for a,_,_ in examples):continue
        if domain in {'integer','integer_pair'} and any(abs(x)>1000000 for a,_,_ in examples for x in a):continue
        try:
            if all(fn(*args)==expected for args,_,expected in examples):matching.append((name,fn,domain))
        except (ValueError,TypeError,OverflowError,RecursionError):pass
    return matching[0] if len(matching)==1 else None


def independent_cases(instruction,examples,seed=42):
    contract=matching_oracle(instruction,examples)
    if contract is None:return None,[]
    name,fn,domain=contract;rng=random.Random(seed)
    text=instruction.lower();positive='positive' in text and 'non-positive' not in text
    nonempty=any(w in text for w in ['non-empty','nonempty','at least one'])
    cases=[]
    if domain in {'string','parentheses'}:
        alphabet='()' if domain=='parentheses' else 'abAB 09!?é'
        fixed=['','a','aa','ab','aba','A a','A!a','12321','éaé'] if domain=='string' else ['','(',')','()','))((','(()())','()())()']
        cases=[(s,) for s in fixed if not nonempty or s]
        cases += [(''.join(rng.choice(alphabet) for _ in range(rng.randint(1 if nonempty else 0,12))),) for _ in range(24)]
    elif domain=='integer_list':
        lo=1 if positive else 0 if 'non-negative' in text or 'nonnegative' in text else -5
        fixed=[[1],[1,1],[2,1],[3,1,2],[0],[0,0],[-2,1,-2,3]]
        cases=[(a,) for a in fixed if all(x>=lo for x in a)]
        if not nonempty:cases.append(([],))
        cases += [([rng.randint(lo,8) for _ in range(rng.randint(1 if nonempty else 0,10))],) for _ in range(24)]
    elif domain=='matrix':
        cases=[([[rng.randint(-5,5) for _ in range(cols)] for _ in range(rows)],) for rows in range(1,5) for cols in range(1,5)]
    elif domain=='factorial':cases=[(n,) for n in range(1 if positive else 0,11)]
    elif domain=='integer':
        lo=1 if positive or 'natural' in text else 0 if 'non-negative' in text or 'nonnegative' in text else -10
        cases=[(n,) for n in [0,1,2,3,4,9,11,25,49,97,101] if n>=lo]
        cases += [(rng.randint(lo,1000),) for _ in range(24)]
    elif domain=='integer_pair':cases=[(a,b) for a,b in [(1,1),(6,9),(13,17),(12,24),(0,7),(7,0),(36,48)] if not positive or min(a,b)>0]+[(rng.randint(1,100),rng.randint(1,100)) for _ in range(24)]
    result=[];seen=set()
    for args in cases:
        key=repr(args)
        if key in seen:continue
        seen.add(key)
        try:expected=fn(*args)
        except (ValueError,TypeError):continue
        result.append((args,expected))
    return name,result
