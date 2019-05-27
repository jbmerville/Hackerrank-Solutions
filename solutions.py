 
# Problem Solving - Larry's Array - Medium
def larrysArray(A):
    search = 1
    i = 0
    while i < len(A):
        while i< len(A) and A[i] == search:
            i = search
            search += 1
        i += 1
        if i < len(A) and A[i] == search:
            k = i
            while k > search: 
                s = A[k-2]    
                A[k-2] = A[k]
                A[k] = A[k-1]
                A[k-1] = s
                k -= 2
            if k+1 < len(A) and k > search-1:
                s = A[k-1]    
                A[k-1] = A[k]
                A[k] = A[k+1]
                A[k+1] = s  
            i = search 
            search += 1

    if A[-2] > A[-1] or search < len(A) : return "NO"
    return "YES"



# Problem Solving - Absolute Permutation - Medium
def absolutePermutation(n, k):
    result = [k+1]
    for i in range(2, n+1):
            if -k+i-k-1 >= 0 and -k+i-k-1 < len(result):
                if -k + i > 0 and -k +  i <= n and not result[-k-k+i-1] == -k+i:
                    result.append(-k+i)
                elif k+i > 0 and k+i <= n:
                    result.append(k+i)
                else:
                    return [-1]
            else:
                if -k + i > 0 and -k +  i <= n:
                    result.append(-k+i)
                elif k+i > 0 and k+i <= n:
                    result.append(k+i)
                else:
                    return [-1]
    return result


# Problem Solving - Non-Divisible Subset - Medium
def nonDivisibleSubset(k, S):
    cnt = [0] * k
    for x in S:
        cnt[x % k] += 1
        
    ans = min(cnt[0], 1)
    for rem in range(1, (k + 1) // 2):
        ans += max(cnt[rem], cnt[k - rem])
    if k % 2 == 0:
        ans += min(cnt[k // 2], 1)
    return ans
 
       


# Problem Solving - The Hurdle Race - Easy
def hurdleRace(k, height):
    max_height = 0
    result = 0
    for h in height:
        if h > max_height: max_height = h
    if k < max_height: result = abs(k-max_height)
    return result


# Problem Solving - Luck Balance - Easy
def luckBalance(k, contests):
    imp = []
    result = 0
    for i in contests:
        if i[1] == 1: imp.append(i[0])
        else: result += i[0]
    imp.sort()
    for i in range(len(imp)-1, -1, -1):
        if k > 0: 
            result += imp[i]
            k -= 1
        else: result -= imp[i]
    return result


# Problem Solving - Apple and Orange - Easy
def countApplesAndOranges(s, t, a, b, apples, oranges):
    count = [0,0]
    for i in apples:
        if a + i >= s and a + i <= t: count[0] += 1
    for i in oranges:
        if b + i >= s and b + i <= t: count[1] += 1
    print(count[0])
    print(count[1])


# Problem Solving - Correctness and the Loop Invariant - Easy
def insertion_sort(l):
    for i in range(1, len(l)):
        j = i
        key = l[i]
        while (j > 0) and (l[j-1] > key):
            l[j] = l[j-1]
            j -= 1
        l[j] = key

m = int(input().strip())
ar = [int(i) for i in input().strip().split()]
insertion_sort(ar)
print(" ".join(map(str,ar)))


# Problem Solving - Sherlock and the Valid String - Medium
def isValid(s):
    ltr = {}
    result = "YES"
    for i in range(len(s)):
        if s[i] not in ltr: ltr[s[i]] = 1
        else: ltr[s[i]] += 1
        nb = {}
    for i in ltr.keys():
        if ltr[i] not in nb: nb[ltr[i]] = 1
        else: nb[ltr[i]] += 1
    print(nb)
        if len(nb)>1:
        oneTwo = [list(nb.keys())[0], nb[list(nb.keys())[0]], list(nb.keys())[1], nb[list(nb.keys())[1]]]
        print(oneTwo)
        if len(nb) > 2: result = "NO"
        elif  oneTwo[1] > 1 and oneTwo[3] > 1: result = "NO"
        elif not ((oneTwo[0] == 1 and oneTwo[1] == 1) or (oneTwo[2] == 1 and oneTwo[3] == 1)):
            print(abs(oneTwo[0] - oneTwo[2]))
            if abs(oneTwo[0] - oneTwo[2]) > 1 :  result = "NO"
    return result


# Problem Solving - Intro to Tutorial Challenges - Easy
def introTutorial(V, arr):
    left = 0
    right =  len(arr) - 1
    while left <= right:
        mid = left + (right - left)//2
        if arr[mid] == V: 
            return mid
        elif V > arr[mid]: left = mid + 1
        else: right = mid - 1
    return False


# Problem Solving - Strong Password - Easy
def minimumNumber(n, password):
        count = 0
    if re.search("\d", password): count +=1
    if re.search("[a-z]", password): count +=1
    if re.search("[A-Z]", password): count +=1
    if re.search("[^a-zA-z0-9]", password): count +=1
    count = 4-count
    if len(password)+count < 6: count = 6-len(password)
    return count
    


# Problem Solving - String Construction  - Easy
def stringConstruction(s):
    ltr = []
    price = 0
    for i in range(len(s)):
        if s[i] not in ltr: 
            ltr.append(s[i])
            price += 1
    return price
                


# Problem Solving - Game of Thrones - I - Easy
def gameOfThrones(s):
    ltr = {}
    for i in range(len(s)):
        if s[i] not in ltr: ltr[s[i]] = 1
        else: ltr[s[i]] += 1
    result = 0
    for i in ltr.keys():
        if not ltr[i] % 2 == 0:
            result += 1
    if result > 1: result = "NO"
    else:result = "YES"
    return result
            
            


# Problem Solving - Making Anagrams - Easy
def makingAnagrams(s1, s2):
    d1 = {}
    d2 = {}
    for i in range(len(s1)):
        if s1[i] not in d1: d1[s1[i]] = 1
        else: d1[s1[i]] += 1
    for i in range(len(s2)):
        if s2[i] not in d2: d2[s2[i]] = 1
        else: d2[s2[i]] += 1

    result = 0
    for i in d1.keys():
        if i in d2: result += min(d1[i], d2[i])
    result = len(s1) + len(s2) - 2 * result
    return result


# Problem Solving - Palindrome Index - Easy
def palindromeIndex(s):
    i = 0
    while i < len(s)//2:
        if not s[i] == s[-i-1]: 
            if s[i] == s[-i-2]:
                k = i+1
                while k < len(s) // 2 and s[k] == s[-k-2]:
                    k+=1
                if k == len(s) // 2: return len(s) - i - 1
            if s[i+1] == s[-i-1]:
                k = i+1
                while k < len(s) // 2 and s[k] == s[-k]:
                    k+=1
                if k == len(s) // 2: return i
        i += 1
    return -1


# Problem Solving - The Love-Letter Mystery - Easy
def theLoveLetterMystery(s):
    count = 0
    for i in range(len(s)//2):
            count += abs(ord(s[i])-ord(s[-i-1]))
    return count


# Problem Solving - Beautiful Binary String - Easy
def beautifulBinaryString(b):
    print("----")
    count = 0
    i = 0
    while i < len(b)-2:
        t = True
        while i < len(b)-2 and b[i] == "0" and b[i+1] == "1" and b[i+2] == "0":
            if t: 
                count += 1
                t = False
            else:
                t = True
            i += 2
            print("i:%d count:%d" %(i-2, count))
        i += 1
    return count


# Problem Solving - Gemstones - Easy
def gemstones(arr):
    gemstones = []
    for i in range(len(arr[0])):
        if arr[0][i] not in gemstones: gemstones.append(arr[0][i])
    for i in range(1, len(arr)):
        stones = []
        for a in range(len(arr[i])):
            if arr[i][a] in gemstones: stones.append(arr[i][a])
        b = 0
        while b < len(gemstones):
            if gemstones[b] not in stones: 
                gemstones.pop(b)
                b -= 1
            b += 1
        print(i, gemstones)


    return len(gemstones)


# Problem Solving - Funny String - Easy
def funnyString(s):
    i = 0
    result = "Funny"
    while i < len(s)-1 and abs(ord(s[i])-ord(s[i+1])) == abs(ord(s[len(s)-1-i])-ord(s[len(s)-2-i])):
        i += 1
    if not i == len(s)-1: result = "Not Funny"
    return result


# Problem Solving - Bigger is Greater - Medium
def biggerIsGreater(w):
    letters = [w[len(w)-1]]
    for a in range(len(w)-2, -1, -1):
        letters.append(w[a])
        optimal = None
        for b in range(a+1, len(w)):
            if w[b] > w[a]:
                if optimal == None: optimal = b
                elif w[optimal] > w[b]: optimal = b
        if not optimal == None: 
            letters.remove(w[optimal])
            letters.sort()
            end = ''
            for c in letters:
                end += c
            w = w[:a] + w[optimal] + end
            return w
    return "no answer"


# Problem Solving - Separate the Numbers - Easy
def separateNumbers(s):
    result = "NO"
    for k in range(1, len(s)//2+1):
        nb = lengthNextNb(s[0:k])
        i = 0
        while i+nb[1]+nb[3] <= len(s) and nb[0]+1 == int(s[i+nb[1]: i+nb[1]+nb[3]]):
            inc = nb[1]
            nb = lengthNextNb(s[i+nb[1]: i+nb[1]+nb[3]])
            i += inc
        if i+nb[1]+nb[3] == len(s)+nb[3]: result = "YES " + s[0:k]
    if len(s) == 1: result = "NO"
    print(result)

def lengthNextNb(s):
    return [int(s), len(s), int(s)+1, len(str(int(s)+1))]


# Problem Solving - Weighted Uniform Strings - Easy
def weightedUniformStrings(s, queries):
    result = []
    weight = {}
    i = 0
    while i < len(s):
        k = i
        w = 0
        while k < len(s) and s[i] == s[k]:
            w += ord(s[i]) - ord("a") + 1
            k += 1
        if (ord(s[i]) - ord("a") + 1 in weight and weight[ord(s[i]) - ord("a") + 1] < k - i) or ord(s[i]) - ord("a") + 1 not in weight:
            weight[ord(s[i]) - ord("a") + 1] = k - i
        i = k
    print(weight)
    for i in queries:
        found = False
        for a in weight.keys():
                        if (i/a)%1 == 0.0 and i <= a*weight[a]: 
                found = True         
        if found: result.append("Yes")
        else: result.append("No")
    return result


# Problem Solving - Pangrams - Easy
def pangrams(s):
    dict = {chr(ord("a")+i):0 for i in range(26)}
    s = s.lower()
    for i in range(len(s)):
        if not s[i] == " ": dict[s[i]] += 1
    result = "pangram"
    for i in dict.keys():
        if dict[i] == 0: result = "not pangram"
    return result


# Problem Solving - HackerRank in a String! - Easy
def hackerrankInString(s):
    found = 0
    r = 0
    for i in range(len(s)):
        if s[i] == "h" and found == 0: found += 1
        if s[i] == "a" and found == 1: found += 1
        if s[i] == "c" and found == 2: found += 1
        if s[i] == "k" and found == 3: found += 1
        if s[i] == "e" and found == 4: found += 1
        if s[i] == "r" and found == 5: 
            found += 1
            r = i
        if s[i] == "r" and not r == i and found == 6: found += 1
        if s[i] == "a" and found == 7: found += 1
        if s[i] == "n" and found == 8: found += 1
        if s[i] == "k" and found == 9: found += 1
        if found == 10: return "YES"
    return "NO"


# Problem Solving - Caesar Cipher - Easy
def caesarCipher(s, k):
    print("k:%d, kmod26:%d" % (k, k%26))
    print(ord("a"))
    print(ord("b"))
    result = ""
    for i in range(len(s)):
        if ord(s[i]) <= ord("z") and ord(s[i]) >= ord("a"):
            if ord(s[i])+k%26 > ord("z"): 
                result += chr(ord("a") + (ord(s[i])+k%26) % ord("z")-1)

            else: result += chr(ord(s[i])+k%26)
        elif ord(s[i]) <= ord("Z") and ord(s[i]) >= ord("A"):
            if ord(s[i])+k%26 > ord("Z"): 
                result += chr(ord("A") + (ord(s[i])+k%26) % ord("Z")-1)
            else: result += chr(ord(s[i])+k%26)
        else: result += s[i]
    return result
 


# Problem Solving - Super Reduced String - Easy
def superReducedString(s):
    i = 0
    while i <len(s)-1:
        if s[i] == s[i+1]:
            k = 0
            while i-k-1>0 and i+2+k<len(s) and s[i-k-1] == s[i+2+k]:
                k += 1
            s = s[:i-k]+s[i+2+k:]
            i -= k+2
        i += 1
        if i <0: i = 0
        print(i)

    if s == "": s = "Empty String"
    return s


# Problem Solving - CamelCase - Easy
def camelcase(s):
    words = 1
    for i in s:
        if ord(i) < 91: words += 1
    return words
    


#  - Hash Tables: Ice Cream Parlor - Medium
def whatFlavors(cost, money):
    original = [i for i in cost]
    cost.sort()
    result = []
    for i in cost:
        if binarySearch(cost, money - i): result = [i, money-i]
    for i in range(2):
        result[i] = original.index(result[i]) + 1
        original[result[i]-1] = 0
    result.sort()
    print(str(result[0]) + " " + str(result[1]))

def binarySearch (cost, nb):
    left = 0
    right = len(cost)-1
    while left <= right:
        mid = (left + right)//2
        if cost[mid] == nb:
            return nb
        elif nb < cost[mid]:
            right = mid - 1
        else: 
            left = mid + 1
    return False


# Problem Solving - Lisa's Workbook - Easy
def workbook(n, k, arr):
    page = 0
    special = 0
    for a in range(n):
        print("----")
        ex = [0,0]
        b = 0
        while b < arr[a]:
            page += 1
            ex[0]=ex[1]+1
            b +=k
            if b > arr[a]:
                ex[1] += arr[a]%k
            else: ex[1] = b
            print(ex)
            if page >= ex[0] and page <= ex[1]:
                special +=1


    return special
        
        


# Problem Solving - Two Strings - Easy
def twoStrings(s1, s2):
    letterS1 = []
    letterS2 = []
    for i in s1:
        if i not in letterS1: letterS1.append(i)
    for i in s2:
        if i not in letterS2: letterS2.append(i)
    result = "NO"
    for a in letterS1:
        if a in letterS2: result = "YES"
    return result
            
            


# Problem Solving - Alternating Characters  - Easy
def alternatingCharacters(s):
    i =0
    count = 0
    repetition = []
    while i < len(s):
        k = i
        while k < len(s)-1 and s[i] == s[k+1] :
            k += 1 
        repetition.append(k-i+1)
        i = k+1
    print(repetition)
    count = len(s) - len(repetition)
    return count


#  - Arrays: Left Rotation - Easy
def rotLeft(a, d):
    shift = len(a) - d
    result = []
    for x in a[-shift:]:
        result.append(x)
    for x in a[0:-shift]:
        result.append(x)
    return result


# Problem Solving - Manasa and Stones - Easy
def stones(n, a, b):
    print("----")
    result = []
    for i in range(n):
        print(i, (n-1-i), i*a+(n-1-i)*b)
        if i*b+(n-1-i)*a not in result:
            result.append(i*b+(n-1-i)*a)
    result.sort()
    return result


# Problem Solving - Cavity Map - Easy
def cavityMap(grid):
    x = "X"
    ind = []
    for a in range(1, len(grid)-1):
        for b in range(1, len(grid[0])-1):
            if grid[a][b] > grid[a-1][b] and grid[a][b] > grid[a+1][b] and grid[a][b] > grid[a][b+1] and grid[a][b] > grid[a][b-1]: ind.append([a,b])
    result = [""]* len(grid)
    for a in range(len(grid)):
        for b in range(len(grid[0])):
            if not [a,b] in ind: result[a] += str(grid[a][b])
            else: result[a] += "X"
    return result


# Problem Solving - Fair Rations - Easy
def fairRations(B):
    loafs = 0
    for i in range(len(B)):
        if not B[i]%2 == 0:
            B[i] += 1
            loafs += 1
            if i == 0:
                 B[i+1] += 1
            elif i == len(B)-1:
                B[i-1] += 1
            else:
                if not B[i-1]%2 == 0: B[i-1] += 1
                else: B[i+1] += 1
            loafs += 1
    print(B, B[-2])
    if not B[-2]%2 == 0: loafs = "NO"
    return loafs


# Problem Solving - 3D Surface Area - Medium
def surfaceArea(A):
    print(A)
    price = 0
    for a in range(len(A)):
        for b in range(len(A[0])):
            print("which:%d" %A[a][b])
            if a == 0:
                price += A[a][b]
            if a == len(A)-1:
                price += A[a][b]
            if b == len(A[0])-1:
                price += A[a][b]
            if b == 0:
                price += A[a][b]
            if b + 1 < len(A[0]):
                if A[a][b] > A[a][b+1]: price += A[a][b]-A[a][b+1]
            if b - 1 >= 0:
                if A[a][b] > A[a][b-1]: price += A[a][b]-A[a][b-1]
            if a - 1 >= 0:
                if A[a][b] > A[a-1][b]: price += A[a][b]-A[a-1][b]
            if a + 1 < len(A):
                if A[a][b] > A[a+1][b]: price += A[a][b]-A[a+1][b]
            price += 2
            print(price)
    return price


# Problem Solving - Strange Counter - Easy
def strangeCounter(t):
    time = [3,3]
    while time[1] < t:
        time[0]*=2
        time[1] = time[1] + time[0]
    return time[1]-t+1
    


# Problem Solving - Happy Ladybugs - Easy
def happyLadybugs(b):
    colors = {}
    for i in b:
        if not i in colors: colors[i] = 1
        else: colors[i] = colors[i]+1
    print(colors)
    result = "NO"
    if colors.keys().__contains__("_"): 
        result = "YES"
        for i in colors.keys():
            if colors[i] == 1 and not i == "_" : result = "NO"
    else:
        result = "YES"
        if len(b) == 1: result = "NO"
        if len(b) > 1:
            if not b[0] == b[1]: result = "NO"
        if len(b) > 2:
            if not b[len(b)-1] == b[len(b)-2]: result = "NO"
        for i in range(1, len(b)-1):
            if  not(b[i] == b[i-1] or b[i] == b[i+1]): result = "NO"
    return result


# Problem Solving - Flatland Space Stations - Easy
def flatlandSpaceStations(n, c):
    c.sort()
    maxDistance = max(c[0], n-c[len(c)-1]-1)
    if len(c)>1:
        extrems = max(math.floor((c[i+1] - c[i])/2) for i in range(len(c)-1))
        maxDistance = max(maxDistance, extrems)
    return maxDistance


# Problem Solving - Mars Exploration - Easy
def marsExploration(s):
    count = 0
    print(len(s))
    for i in range(0,len(s),3):
        if not s[i] == "S": 
            count += 1
            print(i)
        if not s[i+1] == "O": 
            count += 1
            print(i+1)
        if not s[i+2] == "S": 
            count += 1
            print(i+2)
    return count


# Problem Solving - Grading Students - Easy
def gradingStudents(grades):
    result = []
    for i in grades:
        nearest = 5*math.ceil(i/5)
        if nearest - i < 3 and i > 37:
            result.append(nearest) 
        else: result.append(i)
    return result


# Problem Solving - The Grid Search - Medium
def gridSearch(G, P):
    result = "NO"
    l = len(P[0])
    for i in range(len(G)-len(P)+1):
        for z in range(len(G[0])-l+1):
            if G[i][z:z+l] == P[0]: 
                result = "YES"
                c = 1
                for b in range(i+1, i+len(P)):
                    print(c, b, G[b][z:z+l])
                    if not G[b][z:z+l] == P[c]: result = "NO"
                    c += 1
                if result == "YES": return result
    return result


# Problem Solving - Birthday Chocolate - Easy
def birthday(s, d, m):
    count = 0
    for i in range(len(s)-m+1):
        peice = 0
        for b in range(i, i+m):
            peice += s[b]
        if peice == d: count += 1
    return count


# Problem Solving - Kangaroo - Easy
def kangaroo(x1, v1, x2, v2):
    result = "NO"
    if not v1 <= v2:
        while x1 < x2:
            x1 += v1
            x2 += v2
        if x1 == x2: result = "YES"
    return result


# Problem Solving - Service Lane - Easy
def serviceLane(width, n, cases):
    vehicules = []
    for i in cases:
       vehicules.append(min(width[y] for y in range(i[0], i[1]+1)))
    return vehicules


# Problem Solving - Chocolate Feast  - Easy
def chocolateFeast(n, c, m):
    bars = n // c
    wrapper = bars
    print(bars, wrapper)
    while wrapper >= m:
        newBars = wrapper // m
        print(newBars)
        wrapper %= m
        wrapper += newBars
        bars += newBars
    return bars


# Problem Solving - The Time in Words - Medium
def timeInWords(h, m):
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifthteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
    time = ''
    past = False
    if m > 30:
        m = abs(m-60)
        past = True
    print(m)
    if m == 0 and not past:
        time = numbers[h-1] + " o' clock"
    elif m == 30:
        time = "half past " + numbers[h-1]
    else:
        if m == 15:
            time = "quarter"
        else:
            if m > 20:
                time = numbers[19] + " "
            time += numbers[(m-1)%20]
            if m == 1:
                time += " minute"
            else: time += " minutes"
        if past:
            time += " to " + numbers[h]
        else:
            time += " past " + numbers[h-1]
    return time


# Problem Solving - Halloween Sale - Easy
def howManyGames(p, d, m, s):
        i = 0
    while s >= m:
        price = (p - (d*i))
        if price < m: price = m
        s -= price
        if s >= 0: i += 1
    return i


# Problem Solving - Minimum Distances - Easy
def minimumDistances(a):
    seen = {}
    minimum = None
    for i in range(len(a)):
        if a[i] not in seen.keys(): seen[a[i]] = i
        else:
            if minimum == None: minimum = i - seen[a[i]]
            elif minimum > i - seen[a[i]]: minimum = i - seen[a[i]]
    if minimum == None: minimum = -1
    return minimum


# Problem Solving - Beautiful Triplets - Easy
def beautifulTriplets(d, arr):
    count = 0
    for a in range(len(arr)-2):
        for b in range (a+1, len(arr)-1):
            if arr[b] - arr[a] == d:
                for c in range(b+1, len(arr)):
                    if arr[c] - arr[b] == d:
                        count += 1
    return count


# Problem Solving - Modified Kaprekar Numbers - Easy
def kaprekarNumbers(p, q):
    result = ''
    for i in range(p, q+1):
        nb = str(i**2)
        left = 0
        if not nb[:len(nb)-len(str(i))] == "": left = int(nb[:len(nb)-len(str(i))])
        right = int(nb[len(nb)-len(str(i)):])
        if left + right == i:
            result += str(i) + " "
    if result == '': result = 'INVALID RANGE'
    print(result)


# Problem Solving - Insert a Node at the Tail of a Linked List - Easy
def insertNodeAtTail(head, data):
    if head == None:
        node = SinglyLinkedListNode(data)
        return node
    curr_node = head
    while not curr_node.next == None:
        curr_node = curr_node.next
    node = SinglyLinkedListNode(data)
    curr_node.next = node
    return head

    


# Problem Solving - Print the Elements of a Linked List - Easy
def printLinkedList(head):
    while not head.data == None:
        print(head.data)
        head = head.next


# Problem Solving - Encryption - Medium
def encryption(s):

    s = re.sub('-[A-Za-z]*', '', s)
    enc = []
    result = ''
    row = math.floor(math.sqrt(len(s)))
    col = math.ceil(math.sqrt(len(s)))
    print('row:%d col:%d row x col:%d len:%d ' %(row, col, row*col, len(s)))
    encrypt = ''
    b = 0
    for i in range(len(s)):
        encrypt = encrypt + s[i]
        b = (b+1) % col
        if b == 0:
            enc.append(encrypt)
            encrypt = ''
    if not b == 0: enc.append(encrypt)
    print(enc)
    for a in range(len(enc[0])):
        encrypt = ''
        for b in range(len(enc)):
            if  a < len(enc[b]):
                encrypt += enc[b][a]
        result += encrypt + " "
    return result
            


# Problem Solving - Sparse Arrays - Medium
def matchingStrings(strings, queries):
    queryCount = []
    for i in queries:
        count = 0
        for s in strings:
            if i == s : count += 1
        queryCount.append(count)
    return queryCount


# Problem Solving - Organizing Containers of Balls - Medium
def organizingContainers(container):
    print("\n----")
    maxType = [ 0 for i in range(len(container[0]))]
    types = [ 0 for i in range(len(container[0]))]
    for i in range(len(container)):
        for b in range(len(types)):
            types[b] += container[i][b]
            maxType[i] += container[i][b]
    maxType.sort()
    types.sort()
    print('maxType:%s' % str(maxType))
    print('type:%s' % str(types))
    possible = "Possible"
    for i in range(len(types)):
        if types[i] > maxType[i] : possible = "Impossible"
    return possible


# Problem Solving - Taum and B'day - Easy
def taumBday(b, w, bc, wc, z):
    cost = 0
    if bc + z < wc:
        cost = (bc * b) + ((bc + z) * w)
    elif wc + z < bc:
        cost = (wc * w) + ((wc + z) * b)
    else:
        cost = (wc * w) + (bc * b)
    return cost
    


# Problem Solving - Arrays - DS - Easy
def reverseArray(a):
    result = []
    for i in range(len(a)):
        result.append(a[len(a)-1-i])
    return result


# Problem Solving - ACM ICPC Team - Easy
def acmTeam(topic):
    bit = []
    max_topics = 0
    max_way = 0
    for i in topic:
        bit.append(int('0b' + i, 2))
        for a in range(len(bit)):
        for b in range(a + 1, len(bit)):
            count = 0
            nb = bit[a] | bit[b]
            while nb:
                count += nb & 1
                nb>>=1
            if count > max_topics: 
                max_topics = count
                max_way = 1
            elif count == max_topics: max_way += 1
    return [max_topics, max_way]
    
        

                


# Problem Solving - Queen's Attack II - Medium
def queensAttack(n, k, r_q, c_q, obstacles):
    count = 2*(n-1) + 3* min(n-c_q, n-r_q, c_q-1, r_q-1) + max(n-c_q, n-r_q, c_q-1, r_q-1) 
    print("-----")
    print("init count:%d" %count)
    bot_r = [0, 0]
    bot_l = [0, 0]
    top_r = [n+1, n+1]
    top_l = [n+1, n+1]

    left = 0
    right = n+1
    top = n+1
    bottom = 0

    for i in range(len(obstacles)):
        r = abs(obstacles[i][0] - r_q)
        c = abs(obstacles[i][1] - c_q)
        if r ==  c:
            if obstacles[i][0] > r_q and obstacles[i][1] < c_q:                 if top_l[0] == 0 and top_l[1] == 0: 
                    top_l[0] = obstacles[i][0]
                    top_l[1] = obstacles[i][1]
                if r < abs(top_l[0] - r_q):
                    top_l[0] = obstacles[i][0]
                    top_l[1] = obstacles[i][1]
            if obstacles[i][0] > r_q and obstacles[i][1] > c_q:                 if top_r[0] == 0 and top_r[1] == 0: 
                    top_r[0] = obstacles[i][0]
                    top_r[1] = obstacles[i][1]
                if r < abs(top_r[0] - r_q):
                    top_r[0] = obstacles[i][0]
                    top_r[1] = obstacles[i][1]
            if obstacles[i][0] < r_q and obstacles[i][1] < c_q:                 if bot_l[0] == 0 and bot_l[1] == 0: 
                    bot_l[0] = obstacles[i][0]
                    bot_l[1] = obstacles[i][1]
                if r < abs(bot_l[0] - r_q):
                    bot_l[0] = obstacles[i][0]
                    bot_l[1] = obstacles[i][1]
            if obstacles[i][0] < r_q and obstacles[i][1] > c_q:                 if bot_r[0] == 0 and bot_r[1] == 0: 
                    bot_r[0] = obstacles[i][0]
                    bot_r[1] = obstacles[i][1]
                if r < abs(bot_r[0] - r_q):
                    bot_r[0] = obstacles[i][0]
                    bot_r[1] = obstacles[i][1]
        elif obstacles[i][0] == r_q and obstacles[i][1] < c_q:             if left == 0:
                left = obstacles[i][1]
            elif left < obstacles[i][1]: left = obstacles[i][1]
        elif obstacles[i][0] == r_q and obstacles[i][1] > c_q:             if right == n:
                right = obstacles[i][1]
            elif right > obstacles[i][1]: right = obstacles[i][1]
        elif obstacles[i][1] == c_q and obstacles[i][0] > r_q:             if top == n:
                top = obstacles[i][0]
            elif top > obstacles[i][0]: top = obstacles[i][0]
        elif obstacles[i][1] == c_q and obstacles[i][0] < r_q:             if bottom == 0:
                bottom = obstacles[i][0]
            elif bottom < obstacles[i][0]: bottom = obstacles[i][0]  
    print("----")       
    print("closest dia:")
    print("bot left:%s | bot right:%s | top left:%s | top right:%s " % (str(bot_l), str(bot_r), str(top_l), str(top_r)))   
    count -= min(bot_l[0], bot_l[1])
    count -= min(n-top_l[0] + 1, top_l[1])
    print("bot left:%d" % (min(bot_l[0], bot_l[1]) ))
    print("bot right:%d" % (min(bot_r[0],n-bot_r[1] + 1)))
    print("top left:%d" % (min(n-top_l[0] +1, top_l[1])))
    print("top right:%d" % (min(n-top_r[0] +1, n-top_r[1] +1)))
    count -= min(n-top_r[0] + 1, n-top_r[1] + 1) 
    count -= min(bot_r[0],n-bot_r[1] + 1) 
    print("count after dia:%d" %count )
    print("----")
    print("closest row/col:")
    print("left:%d | right:%d | top:%d | bottom:%d " % (left, right, top, bottom))   
    count -= left + (n-right+1) + (n-top+1) + bottom
    print("count after dia:%d" %count )

    return count


# Problem Solving - Equalize the Array - Easy
def equalizeArray(arr):
    dictionnary = {}
    highest_freq = [0,0]
    
    for i in arr:
        if not i in dictionnary.keys(): 
            dictionnary[i] = 1
        else: dictionnary[i] = dictionnary[i] + 1
        if dictionnary[i] > highest_freq[0]: 
            highest_freq[1] = i
            highest_freq[0] = dictionnary[i]
    return len(arr) - dictionnary[highest_freq[1]]


# Problem Solving - Jumping on the Clouds - Easy
def jumpingOnClouds(c):
    i = 0
    count = -1
    while i < len(c):
        if i < (len(c) -2) and c[i+2] == 0:
            i += 2
        else: i += 1
        count += 1
    return count


# Problem Solving - Repeated String - Easy
def repeatedString(s, n): 
    a = sum(1 for i in range(len(s)) if s[i] == "a")
    result = (n // len(s)) * a
    result += sum(1 for i in range(0, n % len(s)) if s[i] == "a") 
    return result


# Problem Solving - Cut the sticks - Easy
def cutTheSticks(arr):
    result = []
    while len(arr)>0:
        smallest = arr[0]
        for i in arr:
            if i < smallest: smallest = i
        cut = 0
        i = 0
        while i < len(arr):
            if not arr[i] == smallest:
                arr[i] -= smallest
                i += 1
            else: arr.pop(i)
            cut += 1
        result.append(cut)
    return result


# Problem Solving - Library Fine - Easy
def libraryFine(d1, m1, y1, d2, m2, y2):
    hackos = 0
    if y1 == y2:
        if m1 == m2:
            if d1 <= d2: hackos = 0
            else: hackos = (d1 - d2) * 15
        elif m1 >= m2: hackos = (m1 - m2) * 500
    elif y1 > y2: hackos = 10000

    
    
    return hackos


# Problem Solving - Sherlock and Squares - Easy
def squares(a, b):
    count = math.floor(math.sqrt(b)) - (math.ceil(math.sqrt(a)) - 1)
    return count
                


# Problem Solving - Append and Delete - Easy
def appendAndDelete(s, t, k):
    i = 0
    result = "No"
    while  i < len(t) and i < len(s) and s[i] == t[i]:
       i += 1
    n = len(s) -1 
    while k > 0 and not s == t and n > i:
        s = s[:n-1]
        n -= 1
        k -= 1
    while n < len(t) and k > 0 and not s == t:
        s = s + t[n]
        n += 1
        k -= 1
    if s == t: result = "Yes"
    return result


# Problem Solving - Extra Long Factorials - Medium
def extraLongFactorials(n):
    factorial = 1
    for i in range(2, n+1):
        factorial = i * factorial
    print(factorial)


# Problem Solving - Find Digits - Easy
def findDigits(n):
    count = 0  
    digit = n
    for i in range(1, len(str(n)) +1): 
        if not digit % 10 == 0 and n % (digit % 10) == 0: count +=1
        digit = digit // 10
    return count

        


# Problem Solving - Jumping on the Clouds: Revisited - Easy
def jumpingOnClouds(c, k):
    n = len(c)
    return 100 - sum(2 * c[i%n] + 1 for i in range(0, n, k))
        


# Problem Solving - Sequence Equation - Easy
def permutationEquation(p):
    result = []
    for i in range(1, len(p) + 1):
        b = p.index(i) + 1
        b = p.index(b) + 1
        result.append(b)
    return result


# Problem Solving - Circular Array Rotation - Easy
def circularArrayRotation(a, k, queries):
    for i in range(k):
       a.insert(0, a[-1])
       del a[-1]
    result = []
    for i in queries:
        result.append(a[i])
    return result


# Problem Solving - Save the Prisoner! - Easy
def saveThePrisoner(seats, candy, s):
    a = s + candy - 1
    result = 0
    if a > seats:
        if a % seats == 0:
            result = seats
        else: result = a % seats
    else: result = a
    return result


# Problem Solving - Viral Advertising - Easy
def viralAdvertising(n):
    liked = 5
    cumulative = 0
    for i in range(1, n+1):
        liked = math.floor(liked/2) 
        cumulative += liked
        liked *= 3
    return cumulative


# Problem Solving - Beautiful Days at the Movies - Easy
def beautifulDays(i, j, k):
    beautifulD = 0
    for i in range (i, j+1):
        if ((reverse(i) - i) / k) % 1 == 0: beautifulD += 1
    return beautifulD
def reverse(n):
    result = 0
    while n > 0:
        result = result*10 + (n % 10)
        n = n // 10
    return result
    


# Problem Solving - Angry Professor - Easy
def angryProfessor(k, a):
    here = 0
    result = "YES"
    for i in a:
        if i <= 0: here += 1
    if here >= k : result = "NO"
    return result


# Problem Solving - Utopian Tree - Easy
def utopianTree(n):
    height = 0
    for i in range(n+1):
        if i % 2 == 0: height += 1
        else: height *= 2        
    return height
    


# Problem Solving - Designer PDF Viewer - Easy
1
def designerPdfViewer(h, word):
    area = 0
    alp = "abcdefghijklmnopqrstuvwxyz"
    for l in range(len(word)):
        if area < h[alp.index(word[l])]: area = h[alp.index(word[l])]
    return area * len(word)


# Problem Solving - Climbing the Leaderboard - Medium
def climbingLeaderboard(scores, alice):
    result = []
    dist_scores = sorted(set(scores), reverse = True)
    l = len(dist_scores)
    for a in alice:
        while (l > 0) and (a >= dist_scores[l-1]):
            l -= 1
        result.append(l+1)
    return result
    


# Problem Solving - Picking Numbers - Easy
def pickingNumbers(a):
    pairs = {}
    max_size = 0
    for x in a:
        if not pairs.__contains__(x): pairs.__setitem__(x, 1)
        else: pairs[x] = pairs[x] + 1
    for x in pairs.keys():
        for y in pairs.keys():
            if pairs[x] > max_size: max_size = pairs[x]
            if not x == y and abs(x - y) == 1 and pairs[x] + pairs[y] > max_size: max_size = pairs[x] + pairs[y]
    return max_size


# Problem Solving - Forming a Magic Square - Medium
def formingMagicSquare(s):
    min_cost = 0 
    m = [[[2,9,4], [7,5,3], [6,1,8]]] 
    for i in range(4):
        n = [ [m[i][0][2],m[i][1][2],m[i][2][2]], [m[i][0][1],m[i][1][1],m[i][2][1]], [m[i][0][0],m[i][1][0],m[i][2][0]] ]
        m.append(n)
    for i in range(4):
        n = [ [m[i][0][2],m[i][0][1],m[i][0][0]], [m[i][1][2],m[i][1][1],m[i][1][0]], [m[i][2][2],m[i][2][1],m[i][2][0]] ]
        m.append(n)
    
    for i in range(len(m)):
        cost = 0
        for y in range(len(s)):
            for x in range(len(s)):
                cost += abs(m[i][y][x]-s[y][x])
        if i == 0: min_cost = cost
        if cost < min_cost: min_cost = cost
    return min_cost


# Problem Solving - Cats and a Mouse - Easy
def catAndMouse(x, y, z):
    result = ""
    if abs(x-z) < abs(y-z): result = "Cat A"
    elif abs(x-z) > abs(y-z): result = "Cat B"
    else: result = "Mouse C"
    return result


# Problem Solving - Electronics Shop - Easy
def getMoneySpent(keyboards, drives, b):
    max = -1 
    for k in keyboards:
        for d in drives:
            price = k + d
            if price > max and price <= b: max = price
    return max


# Problem Solving - Counting Valleys - Easy
def countingValleys(n, s):
    count = [0,0]
    valleys = 0
    for i in range(len(s)):
        if s[i] == "D": count[1] -= 1
        else: count[1] += 1
        if count[0] < 0 and count[1] == 0: valleys += 1
        count[0] = count[1]
    return valleys
        


# Problem Solving - Drawing Book  - Easy
def pageCount(n, p):
    result = p
    if p % 2 == 0: result += 1
    if (n - result) / 2 < (result - 1) / 2: result = math.ceil((n - result) /2)
    else: result = (result - 1) // 2
    return result


# Problem Solving - Sock Merchant - Easy
def sockMerchant(n, ar):
    socks = {}
    pairs = 0
    for i in ar:
        if not socks.__contains__(i): socks.__setitem__(i, 1)
        else: socks[i] = socks[i] + 1
    for i in socks.keys():
        pairs += int(socks[i] / 2)
    return pairs


# Problem Solving - Bon Appétit - Easy
def bonAppetit(bill, k, b):
    bill_allergic = 0
    total_bill = 0
    for i in range(len(bill)):
        if not i == k: bill_allergic += bill[i]
        total_bill += bill[i]
    if b == (bill_allergic / 2) : print("Bon Appetit")
    else : print( int((total_bill - bill_allergic) / 2))
    


# Problem Solving - Day of the Programmer - Easy
def dayOfProgrammer(year):
    result = ""
    isLeapYear = False
    if year > 1918:
        isLeapYear = year % 400 == 0 or (year % 4 == 0 and not year % 100 == 0)
    elif year == 1918:
        return "26.09.1918"
    else: 
        isLeapYear = year % 4 == 0
    if isLeapYear: result = "12.09." + str(year)
    else: result = "13.09." + str(year)
    return result


# Problem Solving - Migratory Birds - Easy
def migratoryBirds(arr):
    birds_freq = {}
    maxFreq = 1
    bird = 0
    for i in arr:
        if i > bird: bird = i
        if not birds_freq.__contains__(i): birds_freq.__setitem__(i, 1)
        else: 
            birds_freq[i] = birds_freq[i] +1
            if birds_freq[i] > maxFreq: maxFreq = birds_freq[i]

    for keys in birds_freq.keys():
        if birds_freq[keys] == maxFreq and bird > keys: bird = keys
    return bird
         


# Problem Solving - Divisible Sum Pairs - Easy
 
def divisibleSumPairs(n, k, ar):
    count = 0
    for a in range(len(ar)-1):
        for b in range(a+1, len(ar)):
            sum = ar[a] + ar[b]
            if sum % k == 0: count += 1
    return count


# Problem Solving - Breaking the Records - Easy
def breakingRecords(scores):
    min_max = [scores[0], scores[0]]
    inc_dec = [0, 0]
    for i in scores:
        if min_max[0] > i:
            min_max[0] = i
            inc_dec[0] += 1
        if min_max[1] < i:
            min_max[1] = i
            inc_dec[1] += 1
    return inc_dec[1], inc_dec[0]


# Problem Solving - Between Two Sets - Easy
def getTotalX(a, b):
    count = 0
    for i in range(a[len(a)-1], b[0]+1):
        isNb = True
        for x in a:
            if not i % x == 0: isNb = False
        for x in b:
            if not x % i == 0: isNb = False
        if isNb: count += 1
    return count


# Problem Solving - Time Conversion - Easy
def timeConversion(s):
    if s[-2:] == "PM":
        if not s[:2] == "12" : s = str(int(s[:2]) + 12) + s[2:]
    elif s[:2] == "12": s = "00" + s[2:]
    return s[:-2]


# Problem Solving - Birthday Cake Candles - Easy
def birthdayCakeCandles(ar):
    candles = {}
    for i in ar:
        if not candles.__contains__(i): candles.__setitem__(i, 1)
        else: 
            count = candles[i]
            candles.__setitem__(i, count + 1)
    max = 0
    for i in candles.keys():
        if max < candles[i]: max = candles[i]
    return max


# Problem Solving - Mini-Max Sum - Easy
def miniMaxSum(arr):
    max_sum = 0
    min_sum = 0
    for a in range(len(arr)):
        sum = 0
        for b in range(len(arr)):
            if a != b : sum += arr[b]
        if a == 0: 
            min_sum = sum
            max_sum = sum
        if max_sum < sum : max_sum =  sum
        if min_sum > sum : min_sum = sum
    print(min_sum, max_sum)


# Problem Solving - Staircase - Easy
def staircase(n):
    space = " "
    hashtage ="    for i in range(1, n+1):
        print(space*(n-i) + hashtage*i)


# Problem Solving - Plus Minus - Easy
def plusMinus(arr):
    result = [0,0,0]
    for i in arr:
        if i > 0: result[0] += 1
        elif i < 0: result[1] += 1
        else: result[2] += 1
    print(result[0]/len(arr))
    print(result[1]/len(arr))
    print(result[2]/len(arr))


# Problem Solving - Diagonal Difference - Easy
def diagonalDifference(arr):
    sum1 = 0
    sum2 = 0
    for i in range (len(arr)):
        sum1 += arr[i][i]
    for i in range (len(arr)):
        sum2 += arr[i][len(arr)-1-i]
    result = sum1 - sum2
    if result < 0 : result = - result
    return result


# Problem Solving - A Very Big Sum - Easy
 
def aVeryBigSum(ar):
    sum = 0
    for i in ar:
        sum += i
    return sum


# Problem Solving - Compare the Triplets - Easy
def compareTriplets(a, b):
    result = [0, 0]
    for i in range(len(a)):
        if a[i] > b[i]: result[0] += 1
        elif a[i] < b[i] : result[1] += 1
    return result


# Problem Solving - Simple Array Sum - Easy
2
def simpleArraySum(ar):
    sum = 0
    for i in ar:
        sum += i
    return sum


# Problem Solving - Solve Me First - Easy
def solveMeFirst(a,b):
	return a + b


num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)


# Problem Solving - Lonely Integer - Easy
def lonelyinteger(a):
    found = {}
    for i in a:
        if not i in found: found[i] = 1
        else: found[i] += 1
    for i in found.keys():
        if found[i] == 1: return i


