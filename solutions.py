import re, math

# www.hackerrank.com/jb_merville
# Name Of The Problem - Difficulty - Points Earned On Completion

# ---- ALGORITHM ----

# Diagonal Difference - Easy - 10pts
def diagonalDifference(arr):
    sum1 = 0
    sum2 = 0
    for i in range (len(arr)):
        sum1 += arr[i][i]
    for i in range (len(arr), 0):
        print(i)
        sum2 += arr[i][len(arr)-i]
    result = sum1 - sum2
    if result < 0 : result = - result
    return result

# Bon Appetit - Easy - 10pts
def bonAppetit(bill, k, b):
    bill_allergic = 0
    total_bill = 0
    for i in range(len(bill)):
        if not i == k: bill_allergic += bill[i]
        total_bill += bill[i]
    if b == (bill_allergic / 2) : print("Bon Appetit")
    else : print( int((total_bill - bill_allergic) / 2))
    
# Sock Merchant - Easy - 10pts
def sockMerchant(n, ar):
    socks = {}
    pairs = 0
    for i in ar:
        if not socks.__contains__(i): socks.__setitem__(i, 1)
        else: socks[i] = socks[i] + 1
    for i in socks.keys():
        pairs += int(socks[i] / 2)
    return pairs

# Drawing Book - Easy - 10pts
def pageCount(n, p):
    result = p
    if p % 2 == 0: result += 1
    if (n - result) / 2 < (result - 1) / 2: result = math.ceil((n - result) /2)
    else: result = (result - 1) // 2
    return result

# Grading Students - Easy - 10pts
def gradingStudents(grades):
    result = []
    for i in grades:
        nearest = 5*math.ceil(i/5)
        if nearest - i < 3 and i > 37:
            result.append(nearest) 
        else: result.append(i)
    return result

# Staircase - Easy - 10pts
def staircase(n):
    space = " "
    hashtage ="#"
    for i in range(1, n+1):
        print(space*(n-i) + (hashtage*i))

# Super Reduced String - Easy - 10pts
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

# Min-Max Sum - Easy - 10pts
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

# Birthday Cake Candles - Easy - 10pts
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
    print(max)

# Compare The Triplets - Easy - 10pts
def compareTriplets(a, b):
    ab = [0,0]
    for i in range(len(a)):
        if a[i] > b[i]: ab[0] += 1
        elif a[i] < b[i]: ab[1] += 1
    return ab

# Apples And Oranges - Easy - 10pts
def countApplesAndOranges(s, t, a, b, apples, oranges):
    apple_count = 0
    orange_count = 0
    for i in apples:
        if a + i >= s and a + i <= t: apple_count += 1
    for i in oranges:
        if b + i >= s and b + i <= t: orange_count += 1
    print(apple_count, orange_count)

# Kangaroo - Easy - 10pts
def kangaroo(x1, v1, x2, v2):
    result = "NO"
    if v1 > v2:
        while x1 < x2:
            x1 += v1
            x2 += v2
            if x1 == x2: result = "YES"
    return result

# Between Two Sets - Easy - 10pts
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

# Breaking The Records - Easy - 10pts
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

# Birthday Chocolate - Easy - 10pts
def birthday(s, d, m):
    count = 0
    for i in range(len(s)-m+1):
        peice = 0
        for b in range(i, i+m):
            peice += s[b]
        if peice == d: count += 1
    return count

# Divisible Sum Pairs - Easy - 10pts
def divisibleSumPairs(n, k, ar):
    count = 0
    for a in range(len(ar)-1):
        for b in range(a+1, len(ar)):
            sum = ar[a] + ar[b]
            if sum % k == 0: count += 1
    return count

# Migratory Birds - Easy - 10pts
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

# Time Conversion - Easy - 15pts
def timeConversion(s):
    if s[-2:] == "PM":
        if not s[:2] == "12" : s = str(int(s[:2]) + 12) + s[2:]
    elif s[:2] == "12": s = "00" + s[2:]
    return s[:-2]

# Day Of The Programmer - Easy - 15pts
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

# The Hurdle Race - Easy - 15pts
def hurdleRace(k, height):
    max_height = 0
    result = 0
    for h in height:
        if h > max_height: max_height = h
    if k < max_height: result = abs(k-max_height)
    return result

# Camel Case - Easy - 15pts
def camelcase(s):
    words = 1
    for i in s:
        if ord(i) < 91: words += 1
    return words

# Beautiful Days At The Movies - Easy - 15pts
def beautifulDays(i, j, k):
    beautifulD = 0
    for i in range (i, j+1):
        if ((reverse(i) - i) / k) % 1 == 0: beautifulD += 1
    return beautifulD

# Viral Adverstising - Easy - 15pts
def viralAdvertising(n):
    liked = 5
    cumulative = 0
    for _ in range(1, n+1):
        liked = math.floor(liked/2) 
        cumulative += liked
        liked *= 3
    return cumulative

# Save The Prisonner - Easy - 15pts
def saveThePrisoner(seats, candy, s):
    a = s + candy - 1
    result = 0
    if a > seats:
        if a % seats == 0:
            result = seats
        else: result = a % seats
    else: result = a
    return result

# Counting Valleys - Easy - 15pts
def countingValleys(s):
    count = [0,0]
    valleys = 0
    for i in range(len(s)):
        if s[i] == "D": count[1] -= 1
        else: count[1] += 1
        if count[0] < 0 and count[1] == 0: valleys += 1
        count[0] = count[1]
    return valleys

# Electronic Shop - Easy - 15pts
def getMoneySpent(keyboards, drives, b):
    max = -1 
    for k in keyboards:
        for d in drives:
            price = k + d
            if price > max and price <= b: max = price
    return max

# Jumping On Clouds: Revisited - Easy - 15pts
def jumpingOnClouds(c, k):
    n = len(c)
    return 100 - sum(2 * c[i%n] + 1 for i in range(0, n, k))

# Cats And Mouse - Easy - 15pts
def catAndMouse(x, y, z):
    result = ""
    if abs(x-z) < abs(y-z): result = "Cat A"
    elif abs(x-z) > abs(y-z): result = "Cat B"
    else: result = "Mouse C"
    return result

# Library Fine - Easy - 15pts
def libraryFine(d1, m1, y1, d2, m2, y2):
    hackos = 0
    if y1 == y2:
        if m1 == m2:
            if d1 <= d2: hackos = 0
            else: hackos = (d1 - d2) * 15
        elif m1 >= m2: hackos = (m1 - m2) * 500
    elif y1 > y2: hackos = 10000
    return hackos

# Forming A Magic Square - Medium - 20pts
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
        print("%d: %d" %(i, cost))
        if i == 0: min_cost = cost
        if cost < min_cost: min_cost = cost
    return min_cost

# Picking Numbers - Easy - 20pts
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

# Unfinished - Climbing The Leaderboard - Medium - 20pts
def climbingLeaderboard(scores, alice):
    result = []
    dist_scores = []
    for s in range(len(scores)):
        if not dist_scores.__contains__(scores[s]): dist_scores.append(scores[s])
    l = len(dist_scores)
    print(dist_scores)
    print(sorted(set(scores), reverse = True))
    for a in alice:
        while (l > 0) and (a >= dist_scores[l-1]):
            l -= 1
        result.append(l+1)
    return result

# Designer PDF Viewer - Easy - 20pts
def designerPdfViewer(h, word):
    area = 0
    alp = "abcdefghijklmnopqrstuvwxyz"
    for l in range(len(word)):
        if area < h[alp.index(word[l])]: area = h[alp.index(word[l])]
    return area * len(word)

# Utopian Tree - Easy - 20pts
def utopianTree(n):
    height = 0
    for i in range(n+1):
        if i % 2 == 0: height += 1
        else: height *= 2
        #print("height:%d i:%d div:%d" %(height, i, i%2))
    return height

# Angry Professor - Easy - 20pts
def angryProfessor(k, a):
    here = 0
    result = "YES"
    for i in a:
        if i <= 0: here += 1
    if here >= k : result = "NO"
    return result

# Circular Array Rotation - Easy - 20pts
def circularArrayRotation(a, k, queries):
    for i in range(k):
       a.insert(0, a[-1])
       del a[-1]
    result = []
    for i in queries:
        result.append(a[i])
    return result

# Extra Long Factorials - Medium - 20pts
def extraLongFactorials(n):
    factorial = 1
    for i in range(2, n+1):
        factorial = i * factorial
    print(factorial)

# Append And Delete - Easy - 20pts
def appendAndDelete(s, t, k):
    i = 0
    result = "No"
    while  i < len(t) and i < len(s) and s[i] == t[i]:
       i += 1
    n = len(s)
    while k > 0 and not s == t and n > i:
        s = s[:n-1]
        print(k, s)
        n -= 1
        k -= 1
    while n < len(t) and k > 0 and not s == t:
        print(k)
        s = s + t[n]
        n += 1
        k -= 1
    print(s, t)
    if s == t: result = "Yes"
    return result

# Sherlock And Squares - Easy - 20pts
def squares(a, b):
    count = math.floor(math.sqrt(b)) - (math.ceil(math.sqrt(a)) - 1)
    return count

# Sequence Equation - Easy - 20pts
def permutationEquation(p):
    result = []
    for i in range(1, len(p) + 1):
        b = p.index(i) + 1
        b = p.index(b) + 1
        result.append(b)
    return result

# Non Divisible Subset - Medium - 20pts 
def nonDivisibleSubset(k, S):
    count = [0] * k
    for x in S:
        count[x % k] += 1
    ans = min(count[0], 1)
    for rem in range(1, (k + 1) // 2):
        ans += max(count[rem], count[k - rem])
    if k % 2 == 0:
        ans += min(count[k // 2], 1)
    return ans

# Repeated String - Easy - 20pts
def repeatedString(s, n): 
    a = sum(1 for i in range(len(s)) if s[i] == "a")
    result = (n // len(s)) * a
    result += sum(1 for i in range(0, n % len(s)) if s[i] == "a") 
    return result

# Jumping On The Clouds - Easy - 20pts
def jumpingOnCloudsV2(c):
    i = 0
    count = -1
    while i < len(c):
        if i < (len(c) -2) and c[i+2] == 0:
            i += 2
        else: i += 1
        count += 1
    return count

# Minimum Distances - Easy - 20pts
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

# Halloween Sale - Easy - 20pts
def howManyGames(p, d, m, s):
    i = 0
    while s >= m:
        price = (p - (d*i))
        if price < m: price = m
        s -= price
        if s >= 0: i += 1
    return i

# Service Lane - Easy - 20pts
def serviceLane(width, n, cases):
    vehicules = []
    for i in cases:
       vehicules.append(min(width[y] for y in range(i[0], i[1]+1)))
    return vehicules

# Equalize Array - Easy - 20pts
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

# The Time In Words - Medium - 25pts
def timeInWords(h, m):
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifthteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
    time = ''
    past = False
    if m > 30:
        m = abs(m-60)
        past = True
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

# Chocolate Feast - Easy - 25pts
def chocolateFeast(n, c, m):
    bars = n // c
    wrapper = bars
    while wrapper >= m:
        newBars = wrapper // m
        wrapper %= m
        wrapper += newBars
        bars += newBars
    return bars

# Find Digits - Easy - 25pts
def findDigits(n):
    count = 0  
    digit = n
    for i in range(1, len(str(n)) +1): 
        if not digit % 10 == 0 and n % (digit % 10) == 0: count +=1
        digit = digit // 10
    return count

# Cut The Sticks - Easy - 25pts
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

# Flatland Space Stations - Easy - 25pts
def flatlandSpaceStations(n, c):
    c.sort()
    maxDistance = max(c[0], n-c[len(c)-1]-1)
    if len(c)>1:
        extrems = max(math.floor((c[i+1] - c[i])/2) for i in range(len(c)-1))
        maxDistance = max(maxDistance, extrems)
    return maxDistance

# Fair Ratoins - Easy - 25pts
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
    if not B[-2]%2 == 0: loafs = "NO"
    return loafs

# ACM ICPC Team - Easy - 25pts - Using Strings (Not Optimal)
def acmTeam2(topic): 
    team = 0
    all_teams = []
    max_topics = 0
    for a in range(len(topic)):
        for b in range(a+1, len(topic)):
            team = sum(1 for i in range(len(topic[0])) if topic[a][i] == "1" or topic[b][i] == "1")
            if team > max_topics: max_topics = team
            all_teams.append(team)
    count = 0
    for i in all_teams:
        if i == max_topics: count += 1
    return [max_topics, count]

# ACM ICPC Team - Easy - 25pts - Using Bits Operations
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

# Lisa's Workbook - Easy - 25pts
def workbook(n, k, arr):
    page = 0
    special = 0
    for a in range(n):
        ex = [0,0]
        b = 0
        while b < arr[a]:
            page += 1
            ex[0]=ex[1]+1
            b +=k
            if b > arr[a]:
                ex[1] += arr[a]%k
            else: ex[1] = b
            if page >= ex[0] and page <= ex[1]:
                special +=1
    return special

# Taum And B'day - Easy - 25pts
def taumBday(b, w, bc, wc, z):
    cost = 0
    if bc + z < wc:
        cost = (bc * b) + ((bc + z) * w)
    elif wc + z < bc:
        cost = (wc * w) + ((wc + z) * b)
    else:
        cost = (wc * w) + (bc * b)
    return cost

# Happy Ladybugs - Easy - 30pts - (Not Optimal)
def happyLadybugs(b):
    colors = {}
    for i in b:
        if not i in colors: colors[i] = 1
        else: colors[i] = colors[i]+1
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

# Strange Counter - Easy - 30pts
def strangeCounter(t):
    time = [3,3]
    while time[1] < t:
        time[0]*=2
        time[1] = time[1] + time[0]
    return time[1]-t+1

# 3D Surface Area - Medium - 30pts
def surfaceArea(A):
    price = 0
    for a in range(len(A)):
        for b in range(len(A[0])):
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
    return price

# Cavity Map - Easy - 30pts
def cavityMap(grid):
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

# Manasa Stones - Easy - 30pts
def stones(n, a, b):
    result = []
    for i in range(n):
        print(i, (n-1-i), i*a+(n-1-i)*b)
        if i*b+(n-1-i)*a not in result:
            result.append(i*b+(n-1-i)*a)
    result.sort()
    return result

# The Grid Search - Medium - 30 pts
def gridSearch(G, P):
    result = "NO"
    l = len(P[0])
    for i in range(len(G)-len(P)+1):
        for z in range(len(G[0])-l+1):
            print(G[i][z:z+l])
            if G[i][z:z+l] == P[0]: 
                for a in range(len(P)-1):
                    result = "YES"
                    c = 1
                    for b in range(i+1, i+len(P)):
                        print(c, b, G[b][z:z+l])
                        if not G[b][z:z+l] == P[c]: result = "NO"
                        c += 1
                    if result == "YES": return result
    return result

# Queen's Attack II - Medium - 30pts - Using Matrix (Not Optimal)
def queensAttack2(n, k, r_q, c_q, obstacles):
    matrix = [ [0 for i in range(n)]  for b in range(n)]
    for i in range(len(obstacles)):
        matrix[n - obstacles[i][0]][obstacles[i][1] - 1] = 1
    i = n -r_q
    b = c_q - 1
    matrix[i][b] = 2
    printMatrix(matrix, False)
    count = 0
    while i < n - 1 and not matrix[i+1][b] == 1:
        i += 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while 0 < i and not matrix[i-1][b] == 1:
        i -= 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while 0 < b and not matrix[i][b-1] == 1:
        b -= 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while b < n - 1 and not matrix[i][b+1] == 1:
        b += 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while 0 < i and b < n - 1 and not matrix[i-1][b+1] == 1:
        b += 1
        i -= 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while 0 < i and 0 < b and not matrix[i-1][b-1] == 1:
        b -= 1
        i -= 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while i < n - 1 and 0 < b and not matrix[i+1][b-1] == 1:
        b -= 1
        i += 1
        count += 1
    i = n -r_q
    b = c_q - 1
    while i < n - 1 and b < n - 1 and not matrix[i+1][b+1] == 1:
        b += 1
        i += 1
        count += 1
    return count

# Queen's Attack II - Medium - 30pts
def queensAttack(n, k, r_q, c_q, obstacles):
    matrix = [ [0 for i in range(n)]  for b in range(n)]
    for i in range(len(obstacles)):
        matrix[n - obstacles[i][0]][obstacles[i][1] - 1] = 1
    i = n -r_q
    b = c_q - 1
    matrix[i][b] = 2
    count = 2*(n-1) + 3* min(n-c_q, n-r_q, c_q-1, r_q-1) + max(n-c_q, n-r_q, c_q-1, r_q-1) 
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
            if obstacles[i][0] > r_q and obstacles[i][1] < c_q: # top left
                if top_l[0] == 0 and top_l[1] == 0: 
                    top_l[0] = obstacles[i][0]
                    top_l[1] = obstacles[i][1]
                if r < abs(top_l[0] - r_q):
                    top_l[0] = obstacles[i][0]
                    top_l[1] = obstacles[i][1]
            if obstacles[i][0] > r_q and obstacles[i][1] > c_q: # top right
                if top_r[0] == 0 and top_r[1] == 0: 
                    top_r[0] = obstacles[i][0]
                    top_r[1] = obstacles[i][1]
                if r < abs(top_r[0] - r_q):
                    top_r[0] = obstacles[i][0]
                    top_r[1] = obstacles[i][1]
            if obstacles[i][0] < r_q and obstacles[i][1] < c_q: # bottom left
                if bot_l[0] == 0 and bot_l[1] == 0: 
                    bot_l[0] = obstacles[i][0]
                    bot_l[1] = obstacles[i][1]
                if r < abs(bot_l[0] - r_q):
                    bot_l[0] = obstacles[i][0]
                    bot_l[1] = obstacles[i][1]
            if obstacles[i][0] < r_q and obstacles[i][1] > c_q: # bottom right
                if bot_r[0] == 0 and bot_r[1] == 0: 
                    bot_r[0] = obstacles[i][0]
                    bot_r[1] = obstacles[i][1]
                if r < abs(bot_r[0] - r_q):
                    bot_r[0] = obstacles[i][0]
                    bot_r[1] = obstacles[i][1]
        elif obstacles[i][0] == r_q and obstacles[i][1] < c_q: # left_row
            if left == 0:
                left = obstacles[i][1]
            elif left < obstacles[i][1]: left = obstacles[i][1]
        elif obstacles[i][0] == r_q and obstacles[i][1] > c_q: # right_row
            if right == n:
                right = obstacles[i][1]
            elif right > obstacles[i][1]: right = obstacles[i][1]
        elif obstacles[i][1] == c_q and obstacles[i][0] > r_q: # top_column
            if top == n:
                top = obstacles[i][0]
            elif top > obstacles[i][0]: top = obstacles[i][0]
        elif obstacles[i][1] == c_q and obstacles[i][0] < r_q: # bottom_column
            if bottom == 0:
                bottom = obstacles[i][0]
            elif bottom < obstacles[i][0]: bottom = obstacles[i][0]  
    count -= min(bot_l[0], bot_l[1])
    count -= min(n-top_l[0] + 1, top_l[1])
    count -= min(n-top_r[0] + 1, n-top_r[1] + 1) 
    count -= min(bot_r[0],n-bot_r[1] + 1)  
    count -= left + (n-right+1) + (n-top+1) + bottom
    return count

# Organizing Containers - Medium - 30pts
def organizingContainers(container):
    maxType = [ 0 for i in range(len(container[0]))]
    types = [ 0 for i in range(len(container[0]))]
    for i in range(len(container)):
        for b in range(len(types)):
            types[b] += container[i][b]
            maxType[i] += container[i][b]
    maxType.sort()
    types.sort()
    possible = "Possible"
    for i in range(len(types)):
        if types[i] > maxType[i] : possible = "Impossible"
    return possible

# Modified Kaprekar Numbers - Easy - 30pts
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

# Encryption - Medium - 30pts
def encryption(s):
    s = re.sub('-[A-Za-z]*', '', s)
    enc = []
    result = ''
    col = math.ceil(math.sqrt(len(s)))
    encrypt = ''
    b = 0
    for i in range(len(s)):
        encrypt = encrypt + s[i]
        b = (b+1) % col
        if b == 0:
            enc.append(encrypt)
            encrypt = ''
    if not b == 0: enc.append(encrypt)
    for a in range(len(enc[0])):
        encrypt = ''
        for b in range(len(enc)):
            if  a < len(enc[b]):
                encrypt += enc[b][a]
        result += encrypt + " "
    return result

# Bigger Is Greater - Medium - 35pts
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

# Absolute Permutation - Medium - 40pts
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

# The Bomberman Game - Medium - 40pts - In Progress
def bomberMan(n, grid):
    for i in grid:
        print(i)
    print("----")
    newG = []
    for a in range(len(grid)):
        newG.append([])
        for b in range(len(grid[0])):
            if grid[a][b] == "O": newG[a].append(1)
            else: newG[a].append(0)
    printGrid(newG)
    for c in range(n):
        coord = []
        for a in range(len(newG)):
            for b in range(len(newG[0])):
                if newG[a][b] == 1: coord.append([a,b])
                newG[a][b] = 1
        newG = detonation(newG, coord)
        print("----")
        printGrid(newG)
    grid = []
    for a in range(len(newG)):
        grid.append("")
        for b in range(len(newG[0])):
            if newG[a][b] == 1: grid[a]+= "O"
            else: grid[a] += "."
    printGrid(grid)
    return grid

def detonation(grid, coord):
    for i in range(len(coord)):
        a = coord[i][0]
        b = coord[i][1]
        grid[a][b] = 0
        if a > 0: grid[a-1][b] = 0
        if a < len(grid)-1: grid[a+1][b] = 0
        if b > 0: grid[a][b-1] = 0
        if b < len(grid[0])-1: grid[a][b+1] = 0
    return grid

def printGrid(grid):
    for a in range(len(grid)):
        line = ""
        for b in range(len(grid[0])):
            line += str(grid[a][b])
        print(line)

# ---- STRINGS ----

# Mars Exploration - Easy - 15pts
def marsExploration(s):
    count = 0
    for i in range(0,len(s),3):
        if not s[i] == "S": 
            count += 1
        if not s[i+1] == "O": 
            count += 1
        if not s[i+2] == "S": 
            count += 1
    return count

# Strong Password - Easy - 15pts
def minimumNumber(n, password):
    count = 0
    if re.search("\d", password): count +=1
    if re.search("[a-z]", password): count +=1
    if re.search("[A-Z]", password): count +=1
    if re.search("[^a-zA-z0-9]", password): count +=1
    count = 4-count
    if len(password)+count < 6: count = 6-len(password)
    return count

# Caesar Cipher - Easy - 15pts
def caesarCipher(s, k):
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

# Alternating Characters - Easy - 20pts
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
    count = len(s) - len(repetition)
    return count

# HackerRank In A String! - Easy - 20pts
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

# Pangrams - Easy - 20pts
def pangrams(s):
    dict = {chr(ord("a")+i):0 for i in range(26)}
    s = s.lower()
    for i in range(len(s)):
        if not s[i] == " ": dict[s[i]] += 1
    result = "pangram"
    for i in dict.keys():
        if dict[i] == 0: result = "not pangram"
    return result

# Weighted Uniform Strings - Easy - 20pts
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
    for i in queries:
        found = False
        for a in weight.keys():
            if (i/a)%1 == 0.0 and i <= a*weight[a]: 
                found = True         
        if found: result.append("Yes")
        else: result.append("No")
    return result

# Separate The Numbers - Easy - 20pts
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

# Gemstones - Easy - 20pts
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

# Beautiful Binary String - Easy - 20pts
def beautifulBinaryString(b):
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
        i += 1
    return count

# The Love-Letter Mystery -  Easy - 20pts
def theLoveLetterMystery(s):
    count = 0
    for i in range(len(s)//2):
            count += abs(ord(s[i])-ord(s[-i-1]))
    return count

# Funny String - Easy - 25pts
def funnyString(s):
    i = 0
    result = "Funny"
    while i < len(s)-1 and abs(ord(s[i])-ord(s[i+1])) == abs(ord(s[len(s)-1-i])-ord(s[len(s)-2-i])):
        i += 1
    if not i == len(s)-1: result = "Not Funny"
    return result
    
# Two Strings - Easy - 25pts
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

# Palindrome Index - Easy - 25pts
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

# Anagram - Easy - 25pts
def anagram(s):
    result = 0
    if not len(s) % 2 == 0:
        result = -1
    else:
        l1 = {}
        for i in range(len(s)//2):
            if s[i] not in l1: 
                l1[s[i]] = 1
                result += 1
            else: 
                l1[s[i]] += 1
                result += 1
        for i in range(len(s)//2):
            if s[-i-1] in l1 and l1[s[-i-1]] > 0: 
                l1[s[-i-1]] -= 1
                result -= 1
    return result

# String Construction - Easy - 25pts
def stringConstruction(s):
    ltr = []
    price = 0
    for i in range(len(s)):
        if s[i] not in ltr: 
            ltr.append(s[i])
            price += 1
    return price

# Making Anagrams - Easy - 30pts
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

# Games Of Thrones - I - Easy - 30pts
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

# Sherlock and the Valid String - Medium - 35pts
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
    if len(nb)>1:
        oneTwo = [list(nb.keys())[0], nb[list(nb.keys())[0]], list(nb.keys())[1], nb[list(nb.keys())[1]]]
        if len(nb) > 2: result = "NO"
        elif  oneTwo[1] > 1 and oneTwo[3] > 1: result = "NO"
        elif not ((oneTwo[0] == 1 and oneTwo[1] == 1) or (oneTwo[2] == 1 and oneTwo[3] == 1)):
            if abs(oneTwo[0] - oneTwo[2]) > 1 :  result = "NO"
    return result
