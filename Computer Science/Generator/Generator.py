
#The number of possible combinations to put n items into one bag is 2^n

# generate all combinations of N items
def powerSet(items):
    N = len(items)
    # enumerate the 2**N possible combinations
    for i in range(2**N):
        combo = []
        for j in range(N):
            # test bit jth of integer i
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        yield combo

val = powerSet([1,2,3,7,8,9])
print(next(val))
print(next(val))


#The number of possible combinations to put n items into two bags is 3^n

def yieldAllCombos(items):
    """
        Generates all combinations of N items into two bags, whereby each
        item is in one or zero bags.

        Yields a tuple, (bag1, bag2), where each bag is represented as a list
        of which item(s) are in each bag.
    """

    N = len(items)
    for i in range(3 ** N):
        bag1 = []
        bag2 = []
        for j in range(N):
            if (i // 3 ** j) % 3 == 1:
                bag1.append(items[j])
            if (i // 3 ** j) % 3 == 2:
                bag2.append(items[j])
        yield (bag1, bag2)
        
val2 = yieldAllCombos([1,2,3,7,8,9])
print(next(val2))
print(next(val2))