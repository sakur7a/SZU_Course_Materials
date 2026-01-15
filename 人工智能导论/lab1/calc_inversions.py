def bruteforce(state: str) -> int:
    "双重循环暴力计算逆序对"
    nums = [int(c) for c in state if c != 'x']
    inversions = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                inversions += 1
    return inversions

def merge_sort(a : list, l : int, r : int) -> int:
    "归并排序计算逆序对"
    res = 0
    if l == r:
        return 0
    mid = (l + r) // 2
    i = l
    j = mid + 1
    temp = []
    
    res += merge_sort(a, i, mid) + merge_sort(a, j, r)

    while i <= mid and j <= r:
        if a[i] <= a[j]:
            temp.append(a[i])
            i += 1
        else:
            temp.append(a[j])
            j += 1
            res += mid - i + 1

    while i <= mid:
        temp.append(a[i])
        i += 1
    while j <= r:
        temp.append(a[j])
        j += 1

    for k in range(len(temp)):
        a[l + k] = temp[k]

    return res


input_elements = input().split()
numeric_list = [int(elem) for elem in input_elements if elem.isdigit()]
print(merge_sort(numeric_list, 0, 7))




