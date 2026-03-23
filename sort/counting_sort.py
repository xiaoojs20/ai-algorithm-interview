def counting_sort(arr):
    """
    计数排序 (Counting Sort)
    
    原理：
    非比较排序 (Non-comparison sort)。适用于数据范围较小的整数。
    1. 统计数组中每个元素出现的次数。
    2. 计算每个元素的起始位置坐标。
    3. 将数据放到目标位置。
    
    复杂度分析：
    - 时间复杂度: O(n + k) (n 是元素数，k 是数据的范围 [max-min])
    - 空间复杂度: O(k) (额外存储计数数组)
    - 稳定性: 稳定排序 (本实现为稳定版)
    """
    if not arr:
        return arr
        
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1
    
    # 1. 建立计数数组
    count = [0] * range_val
    for x in arr:
        count[x - min_val] += 1
        
    # 2. 累计频率，确定各值在输出数组中的结束位置
    for i in range(1, len(count)):
        count[i] += count[i - 1]
        
    # 3. 反向遍历输入数组，保证稳定性 (Stable Sort)
    output = [0] * len(arr)
    for x in reversed(arr):
        idx = x - min_val
        output[count[idx] - 1] = x
        count[idx] -= 1
        
    return output

if __name__ == "__main__":
    test_arr = [4, 2, 2, 8, 3, 3, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {counting_sort(test_arr)}")
