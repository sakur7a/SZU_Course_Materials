import heapq
import time


def calculate_inversion(state: str) -> int:
    """
    计算八数码状态的逆序数
    """
    nums = [int(c) for c in state if c != 'x']
    inversion_count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                inversion_count += 1
    return inversion_count


def is_solvable(initial_state: str, target_state: str) -> bool:
    """
    判断初始状态是否可转换为目标状态（逆序数奇偶性相同）
    """
    initial_inv = calculate_inversion(initial_state)
    target_inv = calculate_inversion(target_state)
    # 奇偶性相同则可解
    return initial_inv % 2 == target_inv % 2


def manhattan_distance(state: str, target_pos: dict) -> int:
    """
    计算当前状态的曼哈顿距离（启发函数h1(n)）
    :param state: 当前八数码状态（字符串）
    :param target_pos: 目标状态中每个数字的坐标映射（如{"1":(0,0), "2":(0,1)}）
    :return: 曼哈顿距离总和（整数）
    """
    distance = 0
    for idx, char in enumerate(state):
        if char == 'x':  # 跳过空白格
            continue
        # 当前数字的坐标（转换为3x3矩阵的行和列）
        current_row = idx // 3
        current_col = idx % 3
        # 目标坐标（从预定义的映射中获取）
        target_row, target_col = target_pos[char]
        # 累加曼哈顿距离（行差绝对值 + 列差绝对值）
        distance += abs(current_row - target_row) + abs(current_col - target_col)
    return distance


def a_star_8puzzle(initial_state: str, target_state: str):
    """
    A*算法解决八数码问题
    """
    # 1. 预处理：建立目标状态的坐标映射（避免重复计算）
    target_pos = {char: (idx // 3, idx % 3) for idx, char in enumerate(target_state)}
    
    # 2. 可解性预判：不可解直接返回
    if not is_solvable(initial_state, target_state):
        return None
    
    # 3. 初始化：优先级队列、代价字典、移动方向
    # 优先级队列元素：(f值, g值, 当前状态, 移动步数)，heapq默认按第一个元素（f值）升序排序
    heap = []
    # 初始状态：f = g(0) + h(初始状态)，g=0，步数=0
    initial_h = manhattan_distance(initial_state, target_pos)
    heapq.heappush(heap, (initial_h, 0, initial_state, 0))
    
    # 代价字典：key=状态字符串，value=该状态的最小g值（避免重复处理高代价状态）
    g_dict = {initial_state: 0}
    
    # 移动方向：上、下、左、右（对应3x3矩阵的坐标偏移）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 4. 循环扩展状态
    while heap:
        # 弹出f值最小的状态（优先级队列特性）
        current_f, current_g, current_state, current_step = heapq.heappop(heap)
        
        # 目标判断：找到目标状态，返回当前步数（A*保证最优）
        if current_state == target_state:
            return current_step
        
        # 无效状态跳过：若当前g值大于已记录的最小g值，说明该状态已被更优路径处理过
        if current_g > g_dict.get(current_state, float('inf')):
            continue
        
        # 找到当前状态中'x'的位置（一维索引和3x3坐标）
        x_idx = current_state.index('x')
        x_row = x_idx // 3
        x_col = x_idx % 3
        
        # 遍历4个移动方向，生成合法子状态
        for dx, dy in directions:
            # 计算移动后'x'的新坐标
            new_row = x_row + dx
            new_col = x_col + dy
            # 检查新坐标是否在3x3矩阵内（0<=行/列<3，避免越界）
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # 转换新坐标为一维索引
                new_x_idx = new_row * 3 + new_col
                # 交换'x'与相邻数字（字符串不可变，先转列表再交换）
                state_list = list(current_state)
                state_list[x_idx], state_list[new_x_idx] = state_list[new_x_idx], state_list[x_idx]
                next_state = ''.join(state_list)  # 子状态（转回字符串）
                
                # 计算子状态的g值（当前g+1，每移动一步代价+1）
                next_g = current_g + 1
                # 若子状态未记录，或新g值更小（找到更优路径），则更新并入队
                if next_state not in g_dict or next_g < g_dict[next_state]:
                    g_dict[next_state] = next_g  # 更新最小g值
                    next_h = manhattan_distance(next_state, target_pos)  # 计算子状态的h值
                    next_f = next_g + next_h  # 计算子状态的f值
                    heapq.heappush(heap, (next_f, next_g, next_state, current_step + 1))
    


if __name__ == "__main__":
    TARGET = "12345678x"
    input_str = input()
    initial_state = input_str.replace(" ", "")  # 去除空格，转为纯字符串（如"23415x768"）
    
    t0 = time.perf_counter()
    result = a_star_8puzzle(initial_state, TARGET)
    t1 = time.perf_counter()
    print(f"{t1 - t0:.6f}")
    print(result)