import time

def calculate_inversion(state: str) -> int:
    """计算逆序数"""
    nums = [int(c) for c in state if c != 'x']
    return sum(1 for i in range(len(nums)) for j in range(i+1, len(nums)) if nums[i] > nums[j])


def is_solvable(initial: str, target: str) -> bool:
    """判断初始状态是否可解）"""
    return calculate_inversion(initial) % 2 == calculate_inversion(target) % 2


def dfs_8puzzle(initial_state: str, target_state: str, max_depth: int = 50):
    """
    DFS求解八数码问题
    """
    # 初始化栈：(当前状态, 当前步数)，栈顶为最新状态
    stack = [(initial_state, 0)]
    # 记录已访问状态（避免重复探索）
    visited = {initial_state}
    # 移动方向：上、下、左、右（坐标偏移）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        current_state, current_step = stack.pop()  # 弹出栈顶状态（深度优先）
        
        # 找到目标状态，返回当前步数（不保证最优）
        if current_state == target_state:
            return current_step
        
        # 超过最大深度，停止该分支探索
        if current_step >= max_depth:
            continue
        
        # 找到'x'的位置（一维索引转3x3坐标）
        x_idx = current_state.index('x')
        x_row, x_col = x_idx // 3, x_idx % 3
        
        # 生成所有合法子状态（按方向顺序入栈，逆序保证探索顺序）
        for dx, dy in reversed(directions):  # 逆序入栈，保证正序探索
            new_row, new_col = x_row + dx, x_col + dy
            # 检查新坐标是否在3x3范围内
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_x_idx = new_row * 3 + new_col
                # 交换'x'与相邻位置，生成子状态
                state_list = list(current_state)
                state_list[x_idx], state_list[new_x_idx] = state_list[new_x_idx], state_list[x_idx]
                next_state = ''.join(state_list)
                
                # 未访问过的状态入栈
                if next_state not in visited:
                    visited.add(next_state)
                    stack.append((next_state, current_step + 1))
    
    # 栈空或超深度仍未找到解
    return None

if __name__ == "__main__":
    TARGET = "12345678x"
    input_str = input()
    initial = input_str.replace(" ", "")
    
    # 先判断可解性
    if not is_solvable(initial, TARGET):
        print("该状态不可解")
    else:
        # 设置最大深度为50（可根据需要调整）
        t0 = time.perf_counter()
        result = dfs_8puzzle(initial, TARGET, max_depth=50)
        t1 = time.perf_counter()
        if result is not None:
            print(f"{t1 - t0:.6f}")
            print(result)
        else:
            print(f"在最大深度{50}内未找到解")
