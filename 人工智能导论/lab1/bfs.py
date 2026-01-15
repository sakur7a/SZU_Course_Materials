from collections import deque
import time

def bfs(start : str) -> int:
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    target = "12345678x"
    q = deque([start])
    d = {start : 0}

    while q:
        t = q.popleft()

        distance = d[t]
        if t == target:
            return distance
        
        k = t.find('x')
        x, y = k // 3, k % 3
        for i in range(4):
            X, Y = x + dx[i], y + dy[i]
            if 0 <= X < 3 and 0 <= Y < 3:
                pos = X * 3 + Y
                cur = list(t)
                cur[pos], cur[k] = cur[k], cur[pos]
                new = ''.join(cur)
                if new not in d:
                    d[new] = distance + 1
                    q.append(new)
    return -1

start = ''.join(input().strip().split())
t0 = time.perf_counter()
ans = bfs(start)
t1 = time.perf_counter()
print(f"{t1 - t0:.6f}")



    