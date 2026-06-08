"""
A*-Greedy 全覆盖路径规划算法

基本思路：
1. 使用标准 A* 算法（Manhattan 启发式）计算当前位置到最近未覆盖可达栅格的最短路径
2. 沿最短路径移动，标记途经栅格为已覆盖
3. 重复直至所有可达自由栅格完成覆盖

适用于简单环境，计算效率较高，但路径通常较长且转弯较多。
"""

import heapq
import time
import numpy as np
from typing import List, Tuple, Optional, Set

# 四连通动作空间
ACTIONS_4CONN = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class AstarGreedyCoverage:
    """
    A*-Greedy 覆盖规划器

    使用 A* 反复搜索最近未覆盖栅格，实现全覆盖路径规划。
    """

    def __init__(self):
        self.planner_name = "A*-Greedy"

    def plan(
        self,
        grid: np.ndarray,
        start: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """
        执行 A*-Greedy 全覆盖路径规划。

        Args:
            grid:  占据栅格地图，0=free, 1=obstacle
            start: 起点坐标 (row, col)

        Returns:
            path:    覆盖路径（栅格坐标序列）
            covered: 覆盖标记矩阵 (bool)，True 表示已覆盖
            runtime: 规划耗时（秒）
        """
        t_start = time.time()
        height, width = grid.shape
        free_mask = (grid == 0)

        # 检查起点有效性
        if not free_mask[start]:
            raise ValueError(f"起点 {start} 位于障碍物上")

        # 连通域分析：找到所有可达的自由栅格
        reachable = self._flood_fill(grid, start)

        # 状态初始化
        covered = np.zeros((height, width), dtype=bool)
        path = [start]
        current = start
        covered[start] = True

        # 主循环：反复寻找最近未覆盖栅格
        while True:
            # 找出当前所有未覆盖的可达栅格
            uncovered = np.where(reachable & ~covered)
            uncovered_list = list(zip(uncovered[0], uncovered[1]))

            if not uncovered_list:
                break  # 所有可达栅格均已覆盖

            # 使用 A* 找到最近未覆盖栅格
            subpath, success, _ = self._astar_to_nearest(
                grid, current, uncovered_list, covered
            )

            if not success:
                # 无法到达任何未覆盖栅格（可能被障碍物隔断）
                break

            # 将子路径加入主路径（跳过起点避免重复）
            for pos in subpath[1:]:
                path.append(pos)
                if free_mask[pos]:
                    covered[pos] = True

            # 更新当前位置
            current = subpath[-1]

        t_end = time.time()
        runtime = t_end - t_start

        return path, covered, runtime

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _flood_fill(
        self, grid: np.ndarray, start: Tuple[int, int]
    ) -> np.ndarray:
        """
        BFS 连通域分析 —— 找出从起点可达的所有自由栅格。

        Args:
            grid:  占据栅格地图
            start: 起点坐标

        Returns:
            bool 矩阵，True 表示从起点可达的自由栅格
        """
        height, width = grid.shape
        reachable = np.zeros((height, width), dtype=bool)
        if grid[start] != 0:
            return reachable

        queue = [start]
        reachable[start] = True

        while queue:
            y, x = queue.pop(0)
            for dy, dx in ACTIONS_4CONN:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if not reachable[ny, nx] and grid[ny, nx] == 0:
                        reachable[ny, nx] = True
                        queue.append((ny, nx))

        return reachable

    def _astar_to_nearest(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
        covered: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], bool, float]:
        """
        使用 A* 查找从 start 到最近未覆盖目标栅格的最短路径。
        启发式函数采用 Manhattan 距离。

        Args:
            grid:    占据栅格地图
            start:   起点坐标
            targets: 候选目标列表（未覆盖栅格）
            covered: 当前覆盖状态

        Returns:
            subpath:  最短路径（含起点和终点）
            success:  是否找到有效路径
            cost:     路径代价
        """
        height, width = grid.shape
        target_set = set(targets)

        # A* 开放集和关闭集
        # 格式：(f_score, g_score, y, x, path_so_far)
        open_set = []
        g_scores = {start: 0}

        # 计算起点到所有目标的最小 Manhattan 距离作为启发式
        h_start = min(
            abs(start[0] - ty) + abs(start[1] - tx)
            for ty, tx in targets
        )
        heapq.heappush(open_set, (h_start, 0, start[0], start[1], [start]))
        closed_set = set()

        while open_set:
            f, g, y, x, path_so_far = heapq.heappop(open_set)

            if (y, x) in closed_set:
                continue
            closed_set.add((y, x))

            # 检查是否到达目标
            if (y, x) in target_set and not covered[y, x]:
                return path_so_far, True, g

            # 扩展邻居
            for dy, dx in ACTIONS_4CONN:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width
                        and grid[ny, nx] == 0):
                    if (ny, nx) in closed_set:
                        continue

                    new_g = g + 1

                    if (ny, nx) not in g_scores or new_g < g_scores[(ny, nx)]:
                        g_scores[(ny, nx)] = new_g
                        # 计算到最近目标的 Manhattan 距离
                        h = min(
                            abs(ny - ty) + abs(nx - tx)
                            for ty, tx in targets
                        )
                        new_f = new_g + h
                        new_path = list(path_so_far)
                        new_path.append((ny, nx))
                        heapq.heappush(
                            open_set, (new_f, new_g, ny, nx, new_path)
                        )

        # 未找到可用目标
        return [start], False, float('inf')
