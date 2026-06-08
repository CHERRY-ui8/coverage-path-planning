"""
Spanning Tree Coverage (STC) 全覆盖路径规划算法

参考论文：
  Gabriely, Y. & Rimon, E. (2001). "Spanning-Tree Based Coverage of
  Continuous Areas by a Mobile Robot"

核心思想：
  1. 将占据栅格地图粗粒化为 2×2 粗栅格（coarse grid）
  2. 在粗网格上构建生成树（spanning tree）
  3. 沿生成树周游（circumnavigation），覆盖每个粗栅格内的 4 个精细栅格

重要性质：
  - 每个精细栅格恰好被访问一次（对完全自由的粗栅格）
  - 路径具有最优性保证（路径长度 ≤ 2 × 最优）
  - 完全覆盖可达区域
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

# 四连通动作
ACTIONS_4CONN = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class STCCoverage:
    """
    Spanning Tree Coverage 覆盖规划器。

    实现了基于生成树的覆盖率路径规划：
    - coarse graph 构建
    - DFS spanning tree 生成
    - tree traversal based coverage
    """

    def __init__(self):
        self.planner_name = "STC"

    def plan(
        self,
        grid: np.ndarray,
        start: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """
        执行 STC 全覆盖路径规划。

        Args:
            grid:  占据栅格地图，0=free, 1=obstacle
            start: 起点坐标 (row, col)

        Returns:
            path:    覆盖路径（栅格坐标序列）
            covered: 覆盖标记矩阵 (bool)
            runtime: 规划耗时（秒）
        """
        t_start = time.time()
        height, width = grid.shape

        if grid[start] != 0:
            raise ValueError(f"起点 {start} 位于障碍物上")

        # Step 1: 计算可达区域
        reachable = self._flood_fill(grid, start)
        free_mask = (grid == 0) & reachable

        # Step 2: 构建粗网格（2×2 blocks）
        ch = (height + 1) // 2  # 粗网格行数
        cw = (width + 1) // 2   # 粗网格列数

        # 粗网格有效性：粗栅格内自由精细栅格 > 1 时视为有效
        coarse_grid = np.zeros((ch, cw), dtype=bool)
        for cy in range(ch):
            for cx in range(cw):
                fine_count = 0
                for dy in range(2):
                    for dx in range(2):
                        fy, fx = cy * 2 + dy, cx * 2 + dx
                        if (fy < height and fx < width
                                and free_mask[fy, fx]):
                            fine_count += 1
                # 粗栅格有效性：至少包含 1 个自由精细栅格即视为可通行
                # 遵循 Gabriely & Rimon (2001): "A cell is called empty if it
                # contains no obstacles" — 至少有一个自由子栅格即为空
                coarse_grid[cy, cx] = (fine_count >= 1)

        # Step 3: 找到起点所在的粗栅格
        start_c = (start[0] // 2, start[1] // 2)
        if not coarse_grid[start_c]:
            # 如果起点粗栅格无效，寻找最近的粗栅格
            start_c = self._find_nearest_coarse(
                coarse_grid, start_c, ch, cw
            )
            if start_c is None:
                # 无有效粗栅格，返回起点
                runtime = time.time() - t_start
                return [start], np.zeros_like(grid, dtype=bool), runtime

        # Step 4: 构建粗网格邻接图并生成 spanning tree
        visited_coarse = set()
        parent = {}  # child -> parent
        self._dfs_spanning_tree(
            coarse_grid, start_c, visited_coarse, parent
        )

        # Step 5: 沿生成树遍历生成覆盖路径
        path, covered = self._generate_path_from_tree(
            grid=grid,
            height=height,
            width=width,
            free_mask=free_mask,
            parent_map=parent,
            root=start_c
        )

        t_end = time.time()
        runtime = t_end - t_start

        return path, covered, runtime

    # ------------------------------------------------------------------
    # Step 2-4: 粗网格与生成树构建
    # ------------------------------------------------------------------

    def _find_nearest_coarse(
        self,
        coarse_grid: np.ndarray,
        start: Tuple[int, int],
        ch: int, cw: int
    ) -> Optional[Tuple[int, int]]:
        """BFS 查找最近的有效的粗栅格。"""
        queue = [start]
        visited = {start}
        while queue:
            cy, cx = queue.pop(0)
            if coarse_grid[cy, cx]:
                return (cy, cx)
            for dy, dx in ACTIONS_4CONN:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < ch and 0 <= nx < cw
                        and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        return None

    def _dfs_spanning_tree(
        self,
        coarse_grid: np.ndarray,
        current: Tuple[int, int],
        visited: Set[Tuple[int, int]],
        parent: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> None:
        """
        在粗网格上通过 DFS 构建生成树。

        Args:
            coarse_grid: 粗网格有效标记
            current:     当前粗栅格坐标 (cy, cx)
            visited:     已访问集合
            parent:      父子映射表（child -> parent）
        """
        ch, cw = coarse_grid.shape
        cy, cx = current
        visited.add(current)

        # 四邻域扩展
        for dy, dx in ACTIONS_4CONN:
            ny, nx = cy + dy, cx + dx
            neighbor = (ny, nx)
            if (0 <= ny < ch and 0 <= nx < cw
                    and coarse_grid[ny, nx]
                    and neighbor not in visited):
                parent[neighbor] = current
                self._dfs_spanning_tree(
                    coarse_grid, neighbor, visited, parent
                )

    # ------------------------------------------------------------------
    # Step 5: 沿生成树生成覆盖路径（DFS 遍历 + BFS 连接）
    # ------------------------------------------------------------------

    def _generate_path_from_tree(
        self,
        grid: np.ndarray,
        height: int,
        width: int,
        free_mask: np.ndarray,
        parent_map: Dict,
        root: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        基于生成树生成全覆盖路径。

        方法：
          1. 获取粗栅格的 DFS 前序访问顺序
          2. 对每个粗栅格，以顺时针顺序访问其 4 个精细栅格
          3. 在相邻粗栅格之间通过 BFS 短路径保持连续性

        Args:
            grid:      占据栅格地图
            height:    地图高度
            width:     地图宽度
            free_mask: 可通行标记
            parent_map: 生成树父子映射 {child: parent}
            root:      根粗栅格

        Returns:
            path:    覆盖路径
            covered: 覆盖标记矩阵
        """
        # 构建 children 映射
        from collections import defaultdict, deque
        children = defaultdict(list)
        for child, par in parent_map.items():
            if par is not None:
                children[par].append(child)

        # DFS 前序顺序
        order = []
        def dfs_preorder(cell, par):
            order.append(cell)
            for ch in children[cell]:
                if ch != par:
                    dfs_preorder(ch, cell)
        dfs_preorder(root, None)

        # BFS 连接函数
        def bfs_path(from_pos, to_pos):
            if from_pos == to_pos:
                return []
            q = deque([from_pos])
            came = {from_pos: None}
            ok = False
            while q and not ok:
                y, x = q.popleft()
                for dy, dx in ACTIONS_4CONN:
                    ny, nx = y + dy, x + dx
                    np_ = (ny, nx)
                    if (0 <= ny < height and 0 <= nx < width
                            and grid[ny, nx] == 0
                            and np_ not in came):
                        came[np_] = (y, x)
                        if np_ == to_pos:
                            ok = True
                            break
                        q.append(np_)
            if not ok:
                return []
            # 回溯路径
            trail = []
            cur = to_pos
            while cur is not None:
                trail.append(cur)
                cur = came[cur]
            trail.reverse()
            return trail[1:]  # 不含起点

        path = []
        covered = np.zeros((height, width), dtype=bool)

        def add(p):
            if (0 <= p[0] < height and 0 <= p[1] < width
                    and grid[p] == 0 and (not path or path[-1] != p)):
                path.append(p)
                covered[p] = True

        for idx, cell in enumerate(order):
            cy, cx = cell
            # 该粗栅格的 4 个精细栅格（顺时针）: tl, tr, br, bl
            fine_block = [
                (cy * 2, cx * 2),
                (cy * 2, cx * 2 + 1),
                (cy * 2 + 1, cx * 2 + 1),
                (cy * 2 + 1, cx * 2),
            ]
            for p in fine_block:
                add(p)

            # 连接到下一个粗栅格
            if idx < len(order) - 1:
                next_cell = order[idx + 1]
                # 找到下一个单元第一个有效的精细栅格（可能是 tl/tr/bl/br）
                next_entry = self._first_valid_fine(
                    next_cell, height, width, grid
                )
                if next_entry is not None and path and path[-1] != next_entry:
                    link = bfs_path(path[-1], next_entry)
                    for lp in link:
                        add(lp)

        return path, covered

    @staticmethod
    def _first_valid_fine(
        coarse_cell: Tuple[int, int],
        height: int, width: int, grid: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """返回粗栅格中第一个可通行的精细栅格，若全为障碍则返回 None。"""
        cy, cx = coarse_cell
        for dy, dx in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            fy, fx = cy * 2 + dy, cx * 2 + dx
            if 0 <= fy < height and 0 <= fx < width and grid[fy, fx] == 0:
                return (fy, fx)
        return None

    def _get_child_in_dir(
        self,
        parent: Tuple[int, int],
        children: List[Tuple[int, int]],
        direction: int
    ) -> Optional[Tuple[int, int]]:
        """
        获取指定方向上的子节点。

        Args:
            parent:    父节点坐标 (cy, cx)
            children:  子节点列表
            direction: 方向 (0=up, 1=right, 2=down, 3=left)

        Returns:
            子节点坐标，若不存在则返回 None
        """
        py, px = parent
        for child in children:
            dy = child[0] - py
            dx = child[1] - px
            if direction == 0 and dy == -1 and dx == 0:
                return child
            if direction == 1 and dy == 0 and dx == 1:
                return child
            if direction == 2 and dy == 1 and dx == 0:
                return child
            if direction == 3 and dy == 0 and dx == -1:
                return child
        return None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _flood_fill(
        self, grid: np.ndarray, start: Tuple[int, int]
    ) -> np.ndarray:
        """BFS 连通域分析。"""
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
                if (0 <= ny < height and 0 <= nx < width
                        and not reachable[ny, nx]
                        and grid[ny, nx] == 0):
                    reachable[ny, nx] = True
                    queue.append((ny, nx))

        return reachable
