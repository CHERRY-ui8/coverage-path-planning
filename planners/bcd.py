"""
Boustrophedon Cellular Decomposition (BCD) 全覆盖路径规划算法

参考论文：
  Choset, H. & Pignon, P. (1998). "Coverage Path Planning: The Boustrophedon
  Cellular Decomposition"

核心思想：
  1. 使用一条从左到右扫过的垂直线（sweep line）对自由空间进行分解
  2. 在临界点（critical point）处检测自由空间连通性的变化，划分单元（cells）
  3. 每个单元内部采用往复式（boustrophedon/lawnmower）覆盖路径
  4. 构建单元邻接图，使用图搜索确定单元遍历顺序

临界点检测（Choset 1998）：
  - IN 事件：   自由区间数量增加（一个新区域出现或单元分裂）
  - OUT 事件：  自由区间数量减少（区域合并）
  - MIDDLE 事件：保持相同区间数但结构变化
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

# 四连通动作空间
ACTIONS_4CONN = [(0, 1), (1, 0), (0, -1), (-1, 0)]


@dataclass
class Cell:
    """
    BCD 分解中的一个单元（cell）。

    每个单元是一个在 y 方向连续、在 x 方向被临界线界定的区域。
    单元内自由空间在任意 x∈[x_min, x_max] 上的截面为一个连通区间。
    """
    id: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    # 属于该单元的自由栅格集合（可选，用于精确覆盖）
    free_cells: Set[Tuple[int, int]] = field(default_factory=set)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def area(self) -> int:
        """单元面积：优先使用 free_cells 集合，回退到边界矩形估计。"""
        if self.free_cells:
            return len(self.free_cells)
        return max(0, self.x_max - self.x_min + 1) * max(0, self.y_max - self.y_min + 1)

    def __repr__(self) -> str:
        return (f"Cell(id={self.id}, "
                f"x=[{self.x_min},{self.x_max}], "
                f"y=[{self.y_min},{self.y_max}], "
                f"area={self.area})")


class BCDCoverage:
    """
    Boustrophedon Cellular Decomposition 覆盖规划器。

    实现了 Choset 1998 论文中描述的：
    - sweep-line 扫描与临界点检测
    - 自由空间单元分解
    - cell 内往复式覆盖
    - cell 间图搜索连接
    """

    def __init__(self):
        self.planner_name = "BCD"
        self.cells: List[Cell] = []
        self.adjacency: List[Set[int]] = []

    def plan(
        self,
        grid: np.ndarray,
        start: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """
        执行 BCD 全覆盖路径规划。

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
        free_mask = (grid == 0)

        if not free_mask[start]:
            raise ValueError(f"起点 {start} 位于障碍物上")

        # Step 1: 连通域分析 —— 找到从起点可达的所有自由栅格
        reachable = self._flood_fill(grid, start)

        # Step 2: BCD 单元分解
        self.cells = []
        self.adjacency = []
        self._decompose(grid, reachable)

        if not self.cells:
            # 退化情况：没有有效单元
            runtime = time.time() - t_start
            return [start], np.zeros((height, width), dtype=bool), runtime

        # Step 3: 为每个单元填充精确的自由栅格集合
        self._populate_cell_free_cells(grid, height, width, reachable)

        # Step 4: 构建单元邻接图
        self._build_adjacency()

        # Step 5: 确定单元遍历顺序（贪心最近邻）
        cell_order = self._compute_cell_order(start)

        # Step 6: 在每个单元内生成往复式路径，并连接单元间路径
        path = self._generate_coverage_path(grid, cell_order, start)

        # 构建覆盖标记矩阵
        covered = np.zeros((height, width), dtype=bool)
        for (y, x) in path:
            covered[y, x] = True

        t_end = time.time()
        runtime = t_end - t_start

        return path, covered, runtime

    # ------------------------------------------------------------------
    # Step 2: BCD 单元分解
    # ------------------------------------------------------------------

    def _decompose(
        self, grid: np.ndarray, reachable: np.ndarray
    ) -> None:
        """
                基于 Choset 1998 的 sweep-line 分解。

        核心思想：
          从左到右扫描每一列，检测自由区间的连通性变化。
          当区间在相邻列之间的连接关系发生变化时，记录临界位置。
          两个临界位置之间的区域构成一个单元。

        具体实现：
          对于每列 x，计算位于可达区间的连续自由段。
          与前一列的区间对比，根据重叠情况判断：
          - 一对一匹配 → 延续已有单元
          - 多对一匹配 → OUT 事件（合并），关闭部分单元
          - 一对多匹配 → IN 事件（分裂），创建新单元
          - 无匹配 → 开始新单元 / 结束单元
        """
        height, width = grid.shape

        # 在扫线过程中维护活动单元
        # active_intervals: List[(y_min, y_max, cell_index)]
        active_intervals: List[Tuple[int, int, int]] = []
        cells: List[Cell] = []
        cell_id_counter = 0

        for x in range(width):
            # 获取当前列的可达自由区间
            intervals = self._find_free_intervals(grid[:, x], reachable[:, x])

            if x == 0:
                # 第一列，直接创建初始单元
                for y_min, y_max in intervals:
                    cell = Cell(
                        id=cell_id_counter,
                        x_min=x, x_max=x,
                        y_min=y_min, y_max=y_max
                    )
                    cells.append(cell)
                    active_intervals.append((y_min, y_max, cell_id_counter))
                    cell_id_counter += 1
                continue

            if not intervals and not active_intervals:
                continue
            if not intervals:
                # 当前列没有自由区间 —— 所有活动单元结束
                for _, _, cid in active_intervals:
                    cells[cid].x_max = x - 1
                active_intervals = []
                continue
            if not active_intervals:
                # 前一列没有活动单元 —— 开启新单元
                for y_min, y_max in intervals:
                    cell = Cell(
                        id=cell_id_counter,
                        x_min=x, x_max=x,
                        y_min=y_min, y_max=y_max
                    )
                    cells.append(cell)
                    active_intervals.append((y_min, y_max, cell_id_counter))
                    cell_id_counter += 1
                continue

            # === 匹配当前列区间与活动单元 ===

            # 匹配矩阵：match_prev[prev_idx] = [curr_idx, ...]
            # 表示前一列第 prev_idx 个区间与当前列哪些区间重叠
            prev_count = len(active_intervals)
            curr_count = len(intervals)

            # 计算重叠：prev_i 与 curr_j 是否有重叠
            # overlap exists if interval y ranges intersect
            overlap = np.zeros((prev_count, curr_count), dtype=bool)
            for pi, (py_min, py_max, _) in enumerate(active_intervals):
                for ci, (cy_min, cy_max) in enumerate(intervals):
                    if py_min < cy_max and py_max > cy_min:
                        overlap[pi, ci] = True

            # 确定每个当前区间匹配哪个前一区间
            # curr_to_prev[ci] = pi（若唯一匹配）或 -1（无匹配）或 -2（多匹配）
            curr_to_prev = []
            for ci in range(curr_count):
                matches = np.where(overlap[:, ci])[0]
                if len(matches) == 0:
                    curr_to_prev.append(-1)  # 新区间
                elif len(matches) == 1:
                    curr_to_prev.append(int(matches[0]))
                else:
                    curr_to_prev.append(-2)  # 多匹配（合并）

            # prev_to_curr[pi] = ci（若唯一匹配）或 -1（无匹配）或 -2（多匹配）
            prev_to_curr = []
            for pi in range(prev_count):
                matches = np.where(overlap[pi, :])[0]
                if len(matches) == 0:
                    prev_to_curr.append(-1)  # 区间消失
                elif len(matches) == 1:
                    prev_to_curr.append(int(matches[0]))
                else:
                    prev_to_curr.append(-2)  # 多匹配（分裂）

            # === 构建新的活动单元列表 ===

            new_active = []
            matched_current = set()

            # 处理延续的区间（一对一匹配）
            for ci in range(curr_count):
                if curr_to_prev[ci] >= 0:
                    pi = curr_to_prev[ci]
                    if prev_to_curr[pi] >= 0:  # 确认双向一对一
                        cid = active_intervals[pi][2]
                        # 扩展单元
                        cells[cid].x_max = x
                        cells[cid].y_min = min(
                            cells[cid].y_min, intervals[ci][0]
                        )
                        cells[cid].y_max = max(
                            cells[cid].y_max, intervals[ci][1]
                        )
                        new_active.append(
                            (intervals[ci][0], intervals[ci][1], cid)
                        )
                        matched_current.add(ci)

            # 处理新区间（无匹配）
            for ci in range(curr_count):
                if ci in matched_current:
                    continue
                if curr_to_prev[ci] == -1:
                    # IN 事件：新单元开始
                    cell = Cell(
                        id=cell_id_counter,
                        x_min=x, x_max=x,
                        y_min=intervals[ci][0],
                        y_max=intervals[ci][1]
                    )
                    cells.append(cell)
                    new_active.append(
                        (intervals[ci][0], intervals[ci][1], cell_id_counter)
                    )
                    cell_id_counter += 1
                    matched_current.add(ci)

            # 处理分裂（一对多）
            for ci in range(curr_count):
                if ci in matched_current:
                    continue
                if curr_to_prev[ci] == -2:
                    # 计算这个新区间匹配哪些前一区间
                    matches = np.where(overlap[:, ci])[0]
                    # 取第一个匹配的区间作为延续
                    pi = int(matches[0])
                    cid = active_intervals[pi][2]
                    cells[cid].x_max = x
                    cells[cid].y_min = min(
                        cells[cid].y_min, intervals[ci][0]
                    )
                    cells[cid].y_max = max(
                        cells[cid].y_max, intervals[ci][1]
                    )
                    new_active.append(
                        (intervals[ci][0], intervals[ci][1], cid)
                    )
                    matched_current.add(ci)

                    # 其余匹配创建新单元
                    for extra_pi in matches[1:]:
                        e_cid = active_intervals[int(extra_pi)][2]
                        # OUT 事件：结束另一边匹配的单元
                        cells[e_cid].x_max = x - 1
                    # 延用第一个匹配的 cell ID
                    # 其余匹配的区间视为消失（已经在前面处理了）

            # 处理消失的区间
            for pi in range(prev_count):
                if prev_to_curr[pi] < 0:
                    # OUT 事件：单元结束
                    cid = active_intervals[pi][2]
                    cells[cid].x_max = x - 1

            # 清理：当前列未匹配的区间（如果上面有遗漏）
            for ci in range(curr_count):
                if ci not in matched_current:
                    y_min, y_max = intervals[ci]
                    cell = Cell(
                        id=cell_id_counter,
                        x_min=x, x_max=x,
                        y_min=y_min, y_max=y_max
                    )
                    cells.append(cell)
                    new_active.append(
                        (intervals[ci][0], intervals[ci][1], cell_id_counter)
                    )
                    cell_id_counter += 1

            active_intervals = new_active

        # 关闭所有剩余活动单元
        for _, _, cid in active_intervals:
            cells[cid].x_max = width - 1

        # 过滤无效单元（面积为零）
        valid_cells = [c for c in cells if c.area > 0 and c.x_max >= c.x_min]

        # 重编号
        self.cells = []
        for i, c in enumerate(valid_cells):
            c.id = i
            self.cells.append(c)

    def _find_free_intervals(
        self, col: np.ndarray, reachable_col: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        查找某列中同时满足自由且可达的连续区间。

        Args:
            col:          该列的占据数据（一维）
            reachable_col: 该列的可达标记

        Returns:
            [(y_start, y_end), ...]，区间为半开 [start, end)
        """
        intervals = []
        y = 0
        while y < len(col):
            if col[y] == 0 and reachable_col[y]:
                y_start = y
                while (y < len(col) and col[y] == 0
                       and reachable_col[y]):
                    y += 1
                y_end = y
                intervals.append((y_start, y_end))
            else:
                y += 1
        return intervals

    # ------------------------------------------------------------------
    # Step 3: 为每个单元填充精确的自由栅格集合
    # ------------------------------------------------------------------

    def _populate_cell_free_cells(
        self, grid: np.ndarray, height: int, width: int,
        reachable: np.ndarray
    ) -> None:
        """为每个 cell 填充属于它的自由栅格坐标集合（带边界检查）。"""
        for cell in self.cells:
            y0 = max(0, cell.y_min)
            y1 = min(height, cell.y_max + 1)
            x0 = max(0, cell.x_min)
            x1 = min(width, cell.x_max + 1)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if grid[y, x] == 0 and reachable[y, x]:
                        cell.free_cells.add((y, x))

    # ------------------------------------------------------------------
    # Step 4: 构建单元邻接图
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> None:
        """
        构建单元邻接图。
        两个单元在某列上左右相邻，且 y 方向有重叠区间，则视为邻接。
        """
        n = len(self.cells)
        self.adjacency = [set() for _ in range(n)]

        for i, cell_a in enumerate(self.cells):
            for j, cell_b in enumerate(self.cells):
                if i >= j:
                    continue
                # 检查两个 cell 是否相邻
                # 条件：一个的右边界与另一个的左边界相邻，且 y 区间重叠
                x_adjacent = (
                    cell_a.x_max + 1 == cell_b.x_min
                    or cell_b.x_max + 1 == cell_a.x_min
                )
                if x_adjacent:
                    y_overlap = (
                        cell_a.y_min < cell_b.y_max
                        and cell_b.y_min < cell_a.y_max
                    )
                    if y_overlap:
                        self.adjacency[i].add(j)
                        self.adjacency[j].add(i)

    # ------------------------------------------------------------------
    # Step 5: 计算单元遍历顺序（贪心最近邻）
    # ------------------------------------------------------------------

    def _compute_cell_order(
        self, start: Tuple[int, int]
    ) -> List[int]:
        """
        确定单元遍历顺序。
        使用贪心最近邻策略，每次选择距离当前位置最近的未访问邻接单元。

        Args:
            start: 起点坐标

        Returns:
            单元 ID 列表 [id_1, id_2, ...]
        """
        if not self.cells:
            return []

        n = len(self.cells)

        # 找到起点所在单元
        start_cell = None
        for cell in self.cells:
            if start in cell.free_cells:
                start_cell = cell.id
                break

        if start_cell is None:
            # 起点不在任何单元中，选择最近的单元
            min_dist = float('inf')
            for cell in self.cells:
                if cell.free_cells:
                    dist = min(
                        abs(start[0] - fy) + abs(start[1] - fx)
                        for fy, fx in cell.free_cells
                    )
                    if dist < min_dist:
                        min_dist = dist
                        start_cell = cell.id

        # 贪心最近邻遍历
        visited = {start_cell}
        order = [start_cell]
        current = start_cell

        while len(visited) < n:
            # 从当前单元的所有邻接未访问单元中选择最近（基于两个单元质心距离）
            best_next = None
            best_dist = float('inf')

            for neighbor in self.adjacency[current]:
                if neighbor in visited:
                    continue
                # 计算质心距离
                cx = np.mean([
                    fy for fy, fx in self.cells[current].free_cells
                ])
                cy = np.mean([
                    fx for fy, fx in self.cells[current].free_cells
                ])
                nx = np.mean([
                    fy for fy, fx in self.cells[neighbor].free_cells
                ])
                ny = np.mean([
                    fx for fy, fx in self.cells[neighbor].free_cells
                ])
                dist = abs(cx - nx) + abs(cy - ny)
                if dist < best_dist:
                    best_dist = dist
                    best_next = neighbor

            if best_next is None:
                # 没有邻接未访问单元，使用 BFS 找最近可达
                best_next = self._find_nearest_unvisited(
                    current, visited
                )
                if best_next is None:
                    break

            visited.add(best_next)
            order.append(best_next)
            current = best_next

        return order

    def _find_nearest_unvisited(
        self, current_idx: int, visited: Set[int]
    ) -> Optional[int]:
        """使用 BFS 在邻接图上找到最近的未访问单元。"""
        if not self.adjacency:
            return None

        queue = [current_idx]
        seen = {current_idx}

        while queue:
            node = queue.pop(0)
            for neighbor in self.adjacency[node]:
                if neighbor not in seen:
                    if neighbor not in visited:
                        return neighbor
                    seen.add(neighbor)
                    queue.append(neighbor)

        return None

    # ------------------------------------------------------------------
    # Step 6: 生成全覆盖路径
    # ------------------------------------------------------------------

    def _generate_coverage_path(
        self,
        grid: np.ndarray,
        cell_order: List[int],
        start: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        生成最终覆盖路径：
        1. 每个单元内生成往复式（boustrophedon）覆盖路径
        2. 使用 A* 连接相邻单元的路径

        Args:
            grid:       占据栅格地图
            cell_order: 单元遍历顺序
            start:      起点坐标

        Returns:
            完整的覆盖路径
        """
        if not cell_order:
            return [start]

        height, width = grid.shape
        full_path = []

        # 确定起点位置：从起点所在单元开始
        # 如果起点不在第一个单元，连接过去
        current_pos = start

        for i, cell_id in enumerate(cell_order):
            cell = self.cells[cell_id]

            # 在当前单元生成往复式路径
            cell_path = self._boustrophedon_path(grid, cell)

            if not cell_path:
                continue

            if i == 0:
                # 第一个单元：从起点到单元路径起点
                if current_pos != cell_path[0]:
                    connect_path = self._astar_connect(
                        grid, current_pos, cell_path[0]
                    )
                    if connect_path:
                        full_path.extend(connect_path[:-1])
                full_path.append(cell_path[0])
                full_path.extend(cell_path[1:])
            else:
                # 连接前一路径末端到当前单元路径起点
                connect_path = self._astar_connect(
                    grid, current_pos, cell_path[0]
                )
                if connect_path:
                    full_path.extend(connect_path[:-1])
                full_path.append(cell_path[0])
                full_path.extend(cell_path[1:])

            current_pos = full_path[-1]

        # ── post-process: fill any remaining jumps with A* paths ──
        full_path = self._smooth_path(grid, full_path)

        return full_path

    def _smooth_path(
        self, grid: np.ndarray, raw_path: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        用 A* 短路径替换原始路径中的非相邻跳跃。
        确保完整路径中任意连续两步均 4-连通相邻。
        """
        if len(raw_path) < 2:
            return raw_path
        smoothed = [raw_path[0]]
        for i in range(1, len(raw_path)):
            prev = smoothed[-1]
            curr = raw_path[i]
            d = abs(prev[0] - curr[0]) + abs(prev[1] - curr[1])
            if d <= 1:
                if prev != curr:
                    smoothed.append(curr)
            else:
                link = self._astar_connect(grid, prev, curr)
                if link:
                    for p in link[1:]:
                        if p != smoothed[-1]:
                            smoothed.append(p)
                else:
                    smoothed.append(curr)
        return smoothed

    def _boustrophedon_path(
        self, grid: np.ndarray, cell: Cell
    ) -> List[Tuple[int, int]]:
        """
        在单元内生成往复式（boustrophedon / lawnmower）覆盖路径。

        路径从单元左下角或左上角开始，垂直往复扫描：
        - 第一遍：从 y_min 到 y_max
        - 然后右移一列
        - 第二遍：从 y_max 到 y_min
        - 重复直到到达右边界

        Args:
            grid: 占据栅格地图
            cell: 待覆盖单元

        Returns:
            单元内的覆盖路径
        """
        path = []
        if cell.x_min > cell.x_max or cell.y_min > cell.y_max:
            return path

        # 剔除障碍物后的有效 x 坐标
        valid_xs = []
        for x in range(cell.x_min, cell.x_max + 1):
            # 检查该列在单元 y 范围内是否有自由栅格
            has_free = any(
                grid[y, x] == 0
                for y in range(cell.y_min, cell.y_max + 1)
                if 0 <= y < grid.shape[0]
            )
            if has_free:
                valid_xs.append(x)

        if not valid_xs:
            return path

        # 垂直往复扫描
        for sweep_idx, x in enumerate(valid_xs):
            # 在单元 y 范围内找到该列的自由栅格
            col_free = [
                y for y in range(cell.y_min, cell.y_max + 1)
                if 0 <= y < grid.shape[0] and grid[y, x] == 0
                and (y, x) in cell.free_cells
            ]

            if not col_free:
                continue

            # 根据扫描轮次决定方向（往复）
            if sweep_idx % 2 == 0:
                # 偶数次：从上到下
                col_free.sort()
            else:
                # 奇数次：从下到上
                col_free.sort(reverse=True)

            for y in col_free:
                if not path or path[-1] != (y, x):
                    path.append((y, x))

        return path

    def _astar_connect(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        使用 A* 计算两个位置之间的路径（用于连接单元）。

        Args:
            grid:  占据栅格地图
            start: 起点
            goal:  终点

        Returns:
            从 start 到 goal 的最短路径（含两端）
        """
        if start == goal:
            return [start]

        height, width = grid.shape
        open_set = []
        g_scores = {start: 0.0}
        came_from = {}

        # Manhattan 启发式
        h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        heapq.heappush(
            open_set, (h, 0, start[0], start[1])
        )
        closed_set = set()

        while open_set:
            f, g, y, x = heapq.heappop(open_set)

            if (y, x) == goal:
                # 回溯路径
                path = []
                current = (y, x)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            if (y, x) in closed_set:
                continue
            closed_set.add((y, x))

            for dy, dx in ACTIONS_4CONN:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width
                        and grid[ny, nx] == 0):
                    if (ny, nx) in closed_set:
                        continue
                    new_g = g + 1
                    if ((ny, nx) not in g_scores
                            or new_g < g_scores[(ny, nx)]):
                        g_scores[(ny, nx)] = new_g
                        h_new = abs(ny - goal[0]) + abs(nx - goal[1])
                        heapq.heappush(
                            open_set, (new_g + h_new, new_g, ny, nx)
                        )
                        came_from[(ny, nx)] = (y, x)

        return []  # 无路径

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


# 导入 heapq（供 _astar_connect 使用）
import heapq
