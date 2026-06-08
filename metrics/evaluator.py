"""
评价指标计算模块

计算以下覆盖路径规划评价指标：
1. Coverage Rate （覆盖率）
2. Path Length （路径长度）
3. Coverage Efficiency （覆盖效率）
4. Number of Turns （转弯次数）
5. Runtime （运行时间）
"""

import numpy as np
import math
from typing import List, Tuple, Dict


class MetricsEvaluator:
    """
    覆盖路径规划评价指标计算器。

    对给定的地图、覆盖路径和规划结果进行多维度评价。
    """

    def __init__(self):
        self.metrics_keys = [
            'coverage_rate',
            'path_length',
            'coverage_efficiency',
            'num_turns',
            'turn_rate',
            'runtime',
            'free_cells_total',
            'free_cells_covered',
            'num_revisits',
            'revisit_rate',
        ]

    def evaluate(
        self,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        runtime: float,
        start: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        计算全覆盖路径规划的各项评价指标。

        Args:
            grid:    占据栅格地图，0=free, 1=obstacle
            path:    规划器输出的路径（栅格坐标序列）
            covered: 覆盖标记矩阵 (bool)
            runtime: 规划耗时（秒）
            start:   起点坐标

        Returns:
            {metric_name: value}
        """
        height, width = grid.shape
        free_mask = (grid == 0)

        # 连通域分析：仅考虑可达的自由栅格
        reachable = self._flood_fill(grid, start)
        reachable_free = free_mask & reachable

        total_free = int(np.sum(reachable_free))
        covered_free = int(np.sum(covered & reachable_free))

        # 1. 覆盖率
        coverage_rate = (
            covered_free / total_free if total_free > 0 else 0.0
        )

        # 2. 路径长度
        path_length = len(path)

        # 3. 覆盖效率（每步覆盖的栅格数）
        #    衡量路径效率：值越大越好
        coverage_efficiency = (
            covered_free / path_length if path_length > 0 else 0.0
        )

        # 4. 转弯次数
        num_turns, turn_rate = self._count_turns(path)

        # 5. 重复访问统计
        num_revisits = 0
        visited_set = set()
        for pos in path:
            if pos in visited_set:
                num_revisits += 1
            visited_set.add(pos)

        revisit_rate = (
            num_revisits / path_length if path_length > 0 else 0.0
        )

        return {
            'coverage_rate':       round(coverage_rate, 4),
            'path_length':         path_length,
            'coverage_efficiency': round(coverage_efficiency, 4),
            'num_turns':           num_turns,
            'turn_rate':           round(turn_rate, 4),
            'runtime':             round(runtime, 4),
            'free_cells_total':    total_free,
            'free_cells_covered':  covered_free,
            'num_revisits':        num_revisits,
            'revisit_rate':        round(revisit_rate, 4),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _count_turns(
        path: List[Tuple[int, int]]
    ) -> Tuple[int, float]:
        """
        统计路径中的转弯次数。

        转弯定义为连续两个运动方向之间的夹角为 90°。
        当路径长度小于 2 时，转弯次数为 0。

        Args:
            path: 路径坐标序列 [(y1, x1), (y2, x2), ...]

        Returns:
            (num_turns, turn_rate): 转弯次数和转弯率（每步转弯次数）
        """
        if len(path) < 3:
            return 0, 0.0

        turns = 0
        for i in range(2, len(path)):
            # 计算上一步和这一步的运动方向
            dy1 = path[i - 1][0] - path[i - 2][0]
            dx1 = path[i - 1][1] - path[i - 2][1]
            dy2 = path[i][0] - path[i - 1][0]
            dx2 = path[i][1] - path[i - 1][1]

            # 判断方向是否改变
            if (dy1, dx1) != (dy2, dx2):
                turns += 1

        turn_rate = turns / len(path) if len(path) > 0 else 0.0
        return turns, turn_rate

    @staticmethod
    def _flood_fill(
        grid: np.ndarray, start: Tuple[int, int]
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
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width
                        and not reachable[ny, nx]
                        and grid[ny, nx] == 0):
                    reachable[ny, nx] = True
                    queue.append((ny, nx))

        return reachable

    @classmethod
    def summary(cls, all_metrics: Dict[str, Dict[str, float]]) -> str:
        """
        生成多算法多地图的评价摘要字符串。

        Args:
            all_metrics: {map_name: {planner_name: {metric: value}}}

        Returns:
            格式化表格字符串
        """
        lines = []
        lines.append("=" * 90)
        lines.append("全覆盖路径规划算法评价结果")
        lines.append("=" * 90)

        metrics_display = [
            ('coverage_rate', '覆盖率'),
            ('path_length', '路径长度'),
            ('coverage_efficiency', '覆盖效率'),
            ('num_turns', '转弯次数'),
            ('turn_rate', '转弯率'),
            ('runtime', '运行时间(s)'),
        ]

        for map_name, planners in all_metrics.items():
            lines.append(f"\n{'─' * 50}")
            lines.append(f"地图: {map_name}")
            lines.append(f"{'─' * 50}")

            header = f"{'指标':<20s}"
            for pname in planners.keys():
                header += f"  {pname:<15s}"
            lines.append(header)
            lines.append("-" * len(header))

            for key, display_name in metrics_display:
                row = f"{display_name:<20s}"
                for pname, metrics in planners.items():
                    val = metrics.get(key, 0)
                    if isinstance(val, float):
                        row += f"  {val:<15.4f}"
                    else:
                        row += f"  {val:<15d}"
                lines.append(row)

        lines.append(f"\n{'=' * 90}")
        return "\n".join(lines)
