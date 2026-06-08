"""
合成占据栅格地图生成器 (Synthetic Occupancy Grid Map Generator)

生成 5 种标准测试地图用于全覆盖路径规划算法比较：
1. Empty（空房间）
2. Sparse Obstacles（稀疏障碍）
3. Dense Obstacles（密集障碍）
4. Corridor（长走廊）
5. Multi-room（多房间）

地图编码：
  0 = free（可通行）
  1 = obstacle（障碍物）

所有地图使用固定随机种子以保证可重复性。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MapGenerator:
    """
    占据栅格地图生成器。
    接收随机种子和地图尺寸参数，生成合成测试地图。
    """

    def __init__(self, seed: int = 42):
        """
        初始化地图生成器。

        Args:
            seed: 随机种子，保证结果可重复
        """
        self.rng = np.random.RandomState(seed)

    def empty(self, height: int = 50, width: int = 50) -> np.ndarray:
        """
        完全空闲地图 —— 无障碍物的开放空间。
        边界默认视为障碍物（不可达）。

        Args:
            height: 地图行数（默认 50）
            width:  地图列数（默认 50）

        Returns:
            shape (height, width) 的占据栅格地图
        """
        grid = np.zeros((height, width), dtype=np.int8)
        # 添加边界障碍物（墙）
        grid[0, :] = 1
        grid[height - 1, :] = 1
        grid[:, 0] = 1
        grid[:, width - 1] = 1
        return grid

    def sparse_obstacles(
        self, height: int = 50, width: int = 50,
        num_obstacles: int = 15,
        min_size: int = 2, max_size: int = 5
    ) -> np.ndarray:
        """
        稀疏障碍物地图 —— 少量随机放置的矩形障碍物。

        Args:
            height:      地图高度
            width:       地图宽度
            num_obstacles: 障碍物数量
            min_size:    障碍物最小尺寸
            max_size:    障碍物最大尺寸

        Returns:
            shape (height, width) 的占据栅格地图
        """
        grid = np.zeros((height, width), dtype=np.int8)
        grid[0, :] = 1
        grid[height - 1, :] = 1
        grid[:, 0] = 1
        grid[:, width - 1] = 1

        for _ in range(num_obstacles):
            h = self.rng.randint(min_size, max_size + 1)
            w = self.rng.randint(min_size, max_size + 1)
            y = self.rng.randint(2, height - h - 1)
            x = self.rng.randint(2, width - w - 1)
            grid[y : y + h, x : x + w] = 1

        return grid

    def dense_obstacles(
        self, height: int = 50, width: int = 50,
        num_obstacles: int = 40,
        min_size: int = 2, max_size: int = 4
    ) -> np.ndarray:
        """
        密集障碍物地图 —— 大量随机障碍物，连通区域受限。

        Args:
            height:      地图高度
            width:       地图宽度
            num_obstacles: 障碍物数量
            min_size:    障碍物最小尺寸
            max_size:    障碍物最大尺寸

        Returns:
            shape (height, width) 的占据栅格地图
        """
        grid = np.zeros((height, width), dtype=np.int8)
        grid[0, :] = 1
        grid[height - 1, :] = 1
        grid[:, 0] = 1
        grid[:, width - 1] = 1

        for _ in range(num_obstacles):
            h = self.rng.randint(min_size, max_size + 1)
            w = self.rng.randint(min_size, max_size + 1)
            y = self.rng.randint(2, height - h - 1)
            x = self.rng.randint(2, width - w - 1)
            grid[y : y + h, x : x + w] = 1

        return grid

    def corridor(
        self, height: int = 20, width: int = 100
    ) -> np.ndarray:
        """
        长走廊地图 —— 两端开放的狭窄通道。
        用于测试算法在受限空间中的表现。

        Args:
            height: 地图高度（默认 20，保持窄长）
            width:  地图宽度（默认 100）

        Returns:
            shape (height, width) 的占据栅格地图
        """
        grid = np.ones((height, width), dtype=np.int8)
        # 在中间开辟一条走廊
        corridor_width = 4
        start_y = height // 2 - corridor_width // 2
        grid[start_y : start_y + corridor_width, :] = 0
        # 边界障碍
        grid[0, :] = 1
        grid[height - 1, :] = 1
        grid[:, 0] = 0  # 入口开放
        grid[:, width - 1] = 0  # 出口开放
        return grid

    def multi_room(
        self, height: int = 50, width: int = 70,
        num_rooms: int = 6,
        door_width: int = 2
    ) -> np.ndarray:
        """
        多房间地图 —— 内墙分隔出若干房间，带门洞连接。
        用于测试算法在结构化环境中的覆盖能力。

        Args:
            height:     地图高度
            width:      地图宽度
            num_rooms:  房间数量
            door_width: 门洞宽度（栅格数）

        Returns:
            shape (height, width) 的占据栅格地图
        """
        grid = np.ones((height, width), dtype=np.int8)

        # 边界墙
        grid[0, :] = 1
        grid[height - 1, :] = 1
        grid[:, 0] = 1
        grid[:, width - 1] = 1

        # 内墙参数
        wall_margin = 3  # 边界留白
        usable_h = height - 2 * wall_margin
        usable_w = width - 2 * wall_margin

        # 使用固定划分：num_rooms 分解为行列
        # 例如 6 个房间 = 2 行 × 3 列
        if num_rooms <= 2:
            rows, cols = 1, num_rooms
        elif num_rooms <= 4:
            rows, cols = 2, (num_rooms + 1) // 2
        else:
            rows, cols = 2, (num_rooms + 1) // 2

        # 计算房间尺寸
        room_h = usable_h // rows
        room_w = usable_w // cols

        # 清空内部
        grid[wall_margin : height - wall_margin,
             wall_margin : width - wall_margin] = 0

        # 收集墙体位置
        wall_rows = []
        wall_cols = []
        for r in range(1, rows):
            y = wall_margin + r * room_h
            wall_rows.append(y)
        for c in range(1, cols):
            x = wall_margin + c * room_w
            wall_cols.append(x)

        # 添加所有墙体作为障碍
        for y in wall_rows:
            grid[y, :] = 1
        for x in wall_cols:
            grid[:, x] = 1

        # 门洞：每个水平墙段和垂直墙段各一个门，确保所有房间连通
        for y in wall_rows:
            # 水平墙上的门（每个房间所在列段各一个门）
            for ri in range(cols):
                left = wall_margin + ri * room_w
                right = left + room_w
                cx = self.rng.randint(left + 1, right - door_width)
                grid[y, cx : cx + door_width] = 0

        for x in wall_cols:
            # 垂直墙上的门（每个房间所在行段各一个门）
            for ci in range(rows):
                top = wall_margin + ci * room_h
                bottom = top + room_h
                cy = self.rng.randint(top + 1, bottom - door_width)
                grid[cy : cy + door_width, x] = 0

        # 如果房间数少于行×列，堵塞多余空间
        if rows * cols > num_rooms:
            # 将最后一个房间堵上，使其成为障碍
            grid[
                wall_margin + (rows - 1) * room_h : wall_margin + rows * room_h,
                wall_margin + (cols - 1) * room_w : wall_margin + cols * room_w
            ] = 1

        return grid


# ---------------------------------------------------------------------------
# 便捷函数：一键生成所有地图
# ---------------------------------------------------------------------------

def generate_all_maps(
    seed: int = 42,
    sizes: Optional[Dict[str, Tuple[int, int]]] = None
) -> Dict[str, np.ndarray]:
    """
    使用统一种子生成全部 5 张测试地图。

    Args:
        seed:  随机种子
        sizes: 可选，自定义每张地图的 (height, width)

    Returns:
        {map_name: grid}
    """
    gen = MapGenerator(seed)

    if sizes is None:
        sizes = {
            'empty':            (50, 50),
            'sparse_obstacles': (50, 50),
            'dense_obstacles':  (50, 50),
            'corridor':         (20, 100),
            'multi_room':       (50, 70),
        }

    maps = {}
    maps['empty']            = gen.empty(*sizes['empty'])
    maps['sparse_obstacles'] = gen.sparse_obstacles(*sizes['sparse_obstacles'])
    maps['dense_obstacles']  = gen.dense_obstacles(*sizes['dense_obstacles'])
    maps['corridor']         = gen.corridor(*sizes['corridor'])
    maps['multi_room']       = gen.multi_room(*sizes['multi_room'])

    return maps


# 便利的 MapSet 类
class MapSet:
    """
    包含 5 张测试地图的统一接口。
    支持通过名称索引，可迭代。
    """

    MAP_NAMES = ['empty', 'sparse_obstacles', 'dense_obstacles',
                 'corridor', 'multi_room']

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.maps = generate_all_maps(seed)
        self.names = list(self.maps.keys())

    def __getitem__(self, name: str) -> np.ndarray:
        return self.maps[name]

    def __iter__(self):
        """迭代生成 (name, grid) 对。"""
        for name in self.MAP_NAMES:
            if name in self.maps:
                yield name, self.maps[name]

    def __len__(self) -> int:
        return len(self.maps)

    def __repr__(self) -> str:
        parts = [f"MapSet(seed={self.seed})"]
        for name, grid in self:
            free = np.sum(grid == 0)
            occ = np.sum(grid == 1)
            parts.append(
                f"  {name:20s}  {grid.shape[0]:3d}×{grid.shape[1]:<3d}"
                f"  free={free:4d}  obstacle={occ:4d}"
            )
        return "\n".join(parts)
