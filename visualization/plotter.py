"""
可视化模块 —— 为全覆盖路径规划结果生成三类图表：

1. 原始占据栅格地图
2. 覆盖轨迹图（路径叠加在地图上）
3. 覆盖顺序热力图（颜色渐变展示覆盖先后顺序）

所有图表使用统一风格，保存为高分辨率 PNG。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict, Optional
import os


# 中文字体配置（macOS）
_cjk_font_found = False
for _f in ['Heiti SC', 'Heiti TC', 'STSong', 'Songti SC',
           'PingFang SC', 'Apple SD Gothic Neo']:
    try:
        _p = fm.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_f] + \
            [x for x in plt.rcParams.get('font.sans-serif', [])
             if x != _f]
        _cjk_font_found = True
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
if _cjk_font_found:
    print(f"[Plotter] CJK font activated: {plt.rcParams['font.sans-serif'][0]}")


# 自定义颜色方案
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    'coverage_heat',
    ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
    N=256
)

TURN_CMAP = LinearSegmentedColormap.from_list(
    'turn_cmap',
    ['#ffffff', '#ff6b6b'],
    N=256
)


class CoveragePlotter:
    """
    全覆盖路径规划结果可视化器。

    提供三种视图的绘制功能：
    - 原始地图
    - 覆盖轨迹
    - 覆盖顺序热力图
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Args:
            figsize: 全局图幅尺寸（英寸）
            dpi:     输出分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self._title_fs = 24     # 子图标题（原文12→14→24）
        self._suptitle_fs = 30  # 总标题（原文16→18→30）
        self._label_fs = 18     # 坐标轴标签（原文默认→11→18）
        self._info_fs = 16      # 图注信息（原文8→10→16）
        self._cbar_fs = 16      # 颜色条（原文8→10→16）
        self._cbar_fs = 10      # 颜色条

    def plot_all(
        self,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        map_name: str,
        planner_name: str,
        metrics: Dict[str, float],
        output_dir: str,
        show: bool = False
    ) -> str:
        """
        为给定地图和规划结果生成完整的可视化面板（三张子图）。

        Args:
            grid:        占据栅格地图
            path:        覆盖路径
            covered:     覆盖标记矩阵
            map_name:    地图名称（用于标题/文件名）
            planner_name: 规划器名称
            metrics:     评价指标
            output_dir:  输出目录
            show:        是否显示图片

        Returns:
            保存的文件路径
        """
        height, width = grid.shape

        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0] * 2.2,
                                                  self.figsize[1] * 0.55))
        fig.suptitle(
            f"{planner_name} — {self._format_map_name(map_name)}",
            fontsize=self._suptitle_fs, fontweight='bold', y=1.02
        )

        # 1. 原始地图
        self._plot_raw_map(axes[0], grid)

        # 2. 覆盖轨迹图
        self._plot_coverage_path(axes[1], grid, path, covered, metrics)

        # 3. 覆盖顺序热力图
        self._plot_coverage_heatmap(axes[2], grid, path, covered)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{map_name}_{planner_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        if show:
            plt.show()
        plt.close(fig)

        return filepath

    # ------------------------------------------------------------------
    # 子图绘制方法
    # ------------------------------------------------------------------

    def _plot_raw_map(self, ax: plt.Axes, grid: np.ndarray) -> None:
        """绘制原始占据栅格地图。"""
        ax.imshow(grid, cmap='gray_r', interpolation='nearest')
        ax.set_title('原始地图 (Occupancy Grid)', fontsize=self._title_fs, pad=12)
        ax.set_xlabel('列 (x)', fontsize=self._label_fs)
        ax.set_ylabel('行 (y)', fontsize=self._label_fs)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_coverage_path(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
        """
        绘制覆盖轨迹图。
        - 深色 = 障碍物
        - 浅色 = 已覆盖栅格
        - 线条 = 覆盖路径（渐变颜色）
        - 标记 = 起点
        """
        height, width = grid.shape

        # 背景：障碍物黑色，自由空间白色
        display = np.zeros((height, width, 3))
        display[grid == 0] = [1, 1, 1]      # 自由 = 白色
        display[grid == 1] = [0.2, 0.2, 0.2]  # 障碍物 = 深灰

        # 覆盖栅格半透明着色
        if np.any(covered):
            overlay = np.zeros((height, width, 3))
            overlay[covered] = [0.6, 0.9, 0.6]  # 浅绿色
            # 仅在半透明区域覆盖
            mask = covered & (grid == 0)
            display[mask] = 0.3 * display[mask] + 0.7 * overlay[mask]

        ax.imshow(display, interpolation='nearest')

        # 绘制路径（分段着色以展示方向）
        if len(path) > 1:
            path_arr = np.array(path)
            # 使用分段线条，颜色从蓝到红渐变
            n_segments = len(path) - 1
            for i in range(max(1, n_segments - 50), n_segments + 1):
                # 取稀疏采样以避免性能问题
                step = max(1, n_segments // 200)
                if i % step != 0 and i != n_segments - 1:
                    continue

                idx_start = max(0, i - 1)
                idx_end = min(len(path), i + 1)
                segment = path_arr[idx_start:idx_end]
                if len(segment) >= 2:
                    color_val = i / max(n_segments, 1)
                    ax.plot(
                        segment[:, 1], segment[:, 0],
                        color=(color_val, 0.2, 1.0 - color_val),
                        linewidth=0.8, alpha=0.8
                    )

        # 起点
        if path:
            sy, sx = path[0]
            ax.plot(sx, sy, marker='o', markersize=8,
                    color='blue', markeredgecolor='white',
                    markeredgewidth=1.5, label='起点')
            # 终点
            ey, ex = path[-1]
            ax.plot(ex, ey, marker='s', markersize=8,
                    color='red', markeredgecolor='white',
                    markeredgewidth=1.5, label='终点')

        ax.set_title('覆盖轨迹', fontsize=self._title_fs, pad=12)
        ax.set_xlabel('列 (x)', fontsize=self._label_fs)
        ax.set_ylabel('行 (y)', fontsize=self._label_fs)
        ax.set_xticks([])
        ax.set_yticks([])

        # 指标图注
        if metrics:
            info = (
                f"覆盖率: {metrics.get('coverage_rate', 0)*100:.1f}%\n"
                f"路径长: {metrics.get('path_length', 0)}\n"
                f"转弯数: {metrics.get('num_turns', 0)}\n"
                f"耗时: {metrics.get('runtime', 0):.2f}s"
            )
            ax.text(
                0.02, 0.02, info, transform=ax.transAxes,
                fontsize=self._info_fs, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white', alpha=0.8)
            )

    def _plot_coverage_heatmap(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        max_samples: int = 20000
    ) -> None:
        """
        绘制覆盖顺序热力图。
        - 颜色从紫色（先覆盖）渐变到黄色（后覆盖）
        - 障碍物显示为黑色
        - 空白表示未覆盖的自由栅格
        """
        height, width = grid.shape

        # 构建覆盖顺序矩阵
        order_map = np.full((height, width), -1, dtype=int)
        step = max(1, len(path) // max_samples)
        for i, (y, x) in enumerate(path):
            if i % step == 0:
                if 0 <= y < height and 0 <= x < width:
                    order_map[y, x] = i

        # 对每个栅格，使用最早访问时间
        order_full = np.full((height, width), -1, dtype=float)
        for i, (y, x) in enumerate(path):
            if order_full[y, x] < 0:
                order_full[y, x] = i

        # 显示
        display = np.full((height, width, 3), 1.0)

        # 障碍物
        display[grid == 1] = [0.15, 0.15, 0.15]

        # 覆盖栅格着色
        covered_mask = (order_full >= 0) & (grid == 0)
        if np.any(covered_mask):
            order_vals = order_full[covered_mask]
            norm_order = order_vals / max(order_vals.max(), 1)
            # 映射到颜色
            colors = COVERAGE_CMAP(norm_order)[:, :3]
            for idx, (y, x) in enumerate(zip(*np.where(covered_mask))):
                display[y, x] = colors[idx]

        ax.imshow(display, interpolation='nearest')

        # 添加上色说明
        ax.set_title('覆盖顺序热力图 (紫→黄=先→后)', fontsize=self._title_fs, pad=12)
        ax.set_xlabel('列 (x)', fontsize=self._label_fs)
        ax.set_ylabel('行 (y)', fontsize=self._label_fs)
        ax.set_xticks([])
        ax.set_yticks([])

        # 添加颜色条
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=COVERAGE_CMAP, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('覆盖顺序 (先→后)', fontsize=self._cbar_fs)

    # ------------------------------------------------------------------
    # 批量可视化
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        grid: np.ndarray,
        results: Dict[str, Tuple[List[Tuple[int, int]],
                                 np.ndarray, Dict[str, float]]],
        map_name: str,
        output_dir: str
    ) -> str:
        """
        绘制多算法对比图：每行一个算法，三列对应三种视图。

        Args:
            grid:    占据栅格地图
            results: {planner_name: (path, covered, metrics)}
            map_name: 地图名称
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        n_algos = len(results)
        fig, axes = plt.subplots(
            n_algos, 3,
            figsize=(self.figsize[0] * 2.0, self.figsize[1] * n_algos * 0.8)
        )

        if n_algos == 1:
            axes = axes.reshape(1, 3)

        fig.suptitle(
            f"算法对比 — {self._format_map_name(map_name)}",
            fontsize=self._suptitle_fs, fontweight='bold', y=0.98
        )

        for row, (planner_name, (path, covered, metrics)) in enumerate(
                results.items()):
            # 第一列：原始地图（仅第一行）
            if row == 0:
                self._plot_raw_map(axes[row, 0], grid)
            else:
                axes[row, 0].axis('off')

            # 第二列：覆盖轨迹
            self._plot_coverage_path(
                axes[row, 1], grid, path, covered, metrics
            )
            axes[row, 1].set_ylabel(
                planner_name, fontsize=self._label_fs, fontweight='bold'
            )

            # 第三列：热力图
            self._plot_coverage_heatmap(axes[row, 2], grid, path, covered)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        os.makedirs(output_dir, exist_ok=True)
        filename = f"comparison_{map_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        return filepath

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _format_map_name(name: str) -> str:
        """将 snake_case 转换为可读的标题格式。"""
        mapping = {
            'empty': '空地图 (Empty)',
            'sparse_obstacles': '稀疏障碍物 (Sparse Obstacles)',
            'dense_obstacles': '密集障碍物 (Dense Obstacles)',
            'corridor': '长走廊 (Corridor)',
            'multi_room': '多房间 (Multi-Room)',
        }
        return mapping.get(name, name.replace('_', ' ').title())
