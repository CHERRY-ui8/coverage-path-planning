"""
全覆盖路径规划结果可视化模块

三类图表：
1. 原始占据栅格地图
2. 覆盖轨迹图（带方向箭头的折线 + 起/终图例）
3. 覆盖顺序热力图（统一配色尺度）

所有图表使用统一风格，高分辨率 PNG 输出。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional
import os

# ── CJK font config ──────────────────────────────────────
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

# ── Colour maps ───────────────────────────────────────────
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    'coverage_heat',
    ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
    N=256
)

PLANNER_LABELS = {
    'astar_greedy': 'A* Greedy',
    'bcd':          'BCD',
    'stc':          'STC',
}

MAP_LABELS = {
    'empty':            'Empty Room',
    'sparse_obstacles': 'Sparse Obstacles',
    'dense_obstacles':  'Dense Obstacles',
    'corridor':         'Corridor',
    'multi_room':       'Multi-Room',
}


class CoveragePlotter:
    """全覆盖路径规划结果可视化器。"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self._title_fs = 26       # 子图标题
        self._suptitle_fs = 32    # 总标题 / 算法行标题
        self._label_fs = 20       # 坐标轴标签
        self._info_fs = 18        # 图注 / 表格文字
        self._cbar_fs = 18        # 颜色条
        self._algo_fs = 28        # 对比图算法名标签

    # ================================================================
    #  Single-algorithm panel  (plot_all)
    # ================================================================

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
        单算法三视图面板。
        版面：上排 3 张图（原始地图 / 覆盖轨迹 / 热力图）
              下排横贯底部的指标卡片。
        """
        height, width = grid.shape
        fig = plt.figure(figsize=(self.figsize[0] * 2.4,
                                  self.figsize[1] * 0.70),
                         constrained_layout=False)
        gs = GridSpec(2, 3, height_ratios=[1, 0.12],
                      hspace=0.25, wspace=0.20)

        fig.suptitle(
            f"{PLANNER_LABELS.get(planner_name, planner_name)}  —  "
            f"{self._format_map_name(map_name)}",
            fontsize=self._suptitle_fs, fontweight='bold', y=1.02
        )

        # Row 0: three views
        ax_map = fig.add_subplot(gs[0, 0])
        ax_traj = fig.add_subplot(gs[0, 1])
        ax_heat = fig.add_subplot(gs[0, 2])

        self._plot_raw_map(ax_map, grid)
        self._plot_coverage_path(ax_traj, grid, path, covered)
        self._plot_coverage_heatmap(ax_heat, grid, path, covered,
                                    vmin=0, vmax=len(path))

        # Row 1: metrics bar (spanning full width)
        ax_metrics = fig.add_subplot(gs[1, :])
        self._plot_metrics_bar(ax_metrics, metrics)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{map_name}_{planner_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        if show:
            plt.show()
        plt.close(fig)
        return filepath

    # ================================================================
    #  Comparison panel  (plot_comparison)
    # ================================================================

    def plot_comparison(
        self,
        grid: np.ndarray,
        results: Dict[str, Tuple[List[Tuple[int, int]],
                                 np.ndarray, Dict[str, float]]],
        map_name: str,
        output_dir: str
    ) -> str:
        """
        多算法对比面板。

        版面 (4 rows × 3 cols GridSpec)：
          Row 0  [  Original Map (col 0–2, full width)  ]
          Row 1  [  A* Greedy label  │  Trajectory  │  Heatmap  ]
          Row 2  [  BCD    label     │  Trajectory  │  Heatmap  ]
          Row 3  [  STC    label     │  Trajectory  │  Heatmap  ]

        每行左侧纵列显示算法名 + 核心指标。
        热力图使用统一配色尺度（以最长路径归一化）。
        """
        n = len(results)  # 3
        fig = plt.figure(figsize=(self.figsize[0] * 2.6,
                                  self.figsize[1] * 1.1),
                         constrained_layout=False)
        gs = GridSpec(n + 1, 3,
                      height_ratios=[0.55] + [1.0] * n,
                      width_ratios=[0.20, 1, 1],
                      hspace=0.25, wspace=0.20)

        fig.suptitle(
            f"Algorithm Comparison — {self._format_map_name(map_name)}",
            fontsize=self._suptitle_fs, fontweight='bold', y=1.02
        )

        # ── Row 0: original map (full width) ──
        ax_map = fig.add_subplot(gs[0, :])
        self._plot_raw_map(ax_map, grid)

        # ── Rows 1-3: per-algorithm ──
        pl_names = list(results.keys())
        # Determine unified heatmap range
        max_path_len = max(len(results[p][0]) for p in pl_names)

        for i, planner_name in enumerate(pl_names):
            path, covered, metrics = results[planner_name]
            row = i + 1

            # Column 0: algorithm label + metrics
            ax_lbl = fig.add_subplot(gs[row, 0])
            self._plot_algo_label(ax_lbl, planner_name, metrics)

            # Column 1: coverage trajectory
            ax_traj = fig.add_subplot(gs[row, 1])
            self._plot_coverage_path(ax_traj, grid, path, covered)

            # Column 2: heatmap (unified vmax)
            ax_heat = fig.add_subplot(gs[row, 2])
            self._plot_coverage_heatmap(ax_heat, grid, path, covered,
                                        vmin=0, vmax=max_path_len)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        os.makedirs(output_dir, exist_ok=True)
        filename = f"comparison_{map_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        return filepath

    # ================================================================
    #  Sub-plotting helpers
    # ================================================================

    # ── raw map ────────────────────────────────────────────

    def _plot_raw_map(self, ax: plt.Axes, grid: np.ndarray) -> None:
        """原始占据栅格地图。"""
        ax.imshow(grid, cmap='gray_r', interpolation='nearest')
        ax.set_title('Original Map\n(Occupancy Grid)',
                     fontsize=self._title_fs, pad=14)
        ax.set_xlabel('Column (x)', fontsize=self._label_fs)
        ax.set_ylabel('Row (y)', fontsize=self._label_fs)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── coverage trajectory with direction arrows ──────────

    def _plot_coverage_path(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
    ) -> None:
        """
        覆盖轨迹图。

        - 深灰 = 障碍物
        - 浅绿半透明 = 已覆盖
        - 蓝→红渐变折线 = 路径（方向）
        - 白色箭头 = 遍历方向标记
        - ● 蓝 = 起点  ■ 红 = 终点
        """
        height, width = grid.shape

        # Background
        display = np.zeros((height, width, 3))
        display[grid == 0] = [1, 1, 1]
        display[grid == 1] = [0.2, 0.2, 0.2]
        if np.any(covered):
            overlay = np.zeros((height, width, 3))
            overlay[covered] = [0.6, 0.9, 0.6]
            mask = covered & (grid == 0)
            display[mask] = 0.3 * display[mask] + 0.7 * overlay[mask]
        ax.imshow(display, interpolation='nearest')

        # ── path line with colour gradient ──
        if len(path) > 1:
            path_arr = np.array(path)
            n_seg = len(path) - 1
            step = max(1, n_seg // 250)
            for i in range(0, n_seg, step):
                i_end = min(i + step + 1, len(path))
                seg = path_arr[i:i_end]
                if len(seg) >= 2:
                    c = i / max(n_seg, 1)
                    ax.plot(seg[:, 1], seg[:, 0],
                            color=(c, 0.15, 1.0 - c),
                            linewidth=1.5, alpha=0.85)

        # ── direction arrows ──
        self._add_direction_arrows(ax, path)

        # ── start / end markers with legend ──
        if path:
            sy, sx = path[0]
            ax.plot(sx, sy, marker='o', markersize=12,
                    color='#0077ff', markeredgecolor='white',
                    markeredgewidth=2, label='Start')
            ey, ex = path[-1]
            ax.plot(ex, ey, marker='s', markersize=12,
                    color='#ff3333', markeredgecolor='white',
                    markeredgewidth=2, label='End')
            ax.legend(loc='upper right', fontsize=self._info_fs,
                      framealpha=0.85, edgecolor='#cccccc')

        ax.set_title('Coverage Trajectory\n(with direction arrows)',
                     fontsize=self._title_fs, pad=14)
        ax.set_xlabel('Column (x)', fontsize=self._label_fs)
        ax.set_ylabel('Row (y)', fontsize=self._label_fs)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── direction arrows ──────────────────────────────

    def _add_direction_arrows(
        self, ax: plt.Axes, path: List[Tuple[int, int]],
        num_arrows: int = 22
    ) -> None:
        """
        沿路径添加白色方向箭头。

        Args:
            ax:           matplotlib Axes
            path:         路径坐标序列 [(row, col), ...]
            num_arrows:   箭头总数（均匀分布）
        """
        n = len(path)
        if n < 40:
            return
        gap = max(1, n // num_arrows)
        # half-gap used to compute local orientation
        hg = max(1, gap // 3)

        for i in range(gap, n - hg, gap):
            # local direction vector
            y1, x1 = path[i - hg]
            y2, x2 = path[i + hg]
            dy = y2 - y1
            dx = x2 - x1
            if dy == 0 and dx == 0:
                continue
            # normalize to unit step
            norm = max(abs(dy), abs(dx))
            dy, dx = dy / norm, dx / norm

            # short white arrow centred at path[i]
            mid_y, mid_x = path[i]
            arr_len = 0.35
            ax.annotate(
                '', xy=(mid_x + dx * arr_len, mid_y + dy * arr_len),
                xytext=(mid_x - dx * arr_len, mid_y - dy * arr_len),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=2.5, alpha=0.95),
                annotation_clip=False
            )

    # ── coverage heatmap ───────────────────────────────────

    def _plot_coverage_heatmap(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        vmin: float = 0.0,
        vmax: Optional[float] = None,
    ) -> None:
        """
        覆盖顺序热力图。

        Args:
            vmin, vmax: 统一配色范围（用于对比图多张热力图对齐）。
        """
        height, width = grid.shape

        vmax = vmax or float(len(path))
        order_full = np.full((height, width), np.nan, dtype=float)
        for i, (y, x) in enumerate(path):
            if np.isnan(order_full[y, x]):
                order_full[y, x] = float(i)

        # Display array
        display = np.ones((height, width, 3))
        display[grid == 1] = [0.15, 0.15, 0.15]

        covered_mask = ~np.isnan(order_full) & (grid == 0)
        if np.any(covered_mask):
            normed = np.clip(order_full / max(vmax, 1), 0, 1)
            colours = COVERAGE_CMAP(normed[covered_mask])[:, :3]
            for idx, (y, x) in enumerate(zip(*np.where(covered_mask))):
                display[y, x] = colours[idx]

        ax.imshow(display, interpolation='nearest')
        ax.set_title('Coverage Order\n(purple → yellow)',
                     fontsize=self._title_fs, pad=14)
        ax.set_xlabel('Column (x)', fontsize=self._label_fs)
        ax.set_ylabel('Row (y)', fontsize=self._label_fs)
        ax.set_xticks([])
        ax.set_yticks([])

        # Colour bar
        norm_obj = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=COVERAGE_CMAP, norm=norm_obj)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.75, pad=0.03)
        cbar.set_label('Coverage Order (early → late)',
                       fontsize=self._cbar_fs)

    # ── metrics bar (for plot_all) ─────────────────────────

    def _plot_metrics_bar(
        self, ax: plt.Axes, metrics: Dict[str, float]
    ) -> None:
        """
        横贯底部的指标卡片。
        显示：覆盖率 / 路径长度 / 覆盖效率 / 转弯次数 / 运行时间
        """
        ax.axis('off')
        items = [
            ("Coverage Rate",
             f"{metrics.get('coverage_rate', 0) * 100:.1f}%"),
            ("Path Length",
             f"{metrics.get('path_length', 0)}"),
            ("Coverage Efficiency",
             f"{metrics.get('coverage_efficiency', 0):.3f}"),
            ("Turns",
             f"{metrics.get('num_turns', 0)}"),
            ("Runtime",
             f"{metrics.get('runtime', 0):.3f}s"),
        ]

        text = '    │    '.join(f'{k}:  {v}' for k, v in items)
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                fontsize=self._info_fs, ha='center', va='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6',
                          facecolor='#f0f0f0', edgecolor='#cccccc'))

    # ── algorithm label column (for comparison) ────────────

    def _plot_algo_label(
        self,
        ax: plt.Axes,
        planner_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        对比图左侧纵列：算法名 + 核心指标。
        """
        ax.axis('off')
        label = PLANNER_LABELS.get(planner_name, planner_name)
        lines = [
            f"{label}",
            "",
            f"Coverage:  {metrics.get('coverage_rate', 0)*100:.1f}%",
            f"Length:    {metrics.get('path_length', 0)}",
            f"Turns:     {metrics.get('num_turns', 0)}",
            f"Runtime:   {metrics.get('runtime', 0):.2f}s",
        ]
        text = '\n'.join(lines)
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                fontsize=self._algo_fs, fontweight='bold',
                ha='center', va='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#fafafa', edgecolor='#dddddd'))

    # ================================================================
    #  Utilities
    # ================================================================

    @staticmethod
    def _format_map_name(name: str) -> str:
        """snake_case → display title."""
        mapping = {
            'empty':            'Empty Room',
            'sparse_obstacles': 'Sparse Obstacles',
            'dense_obstacles':  'Dense Obstacles',
            'corridor':         'Corridor',
            'multi_room':       'Multi-Room',
        }
        return mapping.get(name, name.replace('_', ' ').title())
