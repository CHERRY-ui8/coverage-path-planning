"""
全覆盖路径规划结果可视化模块

三类图表（仅可视化内容，不含数据表格）：
1. 原始占据栅格地图
2. 覆盖轨迹图（带方向箭头 + 起/终图例）
3. 覆盖顺序热力图（统一配色尺度）
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

# ── CJK font config
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


class CoveragePlotter:
    """全覆盖路径规划结果可视化器 — 仅可视化，不含数据文字。"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self._title_fs = 16
        self._label_fs = 13
        self._legend_fs = 11
        self._cbar_fs = 12
        self._algo_fs = 14

    # ================================================================
    #  Single algorithm panel
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
        单算法三视图：原始地图 | 覆盖轨迹 | 覆盖顺序热力图。
        不含表头标题，不含数据指标。
        """
        fig = plt.figure(figsize=(self.figsize[0] * 2.4,
                                  self.figsize[1] * 0.55))
        gs = GridSpec(1, 3, wspace=0.22)

        ax_map = fig.add_subplot(gs[0, 0])
        ax_traj = fig.add_subplot(gs[0, 1])
        ax_heat = fig.add_subplot(gs[0, 2])

        self._plot_raw_map(ax_map, grid)
        self._plot_coverage_path(ax_traj, grid, path, covered)
        self._plot_coverage_heatmap(ax_heat, grid, path, covered,
                                    vmin=0, vmax=len(path))

        plt.tight_layout()
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
    #  Comparison panel  (multiple algorithms)
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
        多算法对比图。

        版面 (4 rows × 3 cols)：
          Row 0  [  Original Map (full width)                    ]
          Row 1  [ A* Greedy  │ Coverage Trajectory │ Coverage Order ]
          Row 2  [ BCD        │       (shared)      │    (shared)     ]
          Row 3  [ STC        │       (shared)      │    (shared)     ]

        图幅比例根据地图形状自适应（宽地图→偏宽，方地图→偏方）。
        列标题只出现在第一行，三行共用。
        """
        n = len(results)
        h, w = grid.shape
        # map aspect ratio, clamped to avoid extreme stretching
        ar = w / h
        # target figure aspect: layout_width_ratio * ar + label_column
        target_ar = max(1.8, min(2.07 * ar, 3.5))
        fig_w = self.figsize[0] * 2.2
        fig_h = fig_w / target_ar

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(n + 1, 3,
                      height_ratios=[0.22] + [0.26] * n,
                      width_ratios=[0.06, 1, 1],
                      hspace=0.10, wspace=0.10)

        # Row 0: original map
        ax_map = fig.add_subplot(gs[0, :])
        self._plot_raw_map(ax_map, grid)

        # Unified heatmap scale
        pl_names = list(results.keys())
        max_path_len = max(len(results[p][0]) for p in pl_names)

        for i, planner_name in enumerate(pl_names):
            path, covered, _ = results[planner_name]
            row = i + 1

            # Algorithm label
            ax_lbl = fig.add_subplot(gs[row, 0])
            self._plot_algo_label(ax_lbl, planner_name)

            # Trajectory (title only on first row)
            ax_traj = fig.add_subplot(gs[row, 1])
            self._plot_coverage_path(
                ax_traj, grid, path, covered,
                show_title=(i == 0)
            )

            # Heatmap (title only on first row)
            ax_heat = fig.add_subplot(gs[row, 2])
            self._plot_coverage_heatmap(
                ax_heat, grid, path, covered,
                vmin=0, vmax=max_path_len,
                show_title=(i == 0)
            )

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"comparison_{map_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        return filepath

    # ================================================================
    #  Sub-plot helpers
    # ================================================================

    def _plot_raw_map(self, ax: plt.Axes, grid: np.ndarray) -> None:
        ax.imshow(grid, cmap='gray_r', interpolation='nearest')
        ax.set_title('Original Map', fontsize=self._title_fs, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── trajectory with direction arrows ─────────────────

    def _plot_coverage_path(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        show_title: bool = True,
    ) -> None:
        height, width = grid.shape

        display = np.zeros((height, width, 3))
        display[grid == 0] = [1, 1, 1]
        display[grid == 1] = [0.2, 0.2, 0.2]
        if np.any(covered):
            overlay = np.zeros((height, width, 3))
            overlay[covered] = [0.6, 0.9, 0.6]
            mask = covered & (grid == 0)
            display[mask] = 0.3 * display[mask] + 0.7 * overlay[mask]
        ax.imshow(display, interpolation='nearest')

        # Path line with colour gradient (blue → red)
        if len(path) > 1:
            path_arr = np.array(path)
            n_seg = len(path) - 1
            step = max(1, n_seg // 200)
            for i in range(0, n_seg, step):
                i_end = min(i + step + 1, len(path))
                seg = path_arr[i:i_end]
                if len(seg) >= 2:
                    c = i / max(n_seg, 1)
                    ax.plot(seg[:, 1], seg[:, 0],
                            color=(c, 0.15, 1.0 - c),
                            linewidth=1.2, alpha=0.85)

        # Direction arrows
        self._add_direction_arrows(ax, path)

        # Start / End markers with legend
        if path:
            sy, sx = path[0]
            ax.plot(sx, sy, marker='o', markersize=8,
                    color='#0077ff', markeredgecolor='white',
                    markeredgewidth=1.5, label='Start')
            ey, ex = path[-1]
            ax.plot(ex, ey, marker='s', markersize=8,
                    color='#ff3333', markeredgecolor='white',
                    markeredgewidth=1.5, label='End')
            ax.legend(loc='upper right', fontsize=self._legend_fs,
                      framealpha=0.85, edgecolor='#cccccc')

        if show_title:
            ax.set_title('Coverage Trajectory',
                         fontsize=self._title_fs, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    def _add_direction_arrows(
        self, ax: plt.Axes, path: List[Tuple[int, int]],
        num_arrows: int = 20
    ) -> None:
        n = len(path)
        if n < 40:
            return
        gap = max(1, n // num_arrows)
        hg = max(1, gap // 3)
        for i in range(gap, n - hg, gap):
            y1, x1 = path[i - hg]
            y2, x2 = path[i + hg]
            dy = y2 - y1
            dx = x2 - x1
            if dy == 0 and dx == 0:
                continue
            norm = max(abs(dy), abs(dx))
            dy, dx = dy / norm, dx / norm
            my, mx = path[i]
            al = 0.35
            ax.annotate(
                '', xy=(mx + dx * al, my + dy * al),
                xytext=(mx - dx * al, my - dy * al),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=2.0, alpha=0.92),
                annotation_clip=False
            )

    # ── heatmap ──────────────────────────────────────────

    def _plot_coverage_heatmap(
        self,
        ax: plt.Axes,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        covered: np.ndarray,
        vmin: float = 0.0,
        vmax: Optional[float] = None,
        show_title: bool = True,
    ) -> None:
        height, width = grid.shape
        vmax = vmax or float(len(path))
        order = np.full((height, width), np.nan, dtype=float)
        for i, (y, x) in enumerate(path):
            if np.isnan(order[y, x]):
                order[y, x] = float(i)

        display = np.ones((height, width, 3))
        display[grid == 1] = [0.15, 0.15, 0.15]

        mask = ~np.isnan(order) & (grid == 0)
        if np.any(mask):
            normed = np.clip(order / max(vmax, 1), 0, 1)
            colours = COVERAGE_CMAP(normed[mask])[:, :3]
            for idx, (y, x) in enumerate(zip(*np.where(mask))):
                display[y, x] = colours[idx]

        ax.imshow(display, interpolation='nearest')
        if show_title:
            ax.set_title('Coverage Order',
                         fontsize=self._title_fs, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

        # Shared colour bar (only on the last heatmap in comparison)
        if vmax and vmax > 0:
            norm_obj = plt.Normalize(vmin=0, vmax=max(vmax, 1))
            sm = plt.cm.ScalarMappable(cmap=COVERAGE_CMAP, norm=norm_obj)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label('Early → Late', fontsize=self._cbar_fs)

    # ── algorithm name label (minimal, no metrics) ────────

    def _plot_algo_label(
        self, ax: plt.Axes, planner_name: str,
    ) -> None:
        ax.axis('off')
        label = PLANNER_LABELS.get(planner_name, planner_name)
        ax.text(0.5, 0.5, label,
                fontsize=self._algo_fs, fontweight='bold',
                ha='center', va='center')

    # ================================================================
    #  Utility
    # ================================================================

    @staticmethod
    def _format_map_name(name: str) -> str:
        mapping = {
            'empty': 'Empty Room', 'sparse_obstacles': 'Sparse Obstacles',
            'dense_obstacles': 'Dense Obstacles', 'corridor': 'Corridor',
            'multi_room': 'Multi-Room',
        }
        return mapping.get(name, name.replace('_', ' ').title())
