"""
实验运行器 —— 自动化全覆盖路径规划实验流程。

执行流程：
  1. 加载或生成所有测试地图
  2. 对每张地图分别运行三种规划算法
  3. 计算评价指标
  4. 保存结果可视化图片
  5. 输出汇总 CSV

使用方式：
  runner = ExperimentRunner(output_dir="./results")
  runner.run()
  runner.save_metrics_csv()
"""

import os
import time
import csv
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

from maps.map_generator import MapSet
from planners.astar_greedy import AstarGreedyCoverage
from planners.bcd import BCDCoverage
from planners.stc import STCCoverage
from metrics.evaluator import MetricsEvaluator
from visualization.plotter import CoveragePlotter


class ExperimentRunner:
    """
    全覆盖路径规划实验运行器。

    管理多地图 × 多算法的全流程实验，包括结果收集、可视化、指标汇总。
    """

    # 默认起点（地图中心附近的首个自由栅格）
    DEFAULT_PLANNERS = ['astar_greedy', 'bcd', 'stc']

    PLANNER_CLASSES = {
        'astar_greedy': AstarGreedyCoverage,
        'bcd': BCDCoverage,
        'stc': STCCoverage,
    }

    def __init__(
        self,
        output_dir: str = "./results",
        planners: Optional[List[str]] = None,
        map_seed: int = 42,
    ):
        """
        Args:
            output_dir: 输出根目录（图片、CSV 保存在此）
            planners:   要运行的规划器名称列表
            map_seed:   地图生成随机种子
        """
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, "images")
        self.csv_dir = output_dir
        self.metrics_dir = output_dir

        # 创建目录
        for d in [self.img_dir, self.csv_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        # 初始化各模块
        self.map_set = MapSet(seed=map_seed)
        self.evaluator = MetricsEvaluator()
        self.plotter = CoveragePlotter()

        # 规划器
        self.planner_names = planners or self.DEFAULT_PLANNERS
        self.planners = {
            name: self.PLANNER_CLASSES[name]()
            for name in self.planner_names
        }

        # 结果存储
        # {map_name: {planner_name: {path, covered, metrics}}}
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def run(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        运行全部实验。

        对 MapSet 中每张地图，运行所有规划器，计算指标并保存图片。

        Returns:
            {map_name: {planner_name: {path, covered, metrics, img_path}}}
        """
        print("=" * 80)
        print("全覆盖路径规划实验开始")
        print(f"地图数量: {len(self.map_set)}")
        print(f"规划算法: {', '.join(self.planner_names)}")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print("=" * 80)

        for map_name, grid in self.map_set:
            if map_name not in self.results:
                self.results[map_name] = {}

            print(f"\n{'─' * 60}")
            print(f"处理地图: {map_name}  ({grid.shape[0]}×{grid.shape[1]})")

            # 计算自动起点（左上角或中心附近的首个自由栅格）
            start = self._find_start(grid)

            for planner_name in self.planner_names:
                planner = self.planners[planner_name]
                print(f"  运行算法: {planner_name} ... ", end='', flush=True)

                try:
                    # 执行规划
                    path, covered, runtime = planner.plan(grid, start)

                    # 计算指标
                    metrics = self.evaluator.evaluate(
                        grid, path, covered, runtime, start
                    )

                    # 保存图片
                    img_path = self.plotter.plot_all(
                        grid=grid,
                        path=path,
                        covered=covered,
                        map_name=map_name,
                        planner_name=planner_name,
                        metrics=metrics,
                        output_dir=self.img_dir,
                        show=False
                    )

                    # 存储结果
                    self.results[map_name][planner_name] = {
                        'path': path,
                        'covered': covered,
                        'metrics': metrics,
                        'img_path': img_path,
                    }

                    coverage = metrics.get('coverage_rate', 0) * 100
                    path_len = metrics.get('path_length', 0)
                    turns = metrics.get('num_turns', 0)
                    run_t = metrics.get('runtime', 0)

                    print(
                        f"✓  覆盖率={coverage:.1f}%  "
                        f"路径={path_len}步  "
                        f"转弯={turns}次  "
                        f"耗时={run_t:.2f}s"
                    )

                except Exception as e:
                    print(f"✗ 失败: {e}")
                    self.results[map_name][planner_name] = {
                        'path': [],
                        'covered': np.zeros_like(grid, dtype=bool),
                        'metrics': {
                            'coverage_rate': 0.0,
                            'path_length': 0,
                            'coverage_efficiency': 0.0,
                            'num_turns': 0,
                            'turn_rate': 0.0,
                            'runtime': 0.0,
                            'free_cells_total': 0,
                            'free_cells_covered': 0,
                            'num_revisits': 0,
                            'revisit_rate': 0.0,
                        },
                        'img_path': '',
                    }

        # 保存对比图
        self._save_comparison_plots()

        # 保存指标 CSV
        csv_path = self.save_metrics_csv()
        print(f"\n{'=' * 60}")
        print(f"实验完成！")
        print(f"结果 CSV: {csv_path}")
        print(f"图片目录: {os.path.abspath(self.img_dir)}")

        # 打印摘要
        print(self.evaluator.summary(self._metrics_dict()))

        return self.results

    def save_metrics_csv(self) -> str:
        """
        将全部评价指标保存为 CSV 文件。

        Returns:
            CSV 文件路径
        """
        csv_path = os.path.join(self.csv_dir, "metrics.csv")

        fieldnames = [
            'map_name', 'planner_name',
            'coverage_rate', 'path_length', 'coverage_efficiency',
            'num_turns', 'turn_rate', 'runtime',
            'free_cells_total', 'free_cells_covered',
            'num_revisits', 'revisit_rate',
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for map_name, planners_dict in self.results.items():
                for planner_name, result in planners_dict.items():
                    metrics = result['metrics']
                    row = {
                        'map_name': map_name,
                        'planner_name': planner_name,
                        **metrics,
                    }
                    writer.writerow(row)

        print(f"\n指标已保存至: {csv_path}")
        return csv_path

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _find_start(self, grid: np.ndarray) -> Tuple[int, int]:
        """
        在地图中自动寻找合适的起点。
        优先选择地图中心附近的自由栅格。

        Args:
            grid: 占据栅格地图

        Returns:
            (row, col) 起点坐标
        """
        height, width = grid.shape

        # 尝试从中心开始向外搜索
        cy, cx = height // 2, width // 2
        for radius in range(max(height, width)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = cy + dy, cx + dx
                    if (0 <= y < height and 0 <= x < width
                            and grid[y, x] == 0):
                        return (y, x)

        # 备用：左上角搜索
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 0:
                    return (y, x)

        raise ValueError("地图中没有自由栅格")

    def _save_comparison_plots(self) -> None:
        """为每张地图生成算法对比图。"""
        print("\n  保存对比图 ... ", end='', flush=True)
        for map_name, grid in self.map_set:
            if map_name in self.results:
                planner_results = {}
                for pname, res in self.results[map_name].items():
                    if res['path']:
                        planner_results[pname] = (
                            res['path'],
                            res['covered'],
                            res['metrics']
                        )
                if planner_results:
                    self.plotter.plot_comparison(
                        grid=grid,
                        results=planner_results,
                        map_name=map_name,
                        output_dir=self.img_dir
                    )
        print("✓")

    def _metrics_dict(
        self
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """提取纯指标的嵌套字典。"""
        d = {}
        for map_name, planners_dict in self.results.items():
            d[map_name] = {}
            for pname, res in planners_dict.items():
                d[map_name][pname] = res['metrics']
        return d


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    """便捷的命令行入口函数。"""
    import argparse

    parser = argparse.ArgumentParser(
        description='全覆盖路径规划算法实验'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./results',
        help='输出目录（默认: ./results）'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子（默认: 42）'
    )
    parser.add_argument(
        '--planners', '-p', type=str, nargs='+',
        default=['astar_greedy', 'bcd', 'stc'],
        choices=['astar_greedy', 'bcd', 'stc'],
        help='要运行的规划算法（默认: 全部三种）'
    )
    args = parser.parse_args()

    runner = ExperimentRunner(
        output_dir=args.output,
        planners=args.planners,
        map_seed=args.seed,
    )
    runner.run()


if __name__ == '__main__':
    main()
