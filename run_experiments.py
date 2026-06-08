#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全覆盖路径规划算法（Coverage Path Planning）比较实验主脚本

运行方式：
    python run_experiments.py

功能：
    1. 生成 5 张合成占据栅格测试地图
    2. 运行三种覆盖算法：A*-Greedy、BCD、STC
    3. 计算覆盖评价指标
    4. 保存可视化结果图片
    5. 输出 CSV 格式指标汇总

输出：
    results/
    ├── images/               # 路径可视化图片
    │   ├── empty_A*-Greedy.png
    │   ├── empty_BCD.png
    │   ├── empty_STC.png
    │   ├── ...
    │   └── comparison_*.png  # 算法对比图
    ├── metrics.csv           # 全部评价指标
    └── experiment_summary.txt # 文本摘要
"""

import os
import sys
import argparse
from datetime import datetime


def main():
    """实验主入口函数。"""
    parser = argparse.ArgumentParser(
        description='全覆盖路径规划算法（CPP）比较实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_experiments.py                         # 运行全部实验
  python run_experiments.py -o ./my_results         # 自定义输出目录
  python run_experiments.py -p astar_greedy stc     # 仅运行指定算法
  python run_experiments.py --seed 123              # 使用自定义随机种子
        """
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./results',
        help='输出目录路径（默认: ./results）'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help='地图生成随机种子（默认: 42）'
    )
    parser.add_argument(
        '--planners', '-p', type=str, nargs='+',
        default=['astar_greedy', 'bcd', 'stc'],
        choices=['astar_greedy', 'bcd', 'stc'],
        help='要运行的规划算法列表（默认全部）'
    )
    parser.add_argument(
        '--skip-comparison', action='store_true',
        help='跳过对比图生成'
    )

    args = parser.parse_args()

    # 打印实验配置
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"启动时间: {timestamp}")
    print(f"输出目录: {os.path.abspath(args.output)}")
    print(f"随机种子: {args.seed}")
    print(f"规划算法: {', '.join(args.planners)}")
    print()

    # 导入并运行实验
    try:
        from experiments.runner import ExperimentRunner
        runner = ExperimentRunner(
            output_dir=args.output,
            planners=args.planners,
            map_seed=args.seed,
        )
        results = runner.run()

        # 保存文本摘要
        summary_path = os.path.join(args.output, "experiment_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"全覆盖路径规划算法实验报告\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"实验时间: {timestamp}\n")
            f.write(f"随机种子: {args.seed}\n")
            f.write(f"规划算法: {', '.join(args.planners)}\n\n")
            f.write(runner.evaluator.summary(
                runner._metrics_dict()
            ))
        print(f"\n实验报告摘要已保存: {summary_path}")

        # 打印完成信息
        total_trials = (
            len(runner.results)
            * len(args.planners)
        )
        successful = sum(
            1 for m in runner.results.values()
            for p in m.values()
            if p['metrics']['coverage_rate'] > 0
        )
        print(f"\n{'=' * 60}")
        print(f"实验完成（{successful}/{total_trials} 成功）")
        print(f"结果保存至: {os.path.abspath(args.output)}")

    except ImportError as e:
        print(f"错误: 无法导入实验模块 — {e}")
        print("请确保在项目根目录运行此脚本。")
        print(f"当前目录: {os.getcwd()}")
        print("应在 coverage_path_planning/ 目录下运行")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 实验运行失败 — {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
