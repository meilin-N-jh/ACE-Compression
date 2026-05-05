#!/usr/bin/env python3
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())

import json
import os
import subprocess
import time
import threading
import sys
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# GPU配置
AVAILABLE_GPUS = [3, 4]  # 使用GPU 3和4

def get_task_files_info() -> List[Dict]:
    """获取所有任务文件信息"""
    task_files = []
    data_dir = "data"

    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return task_files

    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]

    for file in sorted(files):
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            sample_count = len(f.readlines())

        task_files.append({
            'filename': file,
            'filepath': file_path,
            'task_name': file.replace('.jsonl', ''),
            'sample_count': sample_count
        })

    return task_files

def distribute_tasks_to_instances(task_files: List[Dict], num_instances: int) -> List[List[Dict]]:
    """将不同的问卷任务分配给不同实例，避免重复评测"""
    if not task_files:
        return [[] for _ in range(num_instances)]

    # 按样本数量排序，优先分配大任务
    task_files_sorted = sorted(task_files, key=lambda x: x['sample_count'], reverse=True)

    # 为每个实例创建任务列表
    instance_tasks = [[] for _ in range(num_instances)]
    instance_sample_counts = [0] * num_instances

    # 使用贪心算法分配任务：始终分配给当前样本数最少的实例
    for task in task_files_sorted:
        # 找到当前样本数最少的实例
        min_idx = min(range(num_instances), key=lambda i: instance_sample_counts[i])

        instance_tasks[min_idx].append(task)
        instance_sample_counts[min_idx] += task['sample_count']

    # 打印分配结果
    print("\n📋 Task Distribution:")
    for i, tasks in enumerate(instance_tasks):
        if tasks:
            total_samples = sum(task['sample_count'] for task in tasks)
            task_names = [task['task_name'] for task in tasks]
            print(f"  Instance {i}: {len(tasks)} tasks, {total_samples} samples")
            print(f"    Tasks: {', '.join(task_names)}")
        else:
            print(f"  Instance {i}: No tasks assigned")

    return instance_tasks

class ProgressTracker:
    """进度跟踪器"""
    def __init__(self, total_tasks: int, total_samples: int):
        self.total_tasks = total_tasks
        self.total_samples = total_samples
        self.completed_tasks = 0
        self.completed_samples = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.instance_status = {}  # 实例状态跟踪

    def update_instance_status(self, instance_id: int, status: str, current_task: str = None, progress: str = None):
        """更新实例状态"""
        with self.lock:
            self.instance_status[instance_id] = {
                'status': status,
                'current_task': current_task,
                'progress': progress,
                'last_update': time.time()
            }

    def task_completed(self, instance_id: int, task_samples: int):
        """任务完成"""
        with self.lock:
            self.completed_tasks += 1
            self.completed_samples += task_samples

    def print_progress(self):
        """打印进度"""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            task_progress = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
            sample_progress = (self.completed_samples / self.total_samples) * 100 if self.total_samples > 0 else 0

            # 计算ETA
            if task_progress > 0:
                eta = (elapsed_time / task_progress) * (100 - task_progress) / 60
                eta_str = f"{eta:.1f}min"
            else:
                eta_str = "calculating..."

            # 清屏并显示进度
            os.system('clear' if os.name == 'posix' else 'cls')

            print("🚀 Multi-GPU GPTQ Parallel Evaluation Progress")
            print("=" * 60)
            print(f"📊 Overall Progress: {task_progress:.1f}% ({self.completed_tasks}/{self.total_tasks} tasks)")
            print(f"📈 Sample Progress: {sample_progress:.1f}% ({self.completed_samples}/{self.total_samples} samples)")
            print(f"⏱️  Elapsed: {elapsed_time/60:.1f}min | ETA: {eta_str}")
            print(f"🔧 Using GPUs: {AVAILABLE_GPUS}")
            print("\n🖥️  Instance Status:")

            for instance_id, status_info in sorted(self.instance_status.items()):
                status = status_info['status']
                current_task = status_info['current_task'] or 'Idle'
                progress = status_info['progress'] or ''

                if status == 'running':
                    print(f"  🔄 Instance {instance_id}: {current_task} {progress}")
                elif status == 'completed':
                    print(f"  ✅ Instance {instance_id}: Completed")
                elif status == 'error':
                    print(f"  ❌ Instance {instance_id}: Error - {current_task}")
                else:
                    print(f"  ⏸️  Instance {instance_id}: {status}")

            print("=" * 60)

def run_instance_with_tasks(instance_id: int, tasks: List[Dict], model_path: str, gpu_id: int,
                           progress_tracker: ProgressTracker):
    """运行指定实例，处理分配给它的所有任务"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    instance_start_time = time.time()
    total_samples = sum(task['sample_count'] for task in tasks)

    progress_tracker.update_instance_status(instance_id, 'starting')

    print(f"🚀 Starting instance {instance_id} on GPU {gpu_id} with {len(tasks)} tasks ({total_samples} samples)")

    results_summary = []

    for task_idx, task in enumerate(tasks):
        task_start_time = time.time()
        task_name = task['task_name']
        filename = task['filename']

        # 更新状态
        progress_tracker.update_instance_status(
            instance_id, 'running', filename,
            f"({task_idx+1}/{len(tasks)})"
        )

        print(f"  📝 Instance {instance_id} (GPU {gpu_id}): Processing {filename}")

        # 构建命令，使用统一的评测标准（5次尝试）
        cmd = [
            "python", "qwen_gptq.py",
            "--task", filename,
            "--model_name", model_path,
            "--language", "zh",
            "--try_times", "5",
            "--cot", "False"
        ]

        try:
            # 运行任务
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )

            # 实时监控输出并更新进度
            while process.poll() is None:
                time.sleep(2)
                progress_tracker.print_progress()

            stdout, stderr = process.communicate()

            task_time = time.time() - task_start_time

            if process.returncode == 0:
                print(f"  ✅ Instance {instance_id}: {filename} completed in {task_time/60:.1f}min")
                progress_tracker.task_completed(instance_id, task['sample_count'])

                # 检查结果文件
                result_file = f"results/{task_name}_Qwen-7B-Chat-Int4_gptq_results.jsonl"
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_count = len(f.readlines())
                    print(f"    📊 Generated {result_count} results")
                else:
                    print(f"    ⚠️  Result file not found: {result_file}")
                    result_count = 0
            else:
                print(f"  ❌ Instance {instance_id}: {filename} failed")
                print(f"    Error: {stderr[:200]}...")
                task_time = -1
                result_count = 0

            results_summary.append({
                'task_name': task_name,
                'filename': filename,
                'sample_count': task['sample_count'],
                'time_minutes': task_time,
                'result_count': result_count,
                'success': process.returncode == 0
            })

        except Exception as e:
            print(f"  ❌ Instance {instance_id}: Exception processing {filename}")
            print(f"    Error: {str(e)}")

            results_summary.append({
                'task_name': task_name,
                'filename': filename,
                'sample_count': task['sample_count'],
                'time_minutes': -1,
                'result_count': 0,
                'success': False,
                'error': str(e)
            })

    instance_time = time.time() - instance_start_time
    successful_tasks = sum(1 for r in results_summary if r['success'])
    total_results = sum(r['result_count'] for r in results_summary)

    progress_tracker.update_instance_status(instance_id, 'completed')

    print(f"✅ Instance {instance_id} (GPU {gpu_id}) completed in {instance_time/60:.1f}min")
    print(f"   📊 {successful_tasks}/{len(tasks)} tasks successful, {total_results} total results")

    return {
        'instance_id': instance_id,
        'gpu_id': gpu_id,
        'total_time_minutes': instance_time,
        'task_count': len(tasks),
        'successful_tasks': successful_tasks,
        'total_results': total_results,
        'total_samples': total_samples,
        'results_summary': results_summary
    }

def run_multi_gpu_parallel_evaluation(model_path: str, num_instances: int = None):
    """多GPU多实例并行评测主函数 - 为不同实例分配不同问卷任务"""
    print("🚀 Multi-GPU Multi-Instance Parallel GPTQ Evaluation")
    print(f"💻 Available GPUs: {AVAILABLE_GPUS}")
    print(f"💻 Model: {model_path}")
    print("=" * 80)

    # 获取所有任务文件
    task_files = get_task_files_info()

    if not task_files:
        print("❌ No task files found!")
        return

    print(f"📊 Found {len(task_files)} task files:")
    total_samples = 0
    for task in task_files:
        print(f"  📋 {task['filename']}: {task['sample_count']} samples")
        total_samples += task['sample_count']

    print(f"📈 Total samples across all tasks: {total_samples}")

    # 计算最优实例数量
    if num_instances is None:
        # 基于GPU数量计算：每个GPU最多8个实例
        max_instances_total = len(AVAILABLE_GPUS) * 8
        num_instances = min(len(task_files), max_instances_total)

    print(f"\n🔧 Using {num_instances} parallel instances")

    # 分配任务到实例
    instance_tasks = distribute_tasks_to_instances(task_files, num_instances)

    # 分配GPU
    gpu_assignment = []
    for i in range(num_instances):
        gpu_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
        gpu_assignment.append(gpu_id)

    # 初始化进度跟踪器
    progress_tracker = ProgressTracker(len(task_files), total_samples)

    # 启动进度监控线程
    def progress_monitor():
        while True:
            time.sleep(2)
            progress_tracker.print_progress()

    progress_thread = threading.Thread(target=progress_monitor, daemon=True)
    progress_thread.start()

    # 过滤掉空实例
    active_instances = []
    active_tasks = []
    active_gpu_ids = []
    for i, tasks in enumerate(instance_tasks):
        if tasks:  # 只保留有任务的实例
            active_instances.append(i)
            active_tasks.append(tasks)
            active_gpu_ids.append(gpu_assignment[i])

    if not active_tasks:
        print("❌ No tasks to process!")
        return

    print(f"\n🔧 Using {len(active_instances)} active instances")

    # 启动并行评测
    total_start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(active_instances)) as executor:
        # 提交所有实例任务
        futures = []
        for i, tasks in enumerate(active_tasks):
            instance_id = active_instances[i]
            gpu_id = active_gpu_ids[i]
            future = executor.submit(run_instance_with_tasks, instance_id, tasks, model_path, gpu_id, progress_tracker)
            futures.append(future)

        # 等待所有实例完成
        instance_results = []
        for future in as_completed(futures):
            result = future.result()
            instance_results.append(result)
            progress_tracker.print_progress()

    # 汇总结果
    total_time = time.time() - total_start_time

    print(f"\n🎉 All parallel evaluations completed!")
    print("=" * 80)

    print(f"📊 Overall Summary:")
    print(f"  🕐 Total time: {total_time/60:.1f} minutes")
    print(f"  📋 Active instances: {len(active_instances)}")
    print(f"  📈 Total samples processed: {total_samples}")

    successful_instances = sum(1 for r in instance_results if r['successful_tasks'] > 0)
    total_successful_tasks = sum(r['successful_tasks'] for r in instance_results)
    total_generated_results = sum(r['total_results'] for r in instance_results)

    print(f"  ✅ Successful instances: {successful_instances}/{len(active_instances)}")
    print(f"  ✅ Successful tasks: {total_successful_tasks}/{len(task_files)}")
    print(f"  📄 Generated results: {total_generated_results}")

    if total_time > 0:
        avg_speed = total_samples / (total_time / 60)
        print(f"  ⚡ Average processing speed: {avg_speed:.1f} samples/min")

    print(f"\n📁 Results saved in: ./results/")
    print(f"📄 To analyze results: python3 get_results.py")
    print("=" * 80)

    # 生成详细报告
    generate_detailed_report(instance_results, task_files, total_time)

def generate_detailed_report(instance_results: List[Dict], task_files: List[Dict], total_time: float):
    """生成详细的评测报告"""
    report = []
    report.append("📊 Detailed Evaluation Report")
    report.append("=" * 60)
    report.append("")

    # 实例总结
    report.append("🖥️  Instance Performance Summary:")
    for result in sorted(instance_results, key=lambda x: x['instance_id']):
        status = "✅" if result['successful_tasks'] > 0 else "❌"
        report.append(f"  Instance {result['instance_id']}: {status}")
        report.append(f"    Tasks: {result['successful_tasks']}/{result['task_count']}")
        report.append(f"    Time: {result['total_time_minutes']:.1f}min")
        report.append(f"    Results: {result['total_results']}")
        report.append("")

    # 任务详情
    report.append("📋 Task Details:")
    for result in sorted(instance_results, key=lambda x: x['instance_id']):
        for task_result in result['results_summary']:
            status = "✅" if task_result['success'] else "❌"
            time_str = f"{task_result['time_minutes']:.1f}min" if task_result['time_minutes'] > 0 else "Failed"
            report.append(f"  {status} Instance {result['instance_id']}: {task_result['filename']}")
            report.append(f"    Samples: {task_result['sample_count']}, Results: {task_result['result_count']}, Time: {time_str}")

    report.append("")
    report.append("=" * 60)

    # 保存报告
    report_content = "\n".join(report)
    with open("gptq_parallel_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"📄 Detailed report saved to: gptq_parallel_evaluation_report.txt")

if __name__ == "__main__":
    model_path = f"{ARTIFACT_ROOT}/models/Qwen-7B-Chat-Int4"

    # 计算最优实例数量（基于2块GPU，每块最多8个实例）
    max_instances_total = len(AVAILABLE_GPUS) * 8
    num_instances = min(8, max_instances_total)  # 最多8个实例，保守一些

    print(f"🔧 Calculated optimal instances: {num_instances} (max possible: {max_instances_total})")

    # 运行多GPU多实例并行评测
    run_multi_gpu_parallel_evaluation(model_path, num_instances)