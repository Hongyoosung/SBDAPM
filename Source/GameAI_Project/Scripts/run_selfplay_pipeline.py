"""
Self-Play Training Pipeline Orchestrator

Automates the complete self-play training loop:
1. Collect N games of self-play data
2. Train all networks (Value, World Model, RL Policy)
3. Export models to ONNX
4. Copy models to UE5
5. Evaluate against baseline
6. Repeat

Usage:
    python run_selfplay_pipeline.py --games 1000 --iterations 5

Requirements:
    pip install numpy torch tqdm tensorboard
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import shutil


class SelfPlayPipeline:
    """Orchestrates complete self-play training pipeline"""

    def __init__(self, games_per_iteration: int, training_iterations: int,
                 output_base_dir: Path, ue5_project_path: Path = None):
        """
        Args:
            games_per_iteration: Number of games to collect per iteration
            training_iterations: Number of training iterations to run
            output_base_dir: Base directory for all outputs
            ue5_project_path: Path to UE5 project (for automatic model deployment)
        """
        self.games_per_iteration = games_per_iteration
        self.training_iterations = training_iterations
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.ue5_project_path = Path(ue5_project_path) if ue5_project_path else None

        # Directory structure
        self.data_dir = self.output_base_dir / 'selfplay_data'
        self.models_dir = self.output_base_dir / 'models'
        self.logs_dir = self.output_base_dir / 'logs'
        self.eval_dir = self.output_base_dir / 'evaluation'

        for d in [self.data_dir, self.models_dir, self.logs_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Scripts directory
        self.scripts_dir = Path(__file__).parent

        # Pipeline state
        self.current_iteration = 0
        self.pipeline_log = []

        print("=" * 80)
        print("Self-Play Training Pipeline Orchestrator")
        print("=" * 80)
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Training iterations: {training_iterations}")
        print(f"Total games target: {games_per_iteration * training_iterations}")
        print(f"Output directory: {self.output_base_dir}")
        print("=" * 80)

    def run_script(self, script_name: str, args: list) -> bool:
        """Run a Python script and return success status"""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            print(f"Error: Script not found: {script_path}")
            return False

        cmd = [sys.executable, str(script_path)] + args

        print(f"\n[Running] {' '.join(cmd)}")
        print("-" * 80)

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            duration = time.time() - start_time

            self.pipeline_log.append({
                'timestamp': datetime.now().isoformat(),
                'script': script_name,
                'args': args,
                'status': 'success',
                'duration_seconds': duration
            })

            print(f"[Complete] Finished in {duration:.1f}s")
            return True

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"[Error] Script failed after {duration:.1f}s")
            print(f"  Exit code: {e.returncode}")

            self.pipeline_log.append({
                'timestamp': datetime.now().isoformat(),
                'script': script_name,
                'args': args,
                'status': 'failed',
                'duration_seconds': duration,
                'error': str(e)
            })

            return False

    def collect_selfplay_data(self, iteration: int) -> bool:
        """Collect self-play game data"""
        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{self.training_iterations}: Data Collection")
        print("=" * 80)

        iteration_data_dir = self.data_dir / f'iteration_{iteration}'
        iteration_data_dir.mkdir(exist_ok=True)

        args = [
            '--games', str(self.games_per_iteration),
            '--output', str(iteration_data_dir),
            '--save-interval', '10'
        ]

        print("\nIMPORTANT: Make sure UE5 is running with data export enabled!")
        print("Press Enter when ready to start collection...")
        input()

        success = self.run_script('self_play_collector.py', args)

        if success:
            print(f"\nData collection complete: {iteration_data_dir}")

        return success

    def train_models(self, iteration: int) -> bool:
        """Train all models on collected data"""
        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{self.training_iterations}: Model Training")
        print("=" * 80)

        iteration_data_dir = self.data_dir / f'iteration_{iteration}'
        iteration_output_dir = self.models_dir / f'iteration_{iteration}'

        args = [
            '--data-dir', str(iteration_data_dir),
            '--output-dir', str(iteration_output_dir),
            '--iterations', '1',
            '--value-epochs', '50',
            '--world-epochs', '50',
            '--rl-epochs', '50',
            '--batch-size', '64'
        ]

        success = self.run_script('train_coupled_system.py', args)

        if success:
            print(f"\nModels trained: {iteration_output_dir}")

            # Copy models to latest
            latest_dir = self.models_dir / 'latest'
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(iteration_output_dir / 'models', latest_dir)
            print(f"Latest models updated: {latest_dir}")

        return success

    def deploy_models_to_ue5(self) -> bool:
        """Deploy trained models to UE5 project"""
        if not self.ue5_project_path:
            print("\nSkipping UE5 deployment (no project path specified)")
            return True

        print("\n" + "=" * 80)
        print("Deploying Models to UE5")
        print("=" * 80)

        ue5_models_dir = self.ue5_project_path / 'Content' / 'AI' / 'Models'
        if not ue5_models_dir.exists():
            print(f"Creating UE5 models directory: {ue5_models_dir}")
            ue5_models_dir.mkdir(parents=True, exist_ok=True)

        latest_dir = self.models_dir / 'latest'
        if not latest_dir.exists():
            print(f"Error: No trained models found at {latest_dir}")
            return False

        # Copy ONNX models
        onnx_files = list(latest_dir.glob('*.onnx'))
        if not onnx_files:
            print("Warning: No ONNX models found to deploy")
            return False

        for model_file in onnx_files:
            dest = ue5_models_dir / model_file.name
            shutil.copy2(model_file, dest)
            print(f"  Deployed: {model_file.name} -> {dest}")

        print(f"\nModels deployed to UE5: {ue5_models_dir}")
        return True

    def evaluate_agents(self, iteration: int) -> bool:
        """Evaluate trained agents against baseline"""
        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{self.training_iterations}: Agent Evaluation")
        print("=" * 80)

        # Collect evaluation games
        eval_data_dir = self.eval_dir / f'iteration_{iteration}'
        eval_data_dir.mkdir(exist_ok=True)

        print("\nIMPORTANT: Run evaluation games in UE5 (v3.0 vs v2.0)")
        print(f"Export game outcomes to: {eval_data_dir}")
        print("Press Enter when evaluation games are complete...")
        input()

        # Run evaluation script
        eval_output_dir = self.eval_dir / f'iteration_{iteration}_results'

        args = [
            '--data', str(eval_data_dir),
            '--output', str(eval_output_dir),
            '--baseline', 'v2.0',
            '--trained', 'v3.0',
            '--plots'
        ]

        success = self.run_script('evaluate_agents.py', args)

        if success:
            print(f"\nEvaluation complete: {eval_output_dir}")

        return success

    def save_pipeline_state(self):
        """Save pipeline state and logs"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'current_iteration': self.current_iteration,
            'games_per_iteration': self.games_per_iteration,
            'training_iterations': self.training_iterations,
            'total_games_collected': self.current_iteration * self.games_per_iteration,
            'pipeline_log': self.pipeline_log
        }

        state_path = self.output_base_dir / 'pipeline_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"\nPipeline state saved: {state_path}")

    def generate_final_report(self) -> str:
        """Generate final pipeline report"""
        report = []
        report.append("=" * 80)
        report.append("Self-Play Training Pipeline - Final Report")
        report.append("=" * 80)
        report.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Iterations completed: {self.current_iteration}/{self.training_iterations}")
        report.append(f"Total games collected: {self.current_iteration * self.games_per_iteration}")
        report.append("")

        # Summary by stage
        stages = defaultdict(list)
        for entry in self.pipeline_log:
            stages[entry['script']].append(entry)

        report.append("Stage Summary:")
        report.append("-" * 80)

        for script, entries in stages.items():
            successes = sum(1 for e in entries if e['status'] == 'success')
            total_time = sum(e['duration_seconds'] for e in entries)

            report.append(f"\n{script}:")
            report.append(f"  Runs: {len(entries)}")
            report.append(f"  Successes: {successes}/{len(entries)}")
            report.append(f"  Total time: {total_time/3600:.2f} hours")

        report.append("")
        report.append("-" * 80)
        report.append("\nOutput Directories:")
        report.append(f"  Data: {self.data_dir}")
        report.append(f"  Models: {self.models_dir}")
        report.append(f"  Logs: {self.logs_dir}")
        report.append(f"  Evaluation: {self.eval_dir}")

        if self.ue5_project_path:
            report.append(f"\nUE5 Project: {self.ue5_project_path}")
            ue5_models = self.ue5_project_path / 'Content' / 'AI' / 'Models'
            report.append(f"Models deployed to: {ue5_models}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def run(self, skip_collection: bool = False, skip_evaluation: bool = False):
        """Run the complete pipeline"""
        print("\nStarting self-play training pipeline...")
        print(f"Target: {self.training_iterations} iterations, {self.games_per_iteration} games each")
        print("")

        for iteration in range(1, self.training_iterations + 1):
            self.current_iteration = iteration

            print(f"\n{'='*80}")
            print(f"PIPELINE ITERATION {iteration}/{self.training_iterations}")
            print(f"{'='*80}")

            # Stage 1: Data collection
            if not skip_collection:
                if not self.collect_selfplay_data(iteration):
                    print(f"\nError: Data collection failed at iteration {iteration}")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        break

            # Stage 2: Model training
            if not self.train_models(iteration):
                print(f"\nError: Training failed at iteration {iteration}")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    break

            # Stage 3: Deploy to UE5
            if not self.deploy_models_to_ue5():
                print(f"\nWarning: UE5 deployment had issues at iteration {iteration}")

            # Stage 4: Evaluation (optional, not every iteration)
            if not skip_evaluation and (iteration % 2 == 0 or iteration == self.training_iterations):
                if not self.evaluate_agents(iteration):
                    print(f"\nWarning: Evaluation had issues at iteration {iteration}")

            # Save state after each iteration
            self.save_pipeline_state()

            print(f"\n{'='*80}")
            print(f"Iteration {iteration} complete!")
            print(f"{'='*80}")

            # Pause between iterations
            if iteration < self.training_iterations:
                print("\nPreparing for next iteration...")
                time.sleep(2)

        # Final report
        report = self.generate_final_report()
        print("\n" + report)

        report_path = self.output_base_dir / 'pipeline_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nPipeline complete! Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Self-Play Training Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 1000 games over 10 iterations (100 games each)
  python run_selfplay_pipeline.py --games 100 --iterations 10

  # Run with UE5 auto-deployment
  python run_selfplay_pipeline.py --games 100 --iterations 10 \\
      --ue5-project "C:/Projects/SBDAPM"

  # Skip data collection (use existing data)
  python run_selfplay_pipeline.py --games 100 --iterations 5 --skip-collection

  # Skip evaluation (faster iterations)
  python run_selfplay_pipeline.py --games 100 --iterations 10 --skip-evaluation
        """
    )

    parser.add_argument('--games', type=int, default=100,
                        help='Games to collect per iteration (default: 100)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations (default: 10)')

    parser.add_argument('--output', type=str, default='./pipeline_output',
                        help='Base output directory (default: ./pipeline_output)')
    parser.add_argument('--ue5-project', type=str, default=None,
                        help='Path to UE5 project for auto-deployment')

    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection (use existing data)')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation stages')

    args = parser.parse_args()

    # Create pipeline
    pipeline = SelfPlayPipeline(
        games_per_iteration=args.games,
        training_iterations=args.iterations,
        output_base_dir=Path(args.output),
        ue5_project_path=Path(args.ue5_project) if args.ue5_project else None
    )

    # Run pipeline
    try:
        pipeline.run(
            skip_collection=args.skip_collection,
            skip_evaluation=args.skip_evaluation
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        pipeline.save_pipeline_state()
        print("State saved. You can resume later.")


if __name__ == '__main__':
    main()
