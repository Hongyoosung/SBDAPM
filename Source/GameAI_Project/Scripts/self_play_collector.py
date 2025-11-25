"""
Self-Play Data Collector for AlphaZero-Inspired Multi-Agent Combat AI

Automates data collection from Unreal Engine self-play games for training:
- RL tactical experiences (observations, actions, rewards)
- MCTS strategic traces (team states, commands, visit counts, outcomes)
- State transitions (current_state, actions, next_state)

Organizes data for all training pipelines:
- train_tactical_policy_v3.py (RL policy + priors)
- train_value_network.py (MCTS value estimation)
- train_world_model.py (state transition prediction)

Usage:
    python self_play_collector.py --games 100 --output ./selfplay_data

Requirements:
    pip install numpy tqdm
"""

import argparse
import json
import socket
import struct
import numpy as np
from datetime import datetime
from pathlib import Path
import time
from collections import defaultdict
from typing import Dict, List, Any
import threading
import queue

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    tqdm = lambda x, **kwargs: x


class MultiChannelCollector:
    """Collects data from multiple UE5 export sockets simultaneously"""

    def __init__(self, ports: Dict[str, int], output_dir: Path):
        """
        Args:
            ports: Dict mapping data type to port number
                   e.g., {'rl': 9997, 'mcts': 9998, 'transitions': 9999}
            output_dir: Directory to save collected data
        """
        self.ports = ports
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data buffers
        self.rl_experiences = []
        self.mcts_traces = []
        self.state_transitions = []
        self.game_outcomes = []

        # Threading
        self.collection_threads = []
        self.data_queues = {key: queue.Queue() for key in ports.keys()}
        self.stop_event = threading.Event()

        # Statistics
        self.stats = defaultdict(int)

    def start_collection(self):
        """Start all data collection threads"""
        for data_type, port in self.ports.items():
            thread = threading.Thread(
                target=self._collection_worker,
                args=(data_type, port),
                daemon=True
            )
            thread.start()
            self.collection_threads.append(thread)
            print(f"Started {data_type} collector on port {port}")

    def _collection_worker(self, data_type: str, port: int):
        """Worker thread for collecting data from a specific socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Non-blocking with timeout

        try:
            sock.bind(('127.0.0.1', port))
            sock.listen(5)
            print(f"[{data_type}] Listening on port {port}")

            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    conn.settimeout(0.1)
                    self._handle_connection(conn, data_type)
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"[{data_type}] Error: {e}")
        finally:
            sock.close()

    def _handle_connection(self, conn: socket.socket, data_type: str):
        """Handle data from a connected client"""
        try:
            while not self.stop_event.is_set():
                # Receive message length (4 bytes)
                length_data = conn.recv(4)
                if not length_data:
                    break

                message_length = struct.unpack('!I', length_data)[0]

                # Receive message
                data = b''
                while len(data) < message_length:
                    packet = conn.recv(min(4096, message_length - len(data)))
                    if not packet:
                        break
                    data += packet

                if len(data) == message_length:
                    item = json.loads(data.decode('utf-8'))
                    self.data_queues[data_type].put(item)
                    self.stats[f'{data_type}_received'] += 1
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[{data_type}] Connection error: {e}")
        finally:
            conn.close()

    def process_queues(self):
        """Process all queued data items"""
        for data_type, q in self.data_queues.items():
            while not q.empty():
                try:
                    item = q.get_nowait()

                    if data_type == 'rl':
                        self.rl_experiences.append(item)
                    elif data_type == 'mcts':
                        self.mcts_traces.append(item)
                    elif data_type == 'transitions':
                        self.state_transitions.append(item)
                    elif data_type == 'outcomes':
                        self.game_outcomes.append(item)

                except queue.Empty:
                    break

    def save_all(self, batch_id: int = 0):
        """Save all collected data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save RL experiences
        if self.rl_experiences:
            rl_path = self.output_dir / f'rl_experiences_{batch_id}_{timestamp}.json'
            self._save_json(rl_path, {
                'metadata': self._get_metadata('rl_experiences'),
                'experiences': self.rl_experiences
            })
            print(f"Saved {len(self.rl_experiences)} RL experiences to {rl_path.name}")

        # Save MCTS traces
        if self.mcts_traces:
            mcts_path = self.output_dir / f'mcts_traces_{batch_id}_{timestamp}.json'
            self._save_json(mcts_path, {
                'metadata': self._get_metadata('mcts_traces'),
                'traces': self.mcts_traces
            })
            print(f"Saved {len(self.mcts_traces)} MCTS traces to {mcts_path.name}")

        # Save state transitions
        if self.state_transitions:
            trans_path = self.output_dir / f'state_transitions_{batch_id}_{timestamp}.json'
            self._save_json(trans_path, {
                'metadata': self._get_metadata('state_transitions'),
                'transitions': self.state_transitions
            })
            print(f"Saved {len(self.state_transitions)} state transitions to {trans_path.name}")

        # Save game outcomes
        if self.game_outcomes:
            outcomes_path = self.output_dir / f'game_outcomes_{batch_id}_{timestamp}.json'
            self._save_json(outcomes_path, {
                'metadata': self._get_metadata('game_outcomes'),
                'outcomes': self.game_outcomes
            })
            print(f"Saved {len(self.game_outcomes)} game outcomes to {outcomes_path.name}")

        # Save statistics
        stats_path = self.output_dir / f'collection_stats_{batch_id}_{timestamp}.json'
        self._save_json(stats_path, {
            'timestamp': timestamp,
            'batch_id': batch_id,
            'counts': {
                'rl_experiences': len(self.rl_experiences),
                'mcts_traces': len(self.mcts_traces),
                'state_transitions': len(self.state_transitions),
                'game_outcomes': len(self.game_outcomes)
            },
            'stats': dict(self.stats)
        })

    def _save_json(self, path: Path, data: Dict):
        """Save data to JSON file"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_metadata(self, data_type: str) -> Dict:
        """Generate metadata for saved data"""
        return {
            'data_type': data_type,
            'collection_date': datetime.now().isoformat(),
            'collector_version': '3.0',
            'system': 'AlphaZero-Inspired Multi-Agent Combat AI'
        }

    def clear_buffers(self):
        """Clear all data buffers"""
        self.rl_experiences.clear()
        self.mcts_traces.clear()
        self.state_transitions.clear()
        self.game_outcomes.clear()

    def stop(self):
        """Stop all collection threads"""
        self.stop_event.set()
        for thread in self.collection_threads:
            thread.join(timeout=2.0)
        self.process_queues()  # Final queue processing

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            'rl_experiences': len(self.rl_experiences),
            'mcts_traces': len(self.mcts_traces),
            'state_transitions': len(self.state_transitions),
            'game_outcomes': len(self.game_outcomes),
            'raw_stats': dict(self.stats)
        }


class SelfPlaySession:
    """Manages a self-play data collection session"""

    def __init__(self, num_games: int, output_dir: Path,
                 save_interval: int = 10, ports: Dict[str, int] = None):
        """
        Args:
            num_games: Number of games to collect
            output_dir: Directory for saving data
            save_interval: Save data every N games
            ports: Port configuration (None for defaults)
        """
        self.num_games = num_games
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval

        # Default port configuration
        if ports is None:
            ports = {
                'rl': 9997,          # RL tactical experiences
                'mcts': 9998,        # MCTS strategic traces
                'transitions': 9999,  # State transitions
                'outcomes': 10000    # Game outcomes
            }

        self.collector = MultiChannelCollector(ports, output_dir)
        self.games_completed = 0
        self.batch_id = 0

    def run(self):
        """Run the self-play collection session"""
        print("=" * 80)
        print("Self-Play Data Collection Session")
        print("=" * 80)
        print(f"Target games: {self.num_games}")
        print(f"Output directory: {self.output_dir}")
        print(f"Save interval: {self.save_interval} games")
        print("=" * 80)
        print("\nWaiting for Unreal Engine connection...")
        print("Start UE5 with data export enabled in TeamLeaderComponent and FollowerAgentComponent")
        print("Press Ctrl+C to stop collection early\n")

        self.collector.start_collection()

        try:
            # Monitor collection progress
            last_save_game = 0

            with tqdm(total=self.num_games, desc="Games collected") as pbar:
                while self.games_completed < self.num_games:
                    time.sleep(1.0)

                    # Process queued data
                    self.collector.process_queues()

                    # Check for game completion (based on outcomes)
                    stats = self.collector.get_stats()
                    current_games = stats['game_outcomes']

                    if current_games > self.games_completed:
                        new_games = current_games - self.games_completed
                        self.games_completed = current_games
                        pbar.update(new_games)

                    # Periodic save
                    if self.games_completed - last_save_game >= self.save_interval:
                        print(f"\n[Checkpoint] Saving batch {self.batch_id}...")
                        self.collector.save_all(self.batch_id)
                        self.collector.clear_buffers()
                        last_save_game = self.games_completed
                        self.batch_id += 1

                        # Print current stats
                        self._print_stats()

            # Final save
            if len(self.collector.rl_experiences) > 0:
                print(f"\n[Final] Saving batch {self.batch_id}...")
                self.collector.save_all(self.batch_id)

            print("\n" + "=" * 80)
            print("Collection Complete!")
            self._print_final_summary()

        except KeyboardInterrupt:
            print("\n\nStopping collection early...")
            self.collector.save_all(self.batch_id)
            self._print_final_summary()

        finally:
            self.collector.stop()

    def _print_stats(self):
        """Print current collection statistics"""
        stats = self.collector.get_stats()
        print(f"  Games: {self.games_completed}/{self.num_games}")
        print(f"  RL experiences: {stats['rl_experiences']}")
        print(f"  MCTS traces: {stats['mcts_traces']}")
        print(f"  State transitions: {stats['state_transitions']}")

    def _print_final_summary(self):
        """Print final collection summary"""
        print("=" * 80)
        print("Final Statistics:")
        print(f"  Total games collected: {self.games_completed}")

        # Count files
        rl_files = list(self.output_dir.glob('rl_experiences_*.json'))
        mcts_files = list(self.output_dir.glob('mcts_traces_*.json'))
        trans_files = list(self.output_dir.glob('state_transitions_*.json'))

        print(f"  RL experience files: {len(rl_files)}")
        print(f"  MCTS trace files: {len(mcts_files)}")
        print(f"  State transition files: {len(trans_files)}")
        print(f"\nData saved to: {self.output_dir}")
        print("\nReady for training! Run:")
        print(f"  python train_coupled_system.py --data-dir {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Self-Play Data Collector for AlphaZero-Inspired Multi-Agent AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 100 games with default settings
  python self_play_collector.py --games 100 --output ./selfplay_data

  # Collect 1000 games, save every 50 games
  python self_play_collector.py --games 1000 --output ./selfplay_data --save-interval 50

  # Custom port configuration
  python self_play_collector.py --games 100 --rl-port 9997 --mcts-port 9998
        """
    )

    parser.add_argument('--games', type=int, default=100,
                        help='Number of self-play games to collect')
    parser.add_argument('--output', type=str, default='./selfplay_data',
                        help='Output directory for collected data')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save data every N games (default: 10)')

    # Port configuration
    parser.add_argument('--rl-port', type=int, default=9997,
                        help='Port for RL experiences (default: 9997)')
    parser.add_argument('--mcts-port', type=int, default=9998,
                        help='Port for MCTS traces (default: 9998)')
    parser.add_argument('--transitions-port', type=int, default=9999,
                        help='Port for state transitions (default: 9999)')
    parser.add_argument('--outcomes-port', type=int, default=10000,
                        help='Port for game outcomes (default: 10000)')

    args = parser.parse_args()

    # Build port configuration
    ports = {
        'rl': args.rl_port,
        'mcts': args.mcts_port,
        'transitions': args.transitions_port,
        'outcomes': args.outcomes_port
    }

    # Create and run session
    session = SelfPlaySession(
        num_games=args.games,
        output_dir=Path(args.output),
        save_interval=args.save_interval,
        ports=ports
    )

    session.run()


if __name__ == '__main__':
    main()
