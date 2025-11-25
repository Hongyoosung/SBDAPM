"""
MCTS Data Collection Script for Value Network Training

Connects to Unreal Engine via socket and collects MCTS rollout data for training.
Exports data in format expected by train_value_network.py

Usage:
    1. In Unreal, enable data export in TeamLeaderComponent
    2. Run: python collect_mcts_data.py --output mcts_rollouts.json
    3. Play games in Unreal to generate data
    4. Press Ctrl+C to stop collection and save

Requirements:
    pip install numpy
"""

import argparse
import json
import socket
import struct
import numpy as np
from datetime import datetime
import os


class MCTSDataCollector:
    """Collects MCTS rollout data from Unreal Engine"""

    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.rollouts = []

    def connect(self):
        """Connect to Unreal Engine data export socket"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print(f"Listening for Unreal Engine on {self.host}:{self.port}...")

    def receive_data(self):
        """Receive and parse MCTS rollout data"""
        conn, addr = self.socket.accept()
        print(f"Connected to {addr}")

        try:
            while True:
                # Receive message length (4 bytes)
                length_data = conn.recv(4)
                if not length_data:
                    break

                message_length = struct.unpack('!I', length_data)[0]

                # Receive message
                data = b''
                while len(data) < message_length:
                    packet = conn.recv(message_length - len(data))
                    if not packet:
                        break
                    data += packet

                if len(data) == message_length:
                    rollout = json.loads(data.decode('utf-8'))
                    self.rollouts.append(rollout)
                    print(f"Collected rollout {len(self.rollouts)} (outcome: {rollout.get('outcome', 'N/A')})")

        except KeyboardInterrupt:
            print("\nStopping collection...")
        finally:
            conn.close()

    def save(self, output_path):
        """Save collected rollouts to JSON"""
        data = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'num_rollouts': len(self.rollouts),
                'host': self.host,
                'port': self.port
            },
            'rollouts': self.rollouts
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(self.rollouts)} rollouts to {output_path}")

        # Print statistics
        if self.rollouts:
            outcomes = [r.get('outcome', 0.0) for r in self.rollouts]
            wins = sum(1 for o in outcomes if o > 0.5)
            losses = sum(1 for o in outcomes if o < -0.5)
            draws = len(outcomes) - wins - losses

            print(f"Statistics:")
            print(f"  Wins: {wins} ({100*wins/len(outcomes):.1f}%)")
            print(f"  Losses: {losses} ({100*losses/len(outcomes):.1f}%)")
            print(f"  Draws: {draws} ({100*draws/len(outcomes):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Collect MCTS rollout data from Unreal Engine')
    parser.add_argument('--output', type=str, default='mcts_rollouts.json',
                        help='Output JSON file path')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Socket host address')
    parser.add_argument('--port', type=int, default=9999,
                        help='Socket port')

    args = parser.parse_args()

    print("=" * 60)
    print("MCTS Data Collector")
    print("=" * 60)
    print("Instructions:")
    print("  1. Start this script")
    print("  2. Launch Unreal Engine with data export enabled")
    print("  3. Play games to collect data")
    print("  4. Press Ctrl+C to stop and save")
    print("=" * 60)

    collector = MCTSDataCollector(args.host, args.port)
    collector.connect()
    collector.receive_data()
    collector.save(args.output)

    print(f"\nReady for training! Run:")
    print(f"  python train_value_network.py --data {args.output}")


if __name__ == '__main__':
    main()
