"""
StateTree Execution Verification Script

Run this script to parse UE5 Output Log and identify StateTree execution issues.

Usage:
    python VERIFY_STATETREE.py [path_to_output_log.txt]

If no path provided, looks for recent logs in typical UE5 locations.
"""

import re
import sys
from pathlib import Path
from datetime import datetime

class StateTreeDiagnostic:
    def __init__(self, log_path=None):
        self.log_path = log_path
        self.issues = []
        self.warnings = []
        self.successes = []

    def find_recent_log(self):
        """Find most recent UE5 output log"""
        possible_paths = [
            Path.home() / "AppData/Local/UnrealEngine/Common/Saved/Logs/",
            Path("Saved/Logs/"),
        ]

        for base_path in possible_paths:
            if base_path.exists():
                logs = list(base_path.glob("*.log"))
                if logs:
                    return max(logs, key=lambda p: p.stat().st_mtime)
        return None

    def parse_log(self, log_content):
        """Parse log and identify issues"""
        lines = log_content.split('\n')

        # Tracking flags
        has_schola_init = False
        has_statetree_running = False
        has_exec_obj_enter = False
        has_exec_obj_tick = False
        has_schola_action = False
        has_statetree_null_error = False
        has_objective_null_error = False
        has_condition_failure = False

        tick_count = 0

        for line in lines:
            # Check #1: Schola initialization
            if "[SCHOLA INIT]" in line and "Created dummy" in line:
                has_schola_init = True
                self.successes.append("✅ Dummy objective created for Schola training")

            # Check #2: StateTree running status
            if "[STATE TREE]" in line and "Status=Running" in line:
                has_statetree_running = True
                self.successes.append("✅ StateTree is running")

            # Check #3: Task EnterState
            if "[EXEC OBJ]" in line and "ENTER" in line:
                has_exec_obj_enter = True
                # Extract objective type
                obj_match = re.search(r"Objective: (\w+)", line)
                if obj_match:
                    obj_type = obj_match.group(1)
                    if obj_type == "None":
                        has_objective_null_error = True
                        self.issues.append("❌ Task entered but Objective is NULL")
                    else:
                        self.successes.append(f"✅ Task entered with Objective={obj_type}")

            # Check #4: Task Tick
            if "[EXEC OBJ TICK]" in line:
                has_exec_obj_tick = True
                tick_count += 1

            # Check #5: Schola action reception
            if "[SCHOLA ACTION]" in line:
                has_schola_action = True
                self.successes.append("✅ Schola actions being executed")

            # Error detection
            if "StateTreeComp is null" in line:
                has_statetree_null_error = True
                self.issues.append("❌ CRITICAL: StateTreeComp is NULL (binding missing in asset)")

            if "[EXEC OBJ EXIT]" in line and "Objective=NULL" in line:
                has_objective_null_error = True
                self.issues.append("❌ Task exited due to NULL objective")

            if "[CHECK OBJECTIVE TYPE]" in line and "Result=0" in line:
                has_condition_failure = True
                self.issues.append("⚠️ CheckObjectiveType condition failed")

        # Final analysis
        if tick_count > 0:
            self.successes.append(f"✅ Task ticked {tick_count} times")
        else:
            self.issues.append("❌ Task NEVER ticked (EnterState only)")

        # Diagnosis
        if has_statetree_null_error:
            self.issues.append("🔧 FIX: Open StateTree asset, bind StateTreeComp to FollowerStateTreeComponent")

        if has_objective_null_error and not has_schola_init:
            self.issues.append("🔧 FIX: Verify FollowerAgentComponent creates dummy objective in BeginPlay")

        if has_condition_failure:
            self.issues.append("🔧 FIX: Check AcceptedObjectiveTypes includes FormationMove")

        if has_exec_obj_enter and not has_exec_obj_tick:
            self.issues.append("🔧 FIX: Task exits immediately - check for NULL objective or failed conditions")

    def print_report(self):
        """Print diagnostic report"""
        print("\n" + "="*80)
        print("STATETREE DIAGNOSTIC REPORT")
        print("="*80 + "\n")

        if self.successes:
            print("✅ SUCCESSES:")
            for success in self.successes:
                print(f"  {success}")
            print()

        if self.issues:
            print("❌ ISSUES FOUND:")
            for issue in self.issues:
                print(f"  {issue}")
            print()

        if self.warnings:
            print("⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
            print()

        if not self.issues:
            print("✅ ALL CHECKS PASSED - StateTree is functioning correctly!")
        else:
            print(f"\n📋 FOUND {len(self.issues)} ISSUE(S) - See fixes above")
            print("\n📖 For detailed instructions, see: STATETREE_DIAGNOSTIC.md")

        print("\n" + "="*80)

    def run(self):
        """Run diagnostic"""
        if not self.log_path:
            self.log_path = self.find_recent_log()

        if not self.log_path:
            print("❌ ERROR: Could not find UE5 output log")
            print("Please specify log path: python VERIFY_STATETREE.py <path/to/log.txt>")
            return

        print(f"📂 Reading log: {self.log_path}")

        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
        except Exception as e:
            print(f"❌ ERROR: Could not read log file: {e}")
            return

        print("🔍 Analyzing...")
        self.parse_log(log_content)
        self.print_report()

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else None

    diagnostic = StateTreeDiagnostic(log_path)
    diagnostic.run()

if __name__ == "__main__":
    main()
