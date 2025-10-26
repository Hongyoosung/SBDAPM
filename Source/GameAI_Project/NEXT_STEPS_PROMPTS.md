# SBDAPM: Next Implementation Steps - Detailed Prompts

This document provides ready-to-use prompts for implementing the improvements outlined in FINAL_METHODOLOGY.md. Use these prompts sequentially to guide the development process.

---

## Phase 1: Foundation & Code Quality (Weeks 1-2)

### Step 2.1: Expand Observation Space

**Prompt:**
```
I need to expand the observation space in SBDAPM from 3 features to 60-80 features as outlined in FINAL_METHODOLOGY.md (lines 84-122).

Current ObservationElement has only:
- AgentHealth
- DistanceToDestination
- EnemiesNum

Please implement the enhanced observation system with:

1. Agent State (10 features): Position, Velocity, Rotation, Health, Stamina, Shield
2. Combat State (5 features): WeaponCooldown, Ammunition, CurrentWeapon
3. Environment Perception (30 features):
   - RaycastDistances (16 rays, 360° coverage)
   - RaycastHitTypes (object type detection)
4. Enemy Information (15 features):
   - VisibleEnemyCount
   - NearbyEnemies (Top 5 closest with positions, health, distance)
5. Tactical Context (10 features):
   - bHasCover
   - NearestCoverPosition
   - CurrentTerrain
6. Temporal Features (5 features):
   - TimeSinceLastAction
   - LastActionTaken

Files to modify:
- Public/Core/ObservationElement.h
- Private/Core/ObservationElement.cpp
- Public/Core/StateMachine.h (update GetObservation function)
- Private/Core/StateMachine.cpp

Please provide the complete implementation with proper UPROPERTY declarations for Blueprint exposure.
```

---

### Step 2.2: Complete FleeState Implementation

**Prompt:**
```
The FleeState is currently a stub implementation. I need to fully implement it following the same pattern as MoveToState and AttackState.

Requirements:
1. Integrate MCTS for decision making
2. Define flee-specific actions:
   - Sprint away from enemies
   - Move toward cover
   - Use evasive movements (zigzag pattern)
   - Call for help (optional)
3. Implement state transition logic:
   - Enter when: Health < 20% AND EnemiesNum > 3
   - Exit when: Safe distance reached OR health restored > 40%

Files to modify:
- Public/States/FleeState.h
- Private/States/FleeState.cpp
- Create new action files if needed (e.g., SprintAction.h/cpp, MoveToCoverAction.h/cpp)

Reference the implementation patterns in:
- Private/States/MoveToState.cpp (lines 10-59)
- Private/States/AttackState.cpp

Please provide complete implementation with MCTS integration and proper action execution.
```

---

### Step 2.3: Add Configurable MCTS Parameters

**Prompt:**
```
Currently, MCTS parameters are hardcoded in Private/AI/MCTS.cpp. I need to expose these as UPROPERTY for runtime tuning.

Parameters to expose:
- ExplorationParameter (currently 1.41)
- DiscountFactor (currently 0.95)
- MaxTreeDepth (currently 10)
- SimulationCount (number of MCTS iterations)

Requirements:
1. Add UPROPERTY(EditAnywhere, Category="MCTS") to Public/AI/MCTS.h
2. Provide sensible default values
3. Add range metadata (e.g., ClampMin, ClampMax) where appropriate
4. Add tooltips explaining each parameter's effect

Files to modify:
- Public/AI/MCTS.h
- Private/AI/MCTS.cpp (use member variables instead of hardcoded values)

Please implement these changes following Unreal's UPROPERTY best practices.
```

---

### Step 2.4: Add Unit Tests for Core Components

**Prompt:**
```
I need to add unit tests for the SBDAPM core components using Unreal's Automation Framework.

Create tests for:
1. MCTS Algorithm:
   - Test UCT selection logic
   - Test tree expansion
   - Test backpropagation
   - Test observation similarity calculation

2. StateMachine:
   - Test state transitions
   - Test observation updates
   - Test GetCurrentState functionality

3. ObservationElement:
   - Test feature normalization (if implemented)
   - Test struct serialization

Test file structure:
- Create Private/Tests/ directory
- MCTSTest.cpp
- StateMachineTest.cpp
- ObservationElementTest.cpp

Please provide complete test implementations using IMPLEMENT_SIMPLE_AUTOMATION_TEST and related macros. Target at least 80% code coverage for tested components.
```

---

## Phase 2: Neural Network Integration (Weeks 3-4)

### Step 3.1: Design Neural Network Architecture

**Prompt:**
```
I need to implement a neural network policy for SBDAPM that works alongside MCTS (AlphaZero-style).

Architecture (from FINAL_METHODOLOGY.md lines 130-142):
- Input: 80 features (from expanded ObservationElement)
- Hidden Layer 1: 256 units, ReLU activation
- Hidden Layer 2: 256 units, ReLU activation
- Output Heads:
  * Policy Head: Softmax over 15 actions
  * Value Head: Tanh output [-1, 1]

Requirements:
1. Create new files:
   - Public/AI/NeuralNetwork.h
   - Private/AI/NeuralNetwork.cpp

2. Choose integration approach:
   Option A: Unreal's LearningAgents plugin (recommended)
   Option B: PyTorch C++ API (libtorch)
   Option C: ONNX Runtime for inference

3. Implement:
   - Network initialization
   - Forward pass (observation → policy + value)
   - Model loading/saving
   - Inference optimization (batching if needed)

Please provide the complete neural network implementation with clear comments explaining the architecture. Include instructions for linking any required libraries (libtorch or ONNX Runtime).
```

---

### Step 3.2: Implement Hybrid MCTS + Neural Network Policy

**Prompt:**
```
I need to integrate the neural network with MCTS to create a hybrid policy (AlphaZero-style).

Decision process (FINAL_METHODOLOGY.md lines 144-153):
1. Neural network provides prior probabilities for each action
2. MCTS refines these priors through tree search (100-1000 simulations)
3. Best action selected from refined distribution

Requirements:
1. Modify MCTS to accept neural network priors
2. Update MCTSNode to store:
   - Prior probability (from neural network)
   - Visit count
   - Total reward
   - Q-value

3. Update UCT formula to use priors:
   ```
   UCT = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
   ```

4. Create HybridPolicy class:
   - Public/AI/HybridPolicy.h
   - Private/AI/HybridPolicy.cpp
   - Combines NN and MCTS

Files to modify:
- Public/AI/MCTS.h
- Private/AI/MCTS.cpp
- Public/AI/MCTSNode.h
- Private/AI/MCTSNode.cpp

Please implement the hybrid policy with proper integration between the neural network and MCTS components.
```

---

### Step 3.3: Implement Experience Replay Buffer

**Prompt:**
```
I need to create an experience replay buffer for storing training data.

Requirements:
1. Create new files:
   - Public/Training/ReplayBuffer.h
   - Private/Training/ReplayBuffer.cpp

2. Store experiences as tuples:
   - Observation (FObservationElement)
   - Action taken (UAction*)
   - Reward received (float)
   - Next observation (FObservationElement)
   - Done flag (bool)
   - MCTS policy (TArray<float> - action probabilities)

3. Implement methods:
   - AddExperience(...)
   - SampleBatch(int32 BatchSize) → Random sampling
   - GetSize() → Current buffer size
   - Clear() → Reset buffer
   - SaveToFile(FString Path) → Serialize to disk
   - LoadFromFile(FString Path) → Deserialize from disk

4. Configuration:
   - MaxBufferSize (UPROPERTY, default 100,000)
   - Sampling strategy (uniform random)

Please implement a thread-safe replay buffer with efficient memory management.
```

---

## Phase 3: Distributed Training Infrastructure (Weeks 5-7)

### Step 4.1: Create RLlib Environment Wrapper

**Prompt:**
```
I need to create a Python wrapper that allows Ray RLlib to interact with the Unreal Engine SBDAPM environment.

Requirements:
1. Communication method: Socket-based (TCP/IP) or gRPC
2. Python side:
   - Implement gym.Env interface
   - Connect to Unreal Engine instance
   - Send/receive observations, actions, rewards

3. Unreal side:
   - Create Public/Training/RLlibInterface.h
   - Create Private/Training/RLlibInterface.cpp
   - Implement socket server
   - Handle action requests
   - Send observation + reward responses

4. Message protocol:
   ```json
   Request:  {"action": int, "reset": bool}
   Response: {"observation": [floats], "reward": float, "done": bool}
   ```

Files to create:
- Python: `training/unreal_env.py`
- C++: Public/Training/RLlibInterface.h
- C++: Private/Training/RLlibInterface.cpp

Please provide complete implementation with error handling and connection management.
```

---

### Step 4.2: Create Docker Compose Setup for Local Training

**Prompt:**
```
I need to create a Docker Compose configuration for local distributed training.

Requirements (FINAL_METHODOLOGY.md lines 271-285):
1. Services:
   - ray-head: Ray cluster coordinator
   - unreal-worker: 8 parallel Unreal Engine instances
   - trainer: GPU trainer container (CUDA support)

2. Create files:
   - docker-compose.yml
   - Dockerfile.unreal (Unreal Engine headless build)
   - Dockerfile.trainer (Python + PyTorch + RLlib)
   - training/train.py (RLlib training script)

3. Configuration:
   - Shared volume for model checkpoints
   - Network configuration for inter-service communication
   - GPU passthrough for trainer service
   - Resource limits (CPU, memory)

4. Training script should:
   - Initialize Ray cluster
   - Configure PPO algorithm
   - Connect to Unreal workers
   - Train for specified iterations
   - Save checkpoints every 10 iterations

Please provide complete Docker setup with clear README instructions for running local training.
```

---

### Step 4.3: Implement Curriculum Learning

**Prompt:**
```
I need to implement curriculum learning for progressive difficulty training (FINAL_METHODOLOGY.md lines 212-224).

Requirements:
1. Create Public/Training/CurriculumManager.h
2. Create Private/Training/CurriculumManager.cpp

3. Define 5 difficulty levels:
   Level 1: 0 enemies, 500m distance, Flat terrain → 80% success
   Level 2: 1-2 enemies, 1000m, Flat → 80% success
   Level 3: 3-5 enemies, 1500m, Moderate → 80% success
   Level 4: 5-10 enemies, 2000m, Complex → 80% success
   Level 5: 10-20 enemies, 3000m, Very Complex → 80% success

4. Implement:
   - GetCurrentLevel() → Current difficulty
   - UpdatePerformance(bool Success) → Track win rate
   - ShouldAdvanceLevel() → Check if ready for next level
   - ConfigureEnvironment() → Apply current level settings

5. Integration:
   - Modify StateMachine to use CurriculumManager
   - Expose level config to Blueprint

Please implement the curriculum manager with automatic progression based on success rate.
```

---

## Phase 4: Cloud Infrastructure (Weeks 8-10)

### Step 5.1: Create AWS SageMaker Training Pipeline

**Prompt:**
```
I need to set up AWS SageMaker for cloud-based training (FINAL_METHODOLOGY.md lines 289-305).

Requirements:
1. Create SageMaker training script:
   - training/sagemaker_train.py
   - Use ml.p3.8xlarge instances (4x V100 GPUs)
   - Configure 32 parallel workers

2. S3 Integration:
   - Model checkpoints → s3://sbdapm-models/
   - Training logs → s3://sbdapm-logs/
   - Replay buffer → s3://sbdapm-data/

3. Create files:
   - sagemaker/train.py (entry point)
   - sagemaker/requirements.txt
   - sagemaker/Dockerfile
   - scripts/launch_sagemaker.py (deployment script)

4. Features:
   - Automatic checkpointing every 10 iterations
   - TensorBoard logging
   - Spot instance support for cost savings
   - Automatic model versioning

Please provide complete SageMaker setup with deployment instructions and cost estimation.
```

---

### Step 5.2: Implement MLflow Model Registry

**Prompt:**
```
I need to set up MLflow for experiment tracking and model versioning (FINAL_METHODOLOGY.md lines 311-330).

Requirements:
1. MLflow setup:
   - Tracking server (local or AWS hosted)
   - Artifact storage (S3)
   - Metrics logging

2. Track experiments:
   - Hyperparameters (learning rate, discount factor, etc.)
   - Metrics (episode reward, win rate, loss)
   - Model artifacts (PyTorch checkpoint files)
   - Environment config (Unreal version, map, difficulty)

3. Create utilities:
   - training/mlflow_utils.py
     * log_params(config_dict)
     * log_metrics(step, metrics_dict)
     * log_model(model, model_name)
     * load_best_model(experiment_id)

4. Integration:
   - Modify training scripts to use MLflow
   - Create dashboard for comparing experiments
   - Implement model promotion workflow

Please provide complete MLflow setup with example training integration.
```

---

### Step 5.3: Create Model Deployment Pipeline

**Prompt:**
```
I need to create a deployment pipeline for pushing trained models to production (FINAL_METHODOLOGY.md lines 333-346).

Requirements:
1. Deployment options:
   Option A: Local Inference (in-engine)
   - Export PyTorch model to ONNX
   - Load ONNX in C++ using ONNX Runtime
   - Target latency: <10ms

   Option B: SageMaker Endpoint
   - Deploy model to SageMaker inference endpoint
   - Implement C++ client for API calls
   - Fallback to local inference if network fails

2. Create files:
   - deployment/export_onnx.py (PyTorch → ONNX conversion)
   - deployment/deploy_sagemaker.py (SageMaker deployment)
   - Public/Inference/ModelInference.h
   - Private/Inference/ModelInference.cpp

3. Features:
   - Model versioning (semantic versioning)
   - A/B testing support (10% traffic to new model)
   - Automatic rollback on performance degradation
   - Monitoring (latency, throughput, accuracy)

Please provide complete deployment pipeline with both local and cloud inference options.
```

---

## Phase 5: Advanced Features (Weeks 11-12)

### Step 6.1: Implement Inverse Reinforcement Learning

**Prompt:**
```
I need to implement IRL for learning from human demonstrations (FINAL_METHODOLOGY.md lines 253-264).

Requirements:
1. Recording system:
   - Capture human gameplay (observations, actions, rewards)
   - Save to replay buffer format
   - Minimum 100 episodes for training

2. IRL algorithm (choose one):
   Option A: Maximum Entropy IRL
   Option B: Generative Adversarial Imitation Learning (GAIL)

3. Create files:
   - training/irl_trainer.py
   - training/human_demo_recorder.py
   - Public/Training/DemonstrationRecorder.h

4. Workflow:
   - Record expert demonstrations
   - Train reward function to explain expert behavior
   - Use learned reward for RL training
   - Fine-tune with traditional RL

Please implement IRL system with demonstration recording and reward learning.
```

---

### Step 6.2: Add MCTS Visualization Tools

**Prompt:**
```
I need to create visualization tools for debugging and understanding MCTS decisions.

Requirements:
1. In-engine visualization:
   - Draw MCTS tree structure in viewport
   - Color code nodes by Q-value (green = good, red = bad)
   - Show visit counts as node sizes
   - Highlight selected path

2. Web-based dashboard:
   - Export tree to JSON
   - Create interactive D3.js visualization
   - Show action probabilities
   - Compare MCTS vs NN priors

3. Create files:
   - Public/Debug/MCTSVisualizer.h
   - Private/Debug/MCTSVisualizer.cpp
   - visualization/tree_viewer.html
   - visualization/tree_viewer.js

4. Features:
   - Real-time updates during gameplay
   - Pause and inspect tree state
   - Export snapshots for analysis
   - Compare trees across different observations

Please implement MCTS visualization with both in-engine and web-based options.
```

---

## Phase 6: Production Hardening (Weeks 13-14)

### Step 7.1: Performance Optimization

**Prompt:**
```
I need to optimize the system to achieve <10ms inference latency (FINAL_METHODOLOGY.md lines 561-565).

Optimization targets:
1. Neural Network:
   - Quantize model (FP32 → INT8) using ONNX Runtime
   - Optimize for inference (fuse layers, remove dropout)
   - Batch predictions if possible

2. MCTS:
   - Implement parallel tree search (multi-threading)
   - Prune unpromising branches early
   - Cache observation similarity calculations
   - Limit tree depth dynamically based on time budget

3. Profiling:
   - Use Unreal Insights to identify bottlenecks
   - Measure frame time impact
   - Profile memory allocations

4. Async execution:
   - Run MCTS on background thread
   - Use FRunnable for continuous tree search
   - Thread-safe communication with game thread

Files to optimize:
- Private/AI/MCTS.cpp
- Private/AI/NeuralNetwork.cpp
- Add Public/Async/AsyncMCTS.h

Please provide optimized implementations with profiling results showing latency improvements.
```

---

### Step 7.2: Set Up Monitoring and Alerting

**Prompt:**
```
I need to set up production monitoring for the trained model (FINAL_METHODOLOGY.md lines 350-386).

Requirements:
1. Metrics to track:
   - Inference latency (p50, p95, p99)
   - Win rate vs baseline
   - Episode length
   - Model version deployed
   - Error rate
   - CPU/GPU utilization

2. Monitoring stack:
   - Prometheus (metrics collection)
   - Grafana (dashboards)
   - Alertmanager (alerts)

3. Create:
   - monitoring/prometheus.yml
   - monitoring/grafana_dashboards/*.json
   - monitoring/alerts.yml
   - Public/Telemetry/MetricsCollector.h

4. Alerts:
   - Latency > 10ms for 5 minutes
   - Win rate drops > 10% from baseline
   - Error rate > 1%
   - Model serving failures

Please provide complete monitoring setup with Grafana dashboards and alert rules.
```

---

### Step 7.3: Create Comprehensive Documentation

**Prompt:**
```
I need to create complete documentation for the SBDAPM system.

Documentation to create:
1. API Documentation:
   - Doxygen configuration
   - Generate C++ API docs
   - Document all public classes and methods

2. User Guides:
   - docs/GETTING_STARTED.md (setup, building, running)
   - docs/TRAINING_GUIDE.md (local and cloud training)
   - docs/DEPLOYMENT_GUIDE.md (deploying models)
   - docs/CONFIGURATION.md (parameters, tuning)

3. Architecture Documentation:
   - docs/ARCHITECTURE.md (system overview)
   - docs/METHODOLOGY.md (RL approach, algorithms)
   - docs/API_REFERENCE.md (generated from Doxygen)

4. Tutorials:
   - docs/tutorials/01_basic_training.md
   - docs/tutorials/02_curriculum_learning.md
   - docs/tutorials/03_cloud_deployment.md
   - docs/tutorials/04_monitoring.md

Please create comprehensive documentation following best practices for technical documentation.
```

---

## Verification & Testing

### Step 8: End-to-End Testing

**Prompt:**
```
I need to create end-to-end tests to verify the entire SBDAPM pipeline.

Test scenarios:
1. Local Training E2E:
   - Start Unreal workers
   - Run training for 100 iterations
   - Verify checkpoint saved
   - Load checkpoint and test inference
   - Measure win rate improvement

2. Cloud Training E2E:
   - Launch SageMaker job
   - Monitor training progress
   - Verify S3 artifacts uploaded
   - Deploy to inference endpoint
   - Test endpoint response time

3. Deployment E2E:
   - Export trained model to ONNX
   - Load in Unreal Engine
   - Run 100 evaluation episodes
   - Verify latency < 10ms
   - Compare performance to baseline

Create:
- tests/e2e_local_training.py
- tests/e2e_cloud_training.py
- tests/e2e_deployment.py
- CI/CD pipeline (.github/workflows/test.yml)

Please provide complete E2E test suite with automated CI/CD integration.
```

---

## Quick Reference: File Creation Checklist

### Phase 1 (Foundation)
- [ ] Public/Core/ObservationElement.h (expanded)
- [ ] Public/States/FleeState.h (complete implementation)
- [ ] Private/Tests/MCTSTest.cpp
- [ ] Private/Tests/StateMachineTest.cpp

### Phase 2 (Neural Networks)
- [ ] Public/AI/NeuralNetwork.h
- [ ] Private/AI/NeuralNetwork.cpp
- [ ] Public/AI/HybridPolicy.h
- [ ] Private/AI/HybridPolicy.cpp
- [ ] Public/Training/ReplayBuffer.h
- [ ] Private/Training/ReplayBuffer.cpp

### Phase 3 (Distributed Training)
- [ ] Public/Training/RLlibInterface.h
- [ ] Public/Training/CurriculumManager.h
- [ ] training/unreal_env.py
- [ ] docker-compose.yml
- [ ] training/train.py

### Phase 4 (Cloud Infrastructure)
- [ ] sagemaker/train.py
- [ ] training/mlflow_utils.py
- [ ] deployment/export_onnx.py
- [ ] Public/Inference/ModelInference.h

### Phase 5 (Advanced Features)
- [ ] training/irl_trainer.py
- [ ] Public/Debug/MCTSVisualizer.h
- [ ] visualization/tree_viewer.html

### Phase 6 (Production)
- [ ] Public/Async/AsyncMCTS.h
- [ ] Public/Telemetry/MetricsCollector.h
- [ ] monitoring/prometheus.yml
- [ ] docs/ARCHITECTURE.md

---

## Usage Instructions

1. **Sequential Execution**: Work through prompts in order for dependencies
2. **Parallel Execution**: Within each phase, some tasks can be done in parallel
3. **Testing**: Run tests after each phase before proceeding
4. **Documentation**: Update docs as you implement features

## Estimated Timeline

- Phase 1: 2 weeks
- Phase 2: 2 weeks
- Phase 3: 3 weeks
- Phase 4: 3 weeks
- Phase 5: 2 weeks
- Phase 6: 2 weeks

**Total: 14 weeks (3.5 months)**

---

**Generated:** 2025-10-26
**Based on:** FINAL_METHODOLOGY.md
**Status:** Ready for Implementation
