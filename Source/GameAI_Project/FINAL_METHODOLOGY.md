# SBDAPM: Final Methodology Summary

## Executive Summary

This document presents the **final recommended methodology** for the SBDAPM (State-Based Dynamic Action Planning Model) project, consolidating insights from current implementation analysis and proposed improvements.

---

## System Overview

**SBDAPM** is a hybrid AI system for Unreal Engine 5.6 that combines:
- **Finite State Machines (FSM)** for behavior organization
- **Monte Carlo Tree Search (MCTS)** for action planning
- **Deep Reinforcement Learning (DRL)** for policy optimization
- **Distributed Training Infrastructure** for scalability

---

## Final Architecture

### High-Level System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                         GAME RUNTIME                              │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Agent Controller (C++ Component)              │  │
│  │                                                            │  │
│  │  ┌──────────────┐           ┌──────────────────┐         │  │
│  │  │ State Machine│◄─────────►│  Neural Network  │         │  │
│  │  │     (FSM)    │           │   Policy + Value │         │  │
│  │  └──────┬───────┘           └────────┬─────────┘         │  │
│  │         │                             │                   │  │
│  │         ▼                             ▼                   │  │
│  │  ┌─────────────────────────────────────────────┐         │  │
│  │  │     MCTS Engine (Planning Module)           │         │  │
│  │  │  - Uses NN for prior probabilities          │         │  │
│  │  │  - Refines actions through tree search      │         │  │
│  │  │  - Returns best action                      │         │  │
│  │  └─────────────────┬───────────────────────────┘         │  │
│  │                    │                                      │  │
│  │                    ▼                                      │  │
│  │  ┌─────────────────────────────────────────────┐         │  │
│  │  │         Action Execution                    │         │  │
│  │  │  Movement | Combat | Tactical               │         │  │
│  │  └─────────────────────────────────────────────┘         │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Observations & Rewards
                              │
┌──────────────────────────────────────────────────────────────────┐
│                      TRAINING INFRASTRUCTURE                      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Ray RLlib Cluster                         │  │
│  │                                                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │  │
│  │  │Worker 1 │  │Worker 2 │  │Worker N │  │Parameter     │ │  │
│  │  │(Unreal) │  │(Unreal) │  │(Unreal) │  │Server (GPU)  │ │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬───────┘ │  │
│  │       └────────────┼────────────┼───────────────┘         │  │
│  │                    ▼            ▼                          │  │
│  │              ┌──────────────────────┐                      │  │
│  │              │  Experience Buffer   │                      │  │
│  │              │  (Replay Memory)     │                      │  │
│  │              └──────────────────────┘                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              AWS SageMaker (Optional)                      │  │
│  │  - Elastic GPU training                                    │  │
│  │  - Model registry & versioning                             │  │
│  │  - Inference endpoints                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Enhanced Observation System

**Current:** 3 features (Health, Distance, Enemies)
**Improved:** 60-80 features

```cpp
USTRUCT(BlueprintType)
struct FEnhancedObservation
{
    // Agent State (10 features)
    FVector Position, Velocity;
    FRotator Rotation;
    float Health, Stamina, Shield;

    // Combat State (5 features)
    float WeaponCooldown;
    int32 Ammunition;
    EWeaponType CurrentWeapon;

    // Environment Perception (30 features)
    TArray<float> RaycastDistances;      // 16 rays, 360° coverage
    TArray<EObjectType> RaycastHitTypes;

    // Enemy Information (15 features)
    int32 VisibleEnemyCount;
    TArray<FEnemyInfo> NearbyEnemies;    // Top 5 closest

    // Tactical Context (10 features)
    bool bHasCover;
    FVector NearestCoverPosition;
    ETerrainType CurrentTerrain;

    // Temporal Features (5 features)
    float TimeSinceLastAction;
    EActionType LastActionTaken;
};
```

**Benefit:** Rich state representation enables intelligent decision-making.

---

### 2. Hybrid MCTS + Neural Network Policy

**Approach:** Combine tree search (MCTS) with deep learning (neural networks) for optimal performance.

#### Neural Network Architecture
```
Input Layer (80 features)
    ↓
Dense (256 units, ReLU)
    ↓
Dense (256 units, ReLU)
    ↓
┌─────────────────────┬──────────────────┐
│   Policy Head       │   Value Head     │
│   Softmax(Actions)  │   Tanh([-1,1])   │
└─────────────────────┴──────────────────┘
```

#### Decision Process
1. **Neural network** provides prior probabilities for each action
2. **MCTS** refines these priors through tree search (100-1000 simulations)
3. **Best action** selected from refined search distribution

**Benefits:**
- **Fast:** Neural network provides quick baseline (~2ms)
- **Strong:** MCTS refines decisions through lookahead (~8ms for 100 sims)
- **Learnable:** Network improves from MCTS expertise (knowledge distillation)

**Inspiration:** AlphaZero (Chess, Go, Shogi)

---

### 3. Finite State Machine (FSM)

**States:**
- **MoveToState** - Navigate to destination
- **AttackState** - Engage enemies
- **FleeState** - Escape from danger (NEW: fully implemented)
- **DefendState** - Hold position, defensive posture (NEW)
- **DeadState** - Terminal state

**Transitions:**
```cpp
// Example transition logic
void UStateMachine::UpdateStateTransitions()
{
    if (AgentHealth < 0.2f && EnemiesNum > 3)
        ChangeState(FleeState);  // Low health + many enemies → Flee

    else if (EnemiesNum > 0 && DistanceToEnemy < 500.0f)
        ChangeState(AttackState);  // Enemies nearby → Attack

    else if (DistanceToDestination > 100.0f)
        ChangeState(MoveToState);  // Far from goal → Move

    else
        ChangeState(DefendState);  // At goal → Defend
}
```

**Improvement:** All states fully implemented with complete action sets.

---

### 4. Action System

**Categories:**

| Category | Actions |
|----------|---------|
| **Movement** | Forward, Backward, Left, Right, Sprint |
| **Combat** | Melee Attack, Skill Attack, Ranged Attack |
| **Tactical** | Take Cover, Use Item, Reload |
| **Defensive** | Block, Dodge, Retreat |

**Total Actions:** 15 discrete actions (vs 6 currently)

**Implementation:**
- C++ base implementations with physics/logic
- Blueprint overrides for game-specific customization
- Networked for multiplayer (optional)

---

## Training Methodology

### Phase 1: Curriculum Learning

Progressive difficulty training:

| Level | Enemies | Distance | Terrain | Success Threshold |
|-------|---------|----------|---------|-------------------|
| 1 | 0 | 500m | Flat | 80% |
| 2 | 1-2 | 1000m | Flat | 80% |
| 3 | 3-5 | 1500m | Moderate | 80% |
| 4 | 5-10 | 2000m | Complex | 80% |
| 5 | 10-20 | 3000m | Very Complex | 80% |

**Agent progresses to next level only after mastering current level.**

---

### Phase 2: Self-Play Training

**Algorithm:** Proximal Policy Optimization (PPO)
- **Advantage:** Stable, sample-efficient, widely used
- **Hyperparameters:**
  - Learning rate: 0.0003
  - Discount factor (γ): 0.99
  - GAE lambda (λ): 0.95
  - Clip parameter (ε): 0.2
  - Minibatch size: 128
  - Training epochs per batch: 30

**Training Loop:**
```
For each iteration:
    1. Collect 4000 timesteps from 8 parallel workers
    2. Compute advantages using GAE
    3. Update policy network (30 epochs, minibatch size 128)
    4. Update value network
    5. Log metrics (reward, loss, KL divergence)
    6. Save checkpoint every 10 iterations
```

---

### Phase 3: Fine-Tuning with Human Demonstrations

**Inverse Reinforcement Learning (IRL):**
1. Record expert human gameplay (100+ episodes)
2. Learn reward function that explains expert behavior
3. Use learned reward for additional RL training

**Behavior Cloning (BC):**
1. Supervised learning: predict expert actions given observations
2. Initialize policy with BC, then refine with RL

**Benefit:** Jump-start training with human knowledge.

---

## Distributed Training Infrastructure

### Local Training (Development)

**Docker Compose Setup:**
```yaml
services:
  ray-head:        # Ray cluster coordinator
  unreal-worker:   # 8 parallel Unreal instances
  trainer:         # GPU trainer (RTX 4090)
```

**Capacity:**
- 8 parallel workers
- 400 samples/second
- 42 minutes to 1M timesteps

**Cost:** Hardware investment (~$3,600 one-time)

---

### Cloud Training (Production)

**AWS SageMaker Setup:**
- Instance: `ml.p3.8xlarge` (4x V100 GPUs)
- Workers: 32 parallel Unreal instances
- Throughput: 1,600 samples/second
- Time to 1M steps: **10 minutes**

**Cost:** $12.24/hour (use Spot Instances for 70% savings)

**Workflow:**
1. Launch SageMaker training job
2. Parallel workers collect experience
3. Parameter server updates policy on GPU
4. Model automatically saved to S3 every 10 iterations
5. Best model deployed to inference endpoint

---

## Deployment Pipeline

### Model Versioning

**MLflow Integration:**
- Track all experiments (hyperparameters, metrics, artifacts)
- Version models automatically
- Compare performance across experiments

**S3 Model Registry:**
```
s3://sbdapm-models/
├── experiment-001/
│   ├── checkpoints/
│   │   ├── iteration_10.pt
│   │   ├── iteration_20.pt
│   │   └── best_model.pt
│   ├── config.yaml
│   └── metrics.json
├── experiment-002/
└── production/
    └── current_model.pt  # Symlink to best model
```

---

### Inference Deployment

**Option 1: Local Inference (Unreal)**
- Neural network runs in-engine (C++ or PyTorch Mobile)
- Latency: 2-8ms
- No network dependency

**Option 2: SageMaker Endpoint**
- Model hosted on AWS
- Latency: 15ms (includes network)
- Scalable, auto-updates

**Recommendation:** Local for production games, SageMaker for evaluation/testing.

---

## Monitoring and Evaluation

### Training Metrics

**Logged Every Iteration:**
- `episode_reward_mean` - Average reward per episode
- `episode_len_mean` - Average episode length
- `policy_loss` - Policy gradient loss
- `value_loss` - Value function loss
- `entropy` - Policy entropy (exploration measure)
- `kl_divergence` - Policy change magnitude

**Visualized in TensorBoard:**
```bash
tensorboard --logdir=./ray_results
```

---

### Evaluation Protocol

**Periodic Evaluation (Every 50 Iterations):**
1. Freeze policy (no exploration)
2. Run 100 evaluation episodes
3. Measure:
   - Win rate
   - Average reward
   - Average time to goal
   - Survival rate
4. Compare to previous best
5. Promote to production if better

**A/B Testing:**
- Deploy new model to 10% of users
- Monitor performance vs current model
- Full rollout if metrics improve

---

## Code Organization (Refactored)

### New Directory Structure

```
Source/GameAI_Project/
├── Public/                          # Public API headers
│   ├── Core/
│   │   ├── StateMachine.h
│   │   └── ObservationElement.h
│   ├── AI/
│   │   ├── MCTS.h
│   │   ├── MCTSNode.h
│   │   ├── NeuralNetwork.h
│   │   └── HybridPolicy.h
│   ├── States/
│   │   ├── State.h
│   │   ├── MoveToState.h
│   │   ├── AttackState.h
│   │   ├── FleeState.h
│   │   └── DefendState.h
│   ├── Actions/
│   │   ├── Action.h
│   │   ├── MovementAction.h
│   │   ├── CombatAction.h
│   │   └── TacticalAction.h
│   └── Training/
│       ├── ReplayBuffer.h
│       ├── CurriculumManager.h
│       └── RLlibInterface.h
│
└── Private/                         # Implementation files
    ├── Core/
    │   ├── StateMachine.cpp
    │   └── ObservationElement.cpp
    ├── AI/
    │   ├── MCTS.cpp
    │   ├── MCTSNode.cpp
    │   ├── NeuralNetwork.cpp
    │   └── HybridPolicy.cpp
    ├── States/
    │   ├── (implementations)
    ├── Actions/
    │   ├── Movement/
    │   │   ├── MoveForwardAction.cpp
    │   │   ├── MoveBackwardAction.cpp
    │   │   ├── MoveLeftAction.cpp
    │   │   └── MoveRightAction.cpp
    │   ├── Combat/
    │   │   ├── MeleeAttackAction.cpp
    │   │   ├── SkillAttackAction.cpp
    │   │   └── RangedAttackAction.cpp
    │   └── Tactical/
    │       ├── TakeCoverAction.cpp
    │       └── UseItemAction.cpp
    └── Training/
        ├── (implementations)
```

**Benefits:**
- Clear API boundary (Public/ contains only what's needed externally)
- Faster compilation (Private/ changes don't trigger full rebuild)
- Follows Unreal Engine conventions
- Better maintainability

---

## Critical Issues Addressed

### Issue 1: LearningAgents Plugin Unused
**Solution:** Evaluate LearningAgents for integration. If not suitable, remove dependency. If suitable, use for reward specification and policy training.

### Issue 2: Poor Code Organization
**Solution:** Refactor to Public/Private structure (detailed above).

### Issue 3: Incomplete States (Flee, Dead)
**Solution:** Fully implement all states with complete action sets and transition logic.

### Issue 4: Hardcoded Parameters
**Solution:** Expose via `UPROPERTY(EditAnywhere, Category="AI")` for runtime tuning.

### Issue 5: Limited Observation Space
**Solution:** Expand to 60-80 features (raycast vision, tactical context, temporal data).

### Issue 6: No Neural Network
**Solution:** Integrate PyTorch neural network for policy and value functions.

### Issue 7: No Distributed Training
**Solution:** Implement RLlib integration with Docker and AWS SageMaker.

### Issue 8: No Persistence
**Solution:** Serialize models to disk, version in S3/MLflow.

### Issue 9: Synchronous Execution
**Solution:** Run MCTS on background thread, async updates.

### Issue 10: No Training/Eval Separation
**Solution:** Implement training mode (exploration) vs evaluation mode (greedy).

### Issue 11: No Unit Tests
**Solution:** Add tests using Unreal Automation Framework (target: 80% coverage).

### Issue 12: Blueprint Dependency
**Solution:** Provide C++ implementations, allow Blueprint overrides.

---

## Success Metrics

### Technical Metrics

| Metric | Target |
|--------|--------|
| Training Speed | 1M steps in <15 minutes |
| Inference Latency (p95) | <10ms |
| Model Size | <5 MB |
| Win Rate (Level 5) | >80% |
| Episode Reward | >500 average |
| Code Coverage | >80% |

### Quality Metrics

| Metric | Target |
|--------|--------|
| Agent Survives (Level 5) | >90% |
| Reaches Goal (Level 5) | >80% |
| Average Time to Goal | <2 minutes |
| Human Win Rate (vs AI) | 40-60% (balanced) |

### Production Metrics

| Metric | Target |
|--------|--------|
| Inference Endpoint Uptime | 99.9% |
| Deployment Frequency | Weekly |
| Rollback Rate | <5% |
| Training Cost | <$500/month |

---

## Implementation Timeline

### Sprint 1-2: Foundation (2 weeks)
- Refactor code to Public/Private
- Expand observation space
- Complete all state implementations
- Add unit tests (core components)

### Sprint 3-4: Neural Networks (2 weeks)
- Implement neural network (policy + value)
- Integrate PyTorch or ONNX Runtime
- Hybrid MCTS + NN policy
- Experience replay buffer

### Sprint 5-7: Distributed Training (3 weeks)
- RLlib environment wrapper
- Socket-based communication layer
- Docker containerization
- Multi-worker local training

### Sprint 8-10: Cloud Infrastructure (3 weeks)
- AWS SageMaker training pipeline
- S3 model registry
- Inference endpoint deployment
- CI/CD for model updates

### Sprint 11-12: Advanced Features (2 weeks)
- Curriculum learning
- IRL from demonstrations
- Multi-agent coordination (optional)
- MCTS visualization tools

### Sprint 13-14: Production Hardening (2 weeks)
- Performance optimization (latency <10ms)
- Monitoring dashboards (Grafana)
- Documentation (API, deployment guide)
- Production deployment

**Total:** 14 weeks (3.5 months)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Neural network doesn't converge | Medium | High | Start with behavior cloning, extensive hyperparameter tuning |
| Training too slow | Low | Medium | Use distributed training, curriculum learning |
| Inference latency too high | Low | High | Optimize network size, use ONNX, async execution |
| AWS costs exceed budget | Medium | Medium | Use Spot Instances, local training for development |
| Integration issues (RLlib + Unreal) | Medium | High | Prototype early, use socket-based decoupling |
| Model degradation in production | Low | High | A/B testing, automated rollback, monitoring |

---

## Recommended Tools and Frameworks

### Core Frameworks
- **Unreal Engine 5.6** - Game engine
- **Ray RLlib 2.9+** - Distributed RL training
- **PyTorch 2.0+** - Neural network framework
- **Docker 24+** - Containerization

### Cloud Services (Optional)
- **AWS SageMaker** - Managed training and inference
- **AWS S3** - Model storage
- **AWS EC2** - Custom training clusters

### MLOps Tools
- **MLflow** - Experiment tracking and model registry
- **TensorBoard** - Metrics visualization
- **Grafana + Prometheus** - Production monitoring
- **GitHub Actions** - CI/CD pipeline

### Development Tools
- **Visual Studio 2022** - C++ IDE
- **Unreal Automation Framework** - Unit testing
- **clang-format** - Code formatting
- **Docker Compose** - Local orchestration

---

## Conclusion

This final methodology transforms SBDAPM into a **production-grade distributed reinforcement learning system** by:

1. **Enhancing Observations** - 60-80 features for rich state representation
2. **Hybrid Policy** - MCTS + Neural Networks for strong, fast decisions
3. **Distributed Training** - RLlib enables 10-100x faster learning
4. **Cloud Infrastructure** - AWS SageMaker for elastic compute
5. **Proper Code Organization** - Public/Private structure following Unreal standards
6. **MLOps Practices** - Versioning, monitoring, automated deployment
7. **Curriculum Learning** - Progressive difficulty for robust policies
8. **Production Hardening** - <10ms latency, 99.9% uptime

### Key Differentiators

| Aspect | Current | Improved |
|--------|---------|----------|
| Observation Space | 3 features | 60-80 features |
| Policy | MCTS only | Hybrid (MCTS + NN) |
| Training | Single process | 8-32 workers |
| Training Speed | 5.5 hours/1M steps | 10 min/1M steps |
| Inference | 50ms | <10ms |
| Persistence | None | S3 + MLflow |
| Deployment | Manual | Automated CI/CD |
| Monitoring | None | Grafana dashboards |

### Expected ROI

**Development:** 14 weeks
**Cost:** ~$500/month (cloud training) OR $3,600 one-time (local hardware)
**Outcome:**
- Superhuman AI agents
- Scalable training pipeline
- Production-ready deployment
- Extensible framework for future projects

---

**Approval Required:** Yes
**Next Steps:** Review → Prioritize → Implement
**Contact:** See CLAUDE.md for support channels

---

**Document Version:** 1.0 (Final)
**Date:** 2025-10-26
**Author:** Claude Code Assistant
**Status:** Ready for Review
