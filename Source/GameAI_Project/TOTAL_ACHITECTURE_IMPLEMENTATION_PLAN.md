## 🇬🇧 English Translation of the Document

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐      gRPC/Protobuf      ┌────────────────────┐   │
│  │  Unreal Engine   │◄──────────────────────► │   Python Client    │   │
│  │     5.6          │                         │                    │   │
│  │                  │                         │  ┌──────────────┐  │   │
│  │  ┌────────────┐  │   Observations (71+40)  │  │  OpenAI Gym  │  │   │
│  │  │   Schola   │  │ ─────────────────────►  │  │   Wrapper    │  │   │
│  │  │   Plugin   │  │                         │  └──────┬───────┘  │   │
│  │  │            │  │   Actions (0-15)        │         │          │   │
│  │  │  • Agent   │  │ ◄─────────────────────  │  ┌──────▼───────┐  │   │
│  │  │  • Sensors │  │                         │  │    RLlib     │  │   │
│  │  │  • Actuator│  │   Rewards               │  │  (PPO/DQN)   │  │   │
│  │  └────────────┘  │ ─────────────────────►  │  └──────┬───────┘  │   │
│  │                  │                         │         │          │   │
│  │  ┌────────────┐  │                         │  ┌──────▼───────┐  │   │
│  │  │ SBDAPM     │  │                         │  │ AWS SageMaker│  │   │
│  │  │ Components │  │                         │  │ (Distributed)│  │   │
│  │  └────────────┘  │                         │  └──────────────┘  │   │
│  └──────────────────┘                         └────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │   ONNX Model       │
                         │   (Exported)       │
                         └─────────┬──────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE (Production)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Unreal Engine 5.6 + NNE Plugin + ONNX Runtime                         │
│  • No Python dependency                                                 │
│  • Sub-5ms inference                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Game Engine** | Unreal Engine 5.6 | Game environment, agent simulation |
| **UE-Python Bridge** | **Schola Plugin** (AMD) | gRPC server, sensors, actuators |
| **RL Interface** | OpenAI Gym | Standard env interface (`step()`, `reset()`) |
| **RL Framework** | RLlib (Ray) | PPO/DQN algorithms, distributed training |
| **Cloud Training** | AWS SageMaker | Scalable distributed training |
| **Model Format** | ONNX | Cross-platform model export |
| **UE Inference** | NNE + ONNX Runtime | Native C++ inference |

---

### Phase 1: Schola Plugin Integration (gRPC & UE)
The goal of this phase is to integrate the Schola plugin to establish the data pipeline between UE and Python/RLlib.

| Step | Task Description | Difficulty | Key Deliverable |
| :--- | :--- | :--- | :--- |
| 1. | **Install Schola Plugin** Clone Schola repository to `Plugins/Schola`. gRPC/Protobuf dependencies are bundled. | Low | Schola plugin in project |
| 2. | **Configure Schola Agent** Create `ScholaAgentComponent` on follower pawns. Define sensors (observations) and actuators (actions) mapping to existing SBDAPM observation system (71 features). | Medium | Schola Agent Blueprint/C++ |
| 3. | **Install Python Package** `pip install schola[rllib]` - Installs Schola's Gym wrapper with RLlib support. | Low | Python environment ready |
| 4. | **Test Local Communication** Run UE with Schola server, connect Python client, verify observation/action/reward flow. | Medium | Local gRPC communication test |
| **✅ Phase 1 Completion Criteria:** Python script sends actions to UE via Schola, UE returns correct observations (71 features) and rewards. |

---

### Phase 2: Build RLlib Training Pipeline (RLlib & Python)
The goal of this phase is to build the RLlib-based learning system using Schola's Gym wrapper.

| Step | Task Description | Difficulty | Key Deliverable |
| :--- | :--- | :--- | :--- |
| 5. | **Extend Schola Gym Wrapper** Schola provides `schola.envs.UnrealEnv` out of the box. Extend it to map SBDAPM's 71-feature observation and 16-action space. Add reward shaping logic. | Medium | `SBDAPMEnv.py` |
| 6. | **Implement RLlib Training Script** Configure PPO algorithm with appropriate hyperparameters. Use Schola's multi-agent support if training multiple followers simultaneously. | Medium | `train_rllib.py` |
| 7. | **Observation/Action Normalization** Add preprocessing wrappers for observation normalization (mean/std) and action space handling (discrete 0-15 → ETacticalAction). | Low | Preprocessing wrappers |
| 8. | **Local Training Test** Run UE (headless or windowed) + Python training script locally. Verify learning curve improves over episodes. | Medium | TensorBoard logs showing improvement |
| **✅ Phase 2 Completion Criteria:** RLlib trains SBDAPM agents locally, policy loss decreases, reward increases over 100+ episodes. |

---

### Phase 3: Cloud Infrastructure Integration and Scaling (AWS SageMaker & VPC)
The goal of this phase is to move the learning system to the AWS cloud environment to enable large-scale distributed training.

| Step | Task Description | Difficulty | Key Deliverable |
| :--- | :--- | :--- | :--- |
| 9. | **Configure AWS VPC and Network Environment** Set up subnets, security groups within the same **VPC** to allow **UE Server (EC2)** and SageMaker Training Job to communicate. Open gRPC port (default 50051). | High | AWS VPC and Security Group |
| 10. | **Deploy UE Environment to Cloud** Deploy UE build to **EC2 instance** (g4dn.xlarge recommended). Run in **Headless Mode** with Schola server enabled. | Medium | UE Server on EC2 |
| 11. | **Package SageMaker Training Job** Create Docker image with RLlib, Schola Python package, and training scripts. Configure SageMaker to connect to EC2 UE instance. | High | SageMaker Docker image |
| 12. | **Distributed Training with Ray** Configure **Ray Cluster** for parallel environment rollouts. Multiple UE instances can train simultaneously via Schola's multi-env support. | High | Distributed training config |
| **✅ Phase 3 Completion Criteria:** Stable distributed training on AWS, faster convergence than local. |

---

### Phase 4: ONNX Export and Production Inference (NNE & UE)
The goal of this phase is to export the trained model and deploy it for real-time inference in UE without Python.

| Step | Task Description | Difficulty | Key Deliverable |
| :--- | :--- | :--- | :--- |
| 13. | **Export to ONNX** Export trained PyTorch policy to ONNX format (opset 11). Verify with `onnxruntime` in Python first. | Low | `tactical_policy.onnx` |
| 14. | **Enable NNE Plugin** Enable **NNE** and **NNERuntimeORT** plugins in UE Editor. | Low | Plugins enabled |
| 15. | **Load ONNX in UE** Use `URLPolicyNetwork::LoadPolicy()` to load ONNX model. Verify inference produces valid actions. | Medium | Working ONNX inference |
| 16. | **Performance Validation** Benchmark inference time (<5ms target). Profile memory usage. Test in shipping build. | Medium | Performance report |
| **✅ Phase 4 Completion Criteria:** ONNX model runs in UE with <5ms inference, no Python dependency. |

---

## Quick Reference: Key Technologies

| Technology | Version | Purpose | Link |
|------------|---------|---------|------|
| **Schola** | 1.3.0 | UE-Python gRPC bridge | github.com/GPUOpen-LibrariesAndSDKs/Schola |
| **RLlib** | 2.x | RL algorithms (PPO, DQN) | docs.ray.io/en/latest/rllib |
| **OpenAI Gym** | 0.26+ | Standard RL env interface | gymnasium.farama.org |
| **AWS SageMaker** | - | Cloud training | aws.amazon.com/sagemaker |
| **ONNX Runtime** | 1.16+ | Cross-platform inference | onnxruntime.ai |
| **NNE** | UE 5.6 | Native neural network engine | UE built-in plugin |
