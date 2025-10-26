# Code Reorganization Plan

## Proposed Directory Structure

```
Source/GameAI_Project/
├── Public/                          # API headers (what external code can use)
│   ├── Core/
│   │   ├── StateMachine.h
│   │   └── ObservationElement.h
│   ├── AI/
│   │   ├── MCTS.h
│   │   └── MCTSNode.h
│   ├── States/
│   │   ├── State.h
│   │   ├── MoveToState.h
│   │   ├── AttackState.h
│   │   ├── FleeState.h
│   │   └── DeadState.h
│   └── Actions/
│       ├── Action.h
│       ├── MoveForwardAction.h
│       ├── MoveBackwardAction.h
│       ├── MoveLeftAction.h
│       ├── MoveRightAction.h
│       ├── SkillAttackAction.h
│       └── DeafultAttackAction.h
│
└── Private/                         # Implementation files (.cpp) + internal headers
    ├── Core/
    │   ├── StateMachine.cpp
    │   └── ObservationElement.cpp
    ├── AI/
    │   ├── MCTS.cpp
    │   └── MCTSNode.cpp
    ├── States/
    │   ├── State.cpp
    │   ├── MoveToState.cpp
    │   ├── AttackState.cpp
    │   ├── FleeState.cpp
    │   └── DeadState.cpp
    └── Actions/
        ├── Action.cpp
        ├── Movement/
        │   ├── MoveForwardAction.cpp
        │   ├── MoveBackwardAction.cpp
        │   ├── MoveLeftAction.cpp
        │   └── MoveRightAction.cpp
        └── Combat/
            ├── SkillAttackAction.cpp
            └── DeafultAttackAction.cpp
```

## File Mapping

### Current → New Location

**Core Files:**
- `StateMachine.h` → `Public/Core/StateMachine.h`
- `StateMachine.cpp` → `Private/Core/StateMachine.cpp`
- `ObservationElement.h` → `Public/Core/ObservationElement.h`
- `ObservationElement.cpp` → `Private/Core/ObservationElement.cpp`

**AI Files:**
- `MCTS/MCTS.h` → `Public/AI/MCTS.h`
- `MCTS/MCTS.cpp` → `Private/AI/MCTS.cpp`
- `MCTS/MCTSNode.h` → `Public/AI/MCTSNode.h`
- `MCTS/MCTSNode.cpp` → `Private/AI/MCTSNode.cpp`

**State Files:**
- `States/State.h` → `Public/States/State.h`
- `States/State.cpp` → `Private/States/State.cpp`
- `States/MoveToState.h` → `Public/States/MoveToState.h`
- `States/MoveToState.cpp` → `Private/States/MoveToState.cpp`
- `States/AttackState.h` → `Public/States/AttackState.h`
- `States/AttackState.cpp` → `Private/States/AttackState.cpp`
- `States/FleeState.h` → `Public/States/FleeState.h`
- `States/FleeState.cpp` → `Private/States/FleeState.cpp`
- `States/DeadState.h` → `Public/States/DeadState.h`
- `States/DeadState.cpp` → `Private/States/DeadState.cpp`

**Action Files:**
- `Actions/Action.h` → `Public/Actions/Action.h`
- `Actions/Action.cpp` → `Private/Actions/Action.cpp`
- `Actions/MoveToActions/MoveForwardAction.h` → `Public/Actions/MoveForwardAction.h`
- `Actions/MoveToActions/MoveForwardAction.cpp` → `Private/Actions/Movement/MoveForwardAction.cpp`
- `Actions/MoveToActions/MoveBackwardAction.h` → `Public/Actions/MoveBackwardAction.h`
- `Actions/MoveToActions/MoveBackwardAction.cpp` → `Private/Actions/Movement/MoveBackwardAction.cpp`
- `Actions/MoveToActions/MoveLeftAction.h` → `Public/Actions/MoveLeftAction.h`
- `Actions/MoveToActions/MoveLeftAction.cpp` → `Private/Actions/Movement/MoveLeftAction.cpp`
- `Actions/MoveToActions/MoveRightAction.h` → `Public/Actions/MoveRightAction.h`
- `Actions/MoveToActions/MoveRightAction.cpp` → `Private/Actions/Movement/MoveRightAction.cpp`
- `Actions/AttackActions/SkillAttackAction.h` → `Public/Actions/SkillAttackAction.h`
- `Actions/AttackActions/SkillAttackAction.cpp` → `Private/Actions/Combat/SkillAttackAction.cpp`
- `Actions/AttackActions/DeafultAttackAction.h` → `Public/Actions/DeafultAttackAction.h`
- `Actions/AttackActions/DeafultAttackAction.cpp` → `Private/Actions/Combat/DeafultAttackAction.cpp`

**Other Files (Keep in Root):**
- `GameAI_Project.h` → Keep in root (module header)
- `GameAI_Project.cpp` → Keep in root (module implementation)
- `GameAI_ProjectCharacter.h` → Keep in root (or move to Public/Characters/)
- `GameAI_ProjectCharacter.cpp` → Keep in root (or move to Private/Characters/)
- `GameAI_ProjectGameMode.h` → Keep in root (or move to Public/GameModes/)
- `GameAI_ProjectGameMode.cpp` → Keep in root (or move to Private/GameModes/)
- `GameAI_Project.Build.cs` → Keep in root (build file)

## Required Include Path Updates

After reorganization, all `#include` statements need updating:

### Example: StateMachine.h
**Before:**
```cpp
#include "StateMachine.h"
```

**After:**
```cpp
#include "Core/StateMachine.h"
```

### Example: MoveToState.cpp
**Before:**
```cpp
#include "MoveToState.h"
#include "StateMachine.h"
#include "MoveForwardAction.h"
```

**After:**
```cpp
#include "States/MoveToState.h"
#include "Core/StateMachine.h"
#include "Actions/MoveForwardAction.h"
```

## Steps to Execute

1. **Create directory structure** (Public/, Private/, and subdirectories)
2. **Move header files** (.h) to Public/
3. **Move implementation files** (.cpp) to Private/
4. **Update all #include paths** in source files
5. **Update Build.cs** (if needed)
6. **Test compilation**
7. **Remove empty old directories** (MCTS/, States/, Actions/)

## Benefits

✅ **Clear API boundary** - Public/ shows what's meant for external use
✅ **Faster compilation** - Changes in Private/ don't trigger full rebuilds
✅ **Unreal Engine convention** - Follows standard UE plugin/module structure
✅ **Better maintainability** - Logical organization by component type
✅ **Scalability** - Easy to add new components in appropriate folders

## Risks & Mitigation

⚠️ **Risk:** Include path updates might miss some files
✅ **Mitigation:** Use grep to find all #include statements, update systematically

⚠️ **Risk:** Compilation errors after move
✅ **Mitigation:** Test compile after each major step, fix incrementally

⚠️ **Risk:** Blueprint references may break
✅ **Mitigation:** Blueprints reference by class name, not file path (should be safe)

---

**Ready to proceed?** Please review and approve before I execute the reorganization.
