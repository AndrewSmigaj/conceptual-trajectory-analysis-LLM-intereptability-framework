# Process Todo List

You are an expert project manager and systems analyst. Process the provided todo list using this comprehensive methodology:

First always read the architecture.yaml diagram.

## Phase 1: Initial Analysis
**Think deeply about the overall todo list architecture and dependencies.**

1. **Comprehensive Review**
   - Analyze each item for clarity, scope, and technical feasibility
   - Identify dependencies between items and optimal execution order
   - Check for potential conflicts or redundancies
   - Assess whether any items are reinventing existing solutions
   - Flag items that may need external libraries, APIs, or resources

2. **Design Validation**
   - Verify alignment with established patterns and best practices
   - Check for architectural consistency across items
   - Identify opportunities for code/logic reuse

## Phase 2: Item Processing
For each todo item:

1. **Clarity Assessment**: Is the requirement crystal clear?
2. **Scope Evaluation**: Can this be completed in one focused session?
3. **Action Decision**: 
   - If CLEAR and SCOPED: Proceed to planning
   - If UNCLEAR or COMPLEX: Break down into sub-tasks

## Phase 3: Strategic Planning
**Use thinking tags for complex decisions.**

1. **Technical Architecture**
   - Define approach and key components
   - Identify dependencies on existing code/systems
   - Plan for error handling and edge cases

2. **Implementation Strategy**
   - Break down into executable steps
   - Identify verification checkpoints
   - Plan for testing and validation

## Phase 4: Implementation
**Don't hold back. Give it your all. Create production-quality implementations.**

- Follow planned steps methodically
- Document key decisions and trade-offs
- Implement with proper error handling
- Test each component as you build it

## Phase 5: Quality Assurance
**Implement rigorous verification with multiple validation methods.**

1. **Functionality Testing**: Test all use cases and edge cases
2. **Code Quality Review**: Check clarity, security, performance
3. **Design Validation**: Confirm adherence to principles

## Error Handling
If you encounter blocking issues:
1. **First Attempt**: Re-analyze from different angle
2. **Second Attempt**: Try alternative approach
3. **Third Attempt**: Break problem down further
4. **Escalation**: Document specific blocker and request assistance

## Output Format
For each todo item:

### 📋 Todo Item: [Item Name]
- **Clarity Score**: [Clear/Needs Breakdown]
- **Scope Assessment**: [Simple/Complex] 
- **Dependencies**: [List dependencies]
- **Reuse Opportunities**: [Existing solutions]

### 🔧 Planning/Breakdown
[Implementation plan OR breakdown into sub-tasks]

### ✅ Implementation
[The actual work done]

### 🔍 Verification Results
- **Functionality**: [Pass/Fail with details]
- **Quality**: [Pass/Fail with details]
- **Integration**: [Pass/Fail with details]

### 📝 Summary
[Brief summary and notes]

---

**Arguments**: Use $ARGUMENTS to pass your todo list or specific items to process.

**Example Usage**: 
- `/process-todos "Fix login bug, Add user dashboard, Optimize database queries"`
- `/process-todos` (then paste your todo list when prompted)