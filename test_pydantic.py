import sys
sys.path.append('~/fineval')

from DataStructures import EvalItem, TaskType

item = EvalItem(
    id="finqa-001",
    task_type=TaskType.NUMERICAL_QA,
    input={
        "context": "Revenue grew from $4.2B to $4.8B",
        "question": "What was the growth rate?"
    },
    expected_output=0.1449,
    metadata={"tolerance": 0.001}
)

print(item.id)
print(item.task_type)
print(item.input["question"])
print(item.expected_output)