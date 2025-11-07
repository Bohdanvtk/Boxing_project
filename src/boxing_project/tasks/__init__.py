# src/boxing_project/tasks/__init__.py
from .referee import RefereeTask
from .participants import ParticipantsTask

REGISTRY = {
    RefereeTask.name(): RefereeTask,
    ParticipantsTask.name(): ParticipantsTask,
}

def get_task(cfg) -> type:
    task_name = cfg["task"]["name"]  # "referee" або "participants"
    return REGISTRY[task_name]


