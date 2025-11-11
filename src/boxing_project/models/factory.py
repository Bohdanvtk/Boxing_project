# src/boxing_project/models/factory.py
def build_model_for_task(Task, dims: dict, cfg: dict):
    """Determine which model will do tasks"""
    return Task.build_model(dims, cfg)
