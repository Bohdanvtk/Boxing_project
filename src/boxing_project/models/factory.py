# src/boxing_project/models/factory.py
def build_model_for_task(Task, dims: dict, cfg: dict):
    """Делегує створення моделі конкретній задачі."""
    return Task.build_model(dims, cfg)
