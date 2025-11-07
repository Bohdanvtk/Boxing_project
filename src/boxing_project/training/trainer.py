# src/boxing_project/training/trainer.py
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from boxing_project.utils.config import set_seed
from boxing_project.data.loaders import make_train_val
from boxing_project.models.factory import build_model_for_task

def train(cfg: dict, Task):
    set_seed(int(cfg["train"]["seed"]))

    # 1) dims від задачі
    dims = Task.dims(cfg)  # {"NUM_PEOPLE_MAX":..., "JOINTS_PER_PERSON":..., "FEAT_PER_JOINT":...}

    # 2) дані
    (X_tr, y_tr, extras_tr), (X_val, y_val, extras_val) = make_train_val(cfg, Task)

    # 3) модель
    model = build_model_for_task(Task, dims, cfg)
    model.summary()

    monitor = Task.monitor_metric()
    mode = Task.monitor_mode()


    # 4) колбеки
    ck_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ck_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(ck_dir, f'{Task.name()}_best.keras'),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        restore_best_weights=True,
        verbose=1
    )

    # 5) sample weights (якщо задача їх надає)
    sw_tr = sw_val = None
    if hasattr(Task, "make_sample_weights") and bool(cfg["train"].get("use_sample_weights", False)):
        sw_tr = Task.make_sample_weights(X_tr, extras_tr, dims, cfg)
        sw_val = Task.make_sample_weights(X_val, extras_val, dims, cfg)

    # 6) fit
    fit_kwargs = dict(
        x=X_tr, y=y_tr,
        batch_size=int(cfg["train"]["batch_size"]),
        epochs=int(cfg["train"]["epochs"]),
        validation_data=(X_val, y_val) if sw_val is None else (X_val, y_val, sw_val),
        verbose=1, callbacks=[checkpoint]
    )
    if sw_tr is not None:
        fit_kwargs["sample_weight"] = sw_tr

    history = model.fit(**fit_kwargs)

    # 7) експорт (опційно)
    exported = cfg["paths"].get("exported")
    if exported:
        os.makedirs(os.path.dirname(exported), exist_ok=True)
        model.save(exported)

    # 8) постобробка валідації (якщо є)
    post = None
    if hasattr(Task, "postprocess_val"):
        post = Task.postprocess_val(model, X_val, extras_val, dims, cfg)

    return {"history": getattr(history, "history", None), "post": post}



