# src/boxing_project/tasks/participants.py
import os, re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ParticipantsTask:
    @staticmethod
    def name(): return "participants"
    @staticmethod
    def monitor_metric(): return "val_loss"   # або "val_auc"
    @staticmethod
    def monitor_mode(): return "min"          # або "max" для AUC

    @staticmethod
    def dims(cfg):
        t = cfg["task"]
        return {
            "NUM_PEOPLE_MAX": int(t["num_people_max"]),
            "JOINTS_PER_PERSON": int(t["joints_per_person"]),
            "FEAT_PER_JOINT": int(t["feat_per_joint"]),
        }

    # ---- завантаження одного семплу
    @staticmethod
    def _read_int_vector(txt_path: str, length: int, default: int = 0) -> np.ndarray:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                tokens = re.split(r'[\s,]+', f.readline().strip())
            vals = [int(t) for t in tokens if t != '']
            if len(vals) < length: vals += [default] * (length - len(vals))
            return np.array(vals[:length], dtype=np.int32)
        except Exception:
            return np.full((length,), default, dtype=np.int32)

    @staticmethod
    def load_sample(folder_path: str):
        npz = np.load(os.path.join(folder_path, 'keypoints.npz'))
        keypoints4 = npz['data'].astype(np.float32)            # (P,J,4)
        # valid counts
        if 'joints_valid_counts' in npz:
            valid_cnt = npz['joints_valid_counts'].astype(np.float32)
        else:
            is_valid = npz['is_valid'].astype(np.float32)      # (P,J)
            valid_cnt = np.sum(is_valid, axis=-1).astype(np.float32)
        P = keypoints4.shape[0]
        participants = ParticipantsTask._read_int_vector(os.path.join(folder_path, 'participants.txt'),
                                                         length=P, default=0).astype(np.float32)
        return keypoints4, participants, {"valid_counts": valid_cnt}

    # ---- модель і ваги зразків
    @staticmethod
    def build_model(dims: dict, cfg: dict):
        P, J, F = dims["NUM_PEOPLE_MAX"], dims["JOINTS_PER_PERSON"], dims["FEAT_PER_JOINT"]
        hid = int(cfg["train"].get("hid", 64))
        head_hid = int(cfg["train"].get("head_hid", 64))

        inp = keras.Input(shape=(P, J, F), name='keypoints')
        x = layers.Reshape((P, J*F), name='flatten_per_person')(inp)
        e = layers.Dense(128, activation='relu', name='embedding_dense1')(x)
        e = layers.Dense(hid, activation='relu', name='embedding_dense2')(e)
        g = layers.GlobalAveragePooling1D(name='scene_mean')(e)
        g_rep = layers.RepeatVector(P, name='scene_grep')(g)
        h = layers.Concatenate(axis=-1, name='concat_e_g')([e, g_rep])
        h = layers.Dense(head_hid, activation='relu', name='head_dense1')(h)
        out = layers.Dense(1, activation='sigmoid', name='head_dense2')(h)
        probs = layers.Reshape((P,), name='probs')(out)

        model = keras.Model(inputs=inp, outputs=probs, name='RingParticipantNet')
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[keras.metrics.BinaryAccuracy(name="bin_acc"),
                               keras.metrics.AUC(name="auc")])
        return model

    @staticmethod
    def make_sample_weights(X: np.ndarray, extras: dict, dims: dict, cfg: dict):
        J = float(dims["JOINTS_PER_PERSON"])
        # якщо valid_counts є — використовуємо їх
        if "valid_counts" in extras and isinstance(extras["valid_counts"], np.ndarray) \
                and extras["valid_counts"].shape[0] == X.shape[0]:
            valid_counts = extras["valid_counts"]
        else:
            # fallback тільки якщо є 4-та фіча
            if X.shape[-1] <= 3:
                # немає каналу is_valid → рівні ваги
                valid_counts = np.full(shape=X.shape[:2], fill_value=J, dtype=np.float32)
            else:
                is_valid = tf.convert_to_tensor(X)[..., 3] > 0.5
                valid_counts = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=-1).numpy()
        w_person = valid_counts / J  # (N, P)
        w_sample = np.mean(w_person, axis=1)  # (N,)
        return w_sample

    @staticmethod
    def postprocess_val(model, X_val: np.ndarray, extras: dict, dims: dict, cfg: dict):
        # приклад: top-3 індекси
        P = int(dims["NUM_PEOPLE_MAX"])
        k = min(3, P)
        probs = model.predict(X_val, verbose=0)
        probs_tf = tf.convert_to_tensor(probs, dtype=tf.float32)
        top_vals, top_idx = tf.math.top_k(probs_tf, k=k, sorted=False)
        return {"top3_idx": top_idx.numpy(), "top3_vals": top_vals.numpy()}
