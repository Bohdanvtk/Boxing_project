# src/boxing_project/tasks/referee.py
import os, re
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

class RefereeTask:
    @staticmethod
    def name(): return "referee"
    @staticmethod
    def monitor_metric(): return "val_accuracy"
    @staticmethod
    def monitor_mode(): return "max"

    # ---- dims із конфіга
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
    def _labels_to_one_hot(txt: str, P: int) -> np.ndarray:
        txt = re.sub(r'[,\s]+', '', txt)
        label_out = np.zeros(P, dtype=np.float32)
        idx = int(txt)
        if not (1 <= idx <= P):
            raise ValueError(f"label {idx} out of range 1..{P}")
        label_out[idx - 1] = 1.0
        return label_out

    @staticmethod
    def load_sample(folder_path: str):
        npz = np.load(os.path.join(folder_path, 'keypoints.npz'))
        keypoints = npz['data'].astype(np.float32)  # (P,J,F)
        P = keypoints.shape[0]
        with open(os.path.join(folder_path, 'label.txt'), 'r') as f:
            label = RefereeTask._labels_to_one_hot(f.readline(), P)
        # проста аугментація прикладу (як у тебе)
        if np.random.rand() < 0.5:
            perm = np.random.permutation(keypoints.shape[0])
            keypoints, label = keypoints[perm], label[perm]
        return keypoints, label, {}  # extra-поля не потрібні

    # ---- модель
    @staticmethod
    def _l2(alpha=1e-4): return regularizers.l2(alpha)

    @staticmethod
    def _build_person_encoder(joints: int, feat: int):
        inp = keras.Input(shape=(joints, feat))
        h = layers.Dense(64, activation='relu', kernel_regularizer=RefereeTask._l2())(inp)
        h = layers.Dense(64, activation='relu', kernel_regularizer=RefereeTask._l2())(h)
        h = layers.GlobalAveragePooling1D()(h)
        return keras.Model(inputs=inp, outputs=h, name='person_encoder')

    @staticmethod
    def build_model(dims: dict, cfg: dict):
        P, J, F = dims["NUM_PEOPLE_MAX"], dims["JOINTS_PER_PERSON"], dims["FEAT_PER_JOINT"]
        head = int(cfg["train"].get("head", 64))
        inp = layers.Input(shape=(P, J, F), name='inp')
        enc = RefereeTask._build_person_encoder(J, F)
        h = layers.TimeDistributed(enc, name='per_person_enc')(inp)
        h = layers.Dropout(0.3)(h)
        h = layers.TimeDistributed(layers.Dense(head, activation='relu', kernel_regularizer=RefereeTask._l2()),
                                   name='per_person_head')(h)
        logits = layers.TimeDistributed(layers.Dense(1, activation=None), name='logits_per_person')(h)
        logits = layers.Reshape((P,), name='squeezed_logits')(logits)
        probs = layers.Activation('softmax', name='probs')(logits)

        model = keras.Model(inputs=inp, outputs=probs)
        model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                      optimizer='adam', metrics=['accuracy'])
        return model
