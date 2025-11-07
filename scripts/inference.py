import numpy as np
from scripts.single_img_download import process_image
import cv2


INPUT_IMAGE_PATH = '/home/bohdan/Зображення/pictures/test_2.png'

img, npz = process_image(INPUT_IMAGE_PATH)
cv2.imshow('img', img)

from tensorflow.keras.models import load_model

participants_model = load_model('/home/bohdan/PycharmProjects/Boxing_Project/artifacts/exported/participants_model.keras', safe_mode=False)
referee_model = load_model("/home/bohdan/PycharmProjects/Boxing_Project/artifacts/exported/ref_model_2.keras", safe_mode=False)




def sync_participants_and_referee_models(npz, participants_model, referee_model, show=False):

    test_participants = npz.astype(np.float32)

    result_participants = participants_model.predict(np.expand_dims(test_participants, axis=0), verbose=0)

    scores = np.squeeze(result_participants)

    top3_idx = np.argpartition(scores, -3)[-3:]

    probs = scores[top3_idx]

    if show:
        persons = (top3_idx + 1).tolist()
        parts = [f"person{p}: {prob:.2f}" for p, prob in zip(persons, probs)]
        print(f"Their probabilities [{'  '.join(parts)} ])")

    test_referee = [test_participants[i] for i in top3_idx]

    result_referee = referee_model.predict(np.expand_dims(test_referee, axis=0), verbose=0)


    referee_idx = int(np.argmax(result_referee, axis=1)[0])

    referee = test_referee[referee_idx]

    matches = np.isclose(test_participants, referee, atol=1e-6)

    idx_referee = np.where(np.all(matches, axis=(1,2)))[0]

    if show:
        probs_ref = np.ravel(np.squeeze(result_referee))
        persons = (top3_idx + 1).tolist()
        parts_ref = [f"person{p}: {prob:.4f}" for p, prob in zip(persons, probs_ref)]
        print(f"referee model probabilities [{'  '.join(parts_ref)} ])")
        print(f'referee model predict is person {idx_referee[0]+1}')

    return referee_idx


sync_participants_and_referee_models(npz, participants_model, referee_model, show=True)



cv2.waitKey(0)
cv2.destroyAllWindows()





