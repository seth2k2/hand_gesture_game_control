import os
import cv2
import pickle
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,  #make sure that only one hand is captured
    min_detection_confidence=0.5
)

IMG_DIR = './train_images'  #directory for the training images

data = []
labels = []

for dir_ in os.listdir(IMG_DIR):
    dir_path = os.path.join(IMG_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  #continue if a proper directory isn't found

    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            print("Failed to read image:",img_full_path)
            continue  #continue if image cannot be read due to an error

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            hand_landmarks = results.multi_hand_landmarks[0] #if multiple hands detected take only the first detected hand

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            min_x = min(x_coords)
            min_y = min(y_coords)

            data_aux = []
            for lm in hand_landmarks.landmark:
                normalized_x = lm.x - min_x
                normalized_y = lm.y - min_y
                data_aux.extend([normalized_x, normalized_y])

            if len(data_aux) == 42:  #there are 21 land marks for a hand, and by considering x,y coordinates for each land  mark, we get 42 coordinates
                data.append(data_aux)
                labels.append(dir_)
        else:
            print("No hands detected:",img_full_path)

with open('data.pickle', 'wb') as f:  #save the data as a binary file using pickle
    pickle.dump({'data': data, 'labels': labels}, f)

hands.close()
