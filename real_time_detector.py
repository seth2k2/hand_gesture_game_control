import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui 

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#select the default camera
#since i have one camera i didnt check for cameras. But if you have multiple cameras you check them by putting videoCapture(0,1,2...)
cap = cv2.VideoCapture(0) 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

command="nothing" #variable to ensure keybaord key presses are not repeated infinitely, used in line no 73 onwards

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape #the height-H and width-W of  the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the frame is converted from BGR to RGB format for using it with MediaPipe

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] #if multiple hands detected take only the first detected hand
        mp_drawing.draw_landmarks(
        frame, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
        
        for i in range(len(hand_landmarks.landmark)): #collects all x and y coordinates for the hand
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))  #subtract the minimum x and y values from each landmarks x and y coordinates
            data_aux.append(y - min(y_))  #this translate the hand so that the leftmost and topmost landmarks are at (0,0)

        #coordnates for creating a bounding box for the hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        prediction = model.predict([np.asarray(data_aux)]) #getting the predicted category of data(jump,roll,left,right)

        predicted_command = prediction[0]

        #controlling the keyboard according to the predicted category and make sure that key press is only happening once 
        #per one gesture and that it will not repeat infinitely for a available gesture
        if(predicted_command=="jump" and command!="jump"):
            pyautogui.press('up')
            command="jump"
        elif(predicted_command=="roll" and command!="roll"):
            pyautogui.press('down')
            command=predicted_command
        elif(predicted_command=="left" and command!="left"):
            pyautogui.press('left')
            command=predicted_command
        elif(predicted_command=="right" and command!="right"):
            pyautogui.press('right')
            command=predicted_command
        elif predicted_command == "nothing" and command != "nothing":
            command = "nothing"

        #creating a bounding box for the hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        #displaying the predicted command
        cv2.putText(frame, predicted_command, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('p'): #by pressing character p in keyboard i can easily stop the program
        break

cap.release()
cv2.destroyAllWindows()