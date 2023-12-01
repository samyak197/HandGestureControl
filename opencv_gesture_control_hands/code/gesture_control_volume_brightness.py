import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

mpHands = mp.solutions.hands
hands_brightness = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2,
)
Draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRANGE = volume.GetVolumeRange()
midVol = (volumeRANGE[0] + volumeRANGE[1]) / 2
volume.SetMasterVolumeLevel(midVol, None)

camera = cv2.VideoCapture(0)

is_paused = False

while True:
    ret, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            x1, y1 = 0, 0
            x2, y2 = 0, 0

            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:
                    x1, y1 = cx, cy
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

                if id == 8:
                    x2, y2 = cx, cy
                    cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            distance = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(distance, [15, 150], [volumeRANGE[0], volumeRANGE[1]])
            per = np.interp(distance, [15, 150], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)
            mpDraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
            cv2.putText(
                img,
                f"Volume: {int(per)}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    Process = hands_brightness.process(imgRGB)
    landmarkList = []

    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channels = img.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(img, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList != []:
        is_fist = all(landmarkList[i][2] > landmarkList[0][2] for i in range(5))

        if is_fist:
            if not is_paused:
                print("Pause")
                is_paused = True
        else:
            if is_paused:
                print("Play")
                is_paused = False

        x_1, y_1 = landmarkList[8][1], landmarkList[8][2]
        x_2, y_2 = landmarkList[12][1], landmarkList[12][2]

        cv2.circle(img, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

        cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

        L = math.hypot(x_2 - x_1, y_2 - y_1)
        b_level = np.interp(L, [15, 220], [0, 100])

        sbc.set_brightness(int(b_level))

        cv2.putText(
            img,
            f"Brightness: {int(b_level)}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()
