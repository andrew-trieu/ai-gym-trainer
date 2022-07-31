import cv2
import numpy as np
import mediapipe as mp
import time
import PoseModule as pm
import interface as it

# cap = cv2.VideoCapture('ai_trainer/bicepcurl.mp4')
cap = cv2.VideoCapture(0)  # access computer camera for live video feed
pose_detector = pm.PoseDetector()
interface_designer = it.InterfaceDesign()
count, direction = 0, 0
cTime, pTime = 0, 0
width, height = (1050, 750)  # window frame dimensions
inFrame = False
lm1, lm2, lm3 = [0 for _ in range(3)]
(low_angle, max_angle) = (0, 0)

getExercise = input("exercise: ")
repCount = input("num reps: ")
# bicep, shoulder, squat


while count < int(repCount):
    if getExercise.lower() not in ['bicep', 'squat']:
        print("Invalid exercise.")
        break

    # .read() reads video data and returns a single frame that it processed
    # success is a boolean value (did .read() read an image), img is the returned frame
    success, img = cap.read()
    img = cv2.resize(img, (width, height))

    img = pose_detector.find_pose(img, False)
    lmList = pose_detector.find_position(img, False)
    isRight, readyCheck, correctOrder = False, False, False

    # safer to include "if len(lmList) != 0" in case a person's pose is not detected at any point
    # can also include a specific landmark, such as lmList[0] to return coordinates of the person's nose
    if len(lmList) != 0:
        pos = ["Left", "Right"]
        specific_side = pose_detector.determine_side(getExercise.lower())
        if specific_side[0] == 0 or specific_side[0] == 2:
            cv2.putText(img, "Move to either left or right side.", (35, 125),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        else:
            if specific_side[0] == 1:  # user is turned to one side
                if specific_side[1]:  # right side
                    match getExercise.lower():
                        case 'bicep':
                            lm1, lm2, lm3 = 12, 14, 16
                        case 'squat':
                            lm1, lm2, lm3 = 24, 26, 28
                    isRight = True
                else:  #left side
                    match getExercise.lower():
                        case 'bicep':
                            lm1, lm2, lm3 = 11, 13, 15
                        case 'squat':
                            lm1, lm2, lm3 = 23, 25, 27
            readyCheck = True
            (low_angle, max_angle) = (215, 310)

        # case 'shoulder':
        #     pass
        #     # case 'squat':
        #     #     pos = ["Left", "Right"]
        #     #     specificLeg = pose_detector.determine_side(getExercise)
        #     #     lm1, lm2, lm3 = 24, 26, 28
        #
        #     case _:
        #         print("Invalid exercise.")
        #         break

        if readyCheck:
            inFrame = pose_detector.in_frame(lm1, lm2, lm3, width, height)
            if inFrame:  # print all landmark detectors, rep counters & progression measurements to screen
                angle = pose_detector.find_angle(img, lm1, lm2, lm3, isRight)
                cv2.putText(img, f"angle: + {angle}", (200, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                percentage = np.interp(angle, (low_angle, max_angle), (0, 100))
                # corresponding y-values are inverted b/c unfilled bar (0%) is lower on the screen --> larger px value
                # (0,0) in cv2 image is top left of frame
                bar = np.interp(angle, (low_angle, max_angle), (630, 330))
                count = interface_designer.rep_progress(percentage)
                interface_designer.progress_bar(img, percentage, bar)
                interface_designer.rep_count(img)
                it.exercise_name(img, getExercise)

            else:
                cv2.putText(img, "Move entire arm into frame to begin.", (35, 125),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("AI Personal Trainer", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

exit()
