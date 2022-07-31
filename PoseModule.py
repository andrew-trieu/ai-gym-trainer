import cv2
import mediapipe as mp
import time
import math


class PoseDetector:

    def __init__(self, mode=False, model_complexity=1, smoothness=True, segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.8, min_tracking_confidence=0.5):

        self.mode = mode
        self.complexity = model_complexity
        self.smoothness = smoothness
        self.segmentation = segmentation
        self.smooth_seg = smooth_segmentation
        self.min_det_con = min_detection_confidence
        self.min_tra_con = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothness, self.smooth_seg,
                                     self.min_det_con, self.min_tra_con)

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # .process() requires RGB image
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):  # returns list of landmarks in px
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # values of lm (x,y,z) return decimal values (ratio) of the screen
                # need to multiply by frame size (img.sh) to get pixels
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy, lm.visibility])
                if draw:
                    cv2.circle(img, (cx,cy), 25, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def determine_side(self, exercise):
        # if both left and right arms are visible - tell user to turn to one side
        # else check for left or right arm
        # around > 0.6 or > 0.7 as visibility value is good
        right_side = [0, False]

        right = {"bicep": {"shoulder": self.lmList[12][3],
                           "elbow": self.lmList[14][3],
                           "wrist": self.lmList[16][3]},
                 "squat": {"hip": self.lmList[24][3],
                           "knee": self.lmList[26][3],
                           "ankle": self.lmList[28][3]}}

        left = {"bicep": {"shoulder": self.lmList[11][3],
                          "elbow": self.lmList[13][3],
                          "wrist": self.lmList[15][3]},
                "squat": {"hip": self.lmList[23][3],
                          "knee": self.lmList[25][3],
                          "ankle": self.lmList[27][3]}}

        print(right["squat"]["hip"], right["squat"]["knee"], right["squat"]["ankle"])
        match exercise:
            case 'bicep':
                if right["bicep"]["shoulder"] > 0.8 and right["bicep"]["elbow"] > 0.8 and right["bicep"]["wrist"] > 0.8:
                    if left["bicep"]["elbow"] < 0.7:
                        # right arm
                        right_side = [1, True]
                    else:
                        right_side[0] = 2
                elif left["bicep"]["shoulder"] > 0.8 and left["bicep"]["elbow"] > 0.8 and left["bicep"]["wrist"] > 0.8:
                    if right["bicep"]["elbow"] < 0.7:
                        # left arm
                        right_side[0] = 1
                    else:
                        right_side[0] = 2
            case 'squat':
                if right["squat"]["hip"] > 0.8 and right["squat"]["knee"] > 0.8 and right["squat"]["ankle"] > 0.8:
                    if left["squat"]["hip"] < 0.7:
                        # right leg
                        right_side = [1, True]
                    else:
                        right_side[0] = 2
                elif left["squat"]["hip"] > 0.8 and left["squat"]["knee"] > 0.8 and left["squat"]["ankle"] > 0.8:
                    if right["squat"]["hip"] < 0.7:
                        # left leg
                        right_side[0] = 1
                    else:
                        right_side[0] = 2
        return right_side

    def find_angle(self, img, lm1, lm2, lm3, isRight=False, draw=True):
        # look at x and y coordinates for the landmarks specified (lm1-lm3)
        # need to look at 2nd and 3rd indices for coordinates as 1st index is landmark number
        x1, y1 = self.lmList[lm1][1:3]
        x2, y2 = self.lmList[lm2][1:3]
        x3, y3 = self.lmList[lm3][1:3]

        # calculating the angle between landmarks
        if isRight:
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))
        else:
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360
        # print(angle)
        # cv2.putText(img, "theta r-g: " + str(int(math.degrees(math.atan2(y3 - y2, x3 - x2)))), (200, 50), cv2.FONT_HERSHEY_PLAIN, 3,
        #             (255, 0, 0), 2)
        # cv2.putText(img, f"red coordinates: ({x3}, {y3})", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        # cv2.putText(img, f"green coordinates: ({x2}, {y2})", (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # cv2.putText(img, "phi blue - green: " + str(int(math.degrees(math.atan2(y1 - y2, x1 - x2)))), (0, 100),
        #             cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # cv2.putText(img, str(int(angle)), (x2+20, y2+50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
        return angle

    def in_frame(self, lm1, lm2, lm3, width, height):
        x1, y1 = self.lmList[lm1][1:3]
        x2, y2 = self.lmList[lm2][1:3]
        x3, y3 = self.lmList[lm3][1:3]

        if width > x1 and width > x2 and width > x3:
            if height > y1 and height > y2 and height > y3:
                return True
        return False

    # def display_info(self, percentage, img):
    #     pass

    # def bicep_curl(self, img, left=True):
    #     pass


def main():
    # .VideoCapture() can take video data from a webcam or mp4 file
    # videos 1-3 have a moment when no pose is detected, meaning .process() does not return any values
    # therefore pose_landmarks has no values and thus causing list index out of range if looking at coordinates of
    # a specific landmark
    cap = cv2.VideoCapture('PoseVideos/4resize.mp4')

    pTime = 0
    cTime = 0
    detector = PoseDetector()

    while True:
        # .read() reads video data and returns a single frame that it processed
        # success is a boolean value (did .read() read an image), img is the returned frame
        success, img = cap.read()
        img = detector.find_pose(img)
        lmList = detector.find_position(img)
        # safer to include "if len(lmList) != 0" in case a person's pose is not detected at any point
        # can also include a specific landmark, such as lmList[0] to return coordinates of the person's nose
        if len(lmList) != 0:
            print(lmList)
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (255, 0, 255), cv2.FILLED)

        # fps calculations
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # .imshow() displays an image (a frame) in a window labelled "Image"
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()