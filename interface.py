import cv2
import PoseModule as pm

"""
Print all graphics to screen
"""


def exercise_name(img, exercise):
    cv2.putText(img, f'Exercise: {exercise.capitalize()}', (600, 50),  cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), thickness=2)


class InterfaceDesign:
    def __init__(self, color=(127, 0, 255), count=0, direction=0):
        self.color = color
        self.count = count
        self.direction = direction

    def rep_progress(self, percentage):
        if percentage == 100:  # muscle fully contracted
            self.color = (0, 255, 0)
            if self.direction == 0:  # was in the motion of contracting before
                self.count += 0.5  # first half of rep complete
                self.direction = 1  # turning direction --> muscle relaxation
        elif percentage == 0:
            if self.direction == 1:  # was in the motion of relaxing before
                self.count += 0.5  # second half of rep complete
                self.direction = 0  # turning direction --> muscle contraction
        else:
            self.color = (127, 0, 255)

        return self.count

    def progress_bar(self, img, percentage, percentage_bar):
        cv2.rectangle(img, (0, 330), (80, 630), self.color, 2)
        cv2.rectangle(img, (0, int(percentage_bar)), (80, 630), self.color, cv2.FILLED)
        cv2.putText(img, f"{int(percentage)}%", (0, 315), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, thickness=2)

    def rep_count(self, img):
        cv2.rectangle(img, (0, 630), (400, 750), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f'Reps: {self.count}', (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=4)

