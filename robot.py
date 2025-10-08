import matplotlib.pyplot as plt
import threading 
import time
import cv2
import pyttsx3
from draw_util import *
import mediapipe as mp
import string
import numpy as np

class Robot():

    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.joint_angles = {'pan': 0, 'tilt': 0}
        self.objects_in_env = {
            'picture': (5, -1, 1),
            'bookshelf': (-5, -1, 1),
        }

        self.completed = False
        self.mutual_gaze_flag = False
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self.speaking_phrase = None
        self.current_word = None
        self.engine.connect('finished-utterance', self._on_speech_end)
        self.engine.connect('started-word', self._on_word_start)
        self.engine.connect('started-utterance', self._on_speech_start)

        self.deictic_done = False

    def _on_speech_start(self, name):
        self.is_speaking = True

    def _on_speech_end(self, name, completed):
        self.is_speaking = False
        self.current_word = None
        self.deictic_done = False

    def _on_word_start(self, name, location, length):
        if self.speaking_phrase:
            self.current_word = self.speaking_phrase[location:location+length]

    def wait_until_done_speaking(self):
        while self.is_speaking:
            time.sleep(0.1)

    def speak(self, text, blocking=False):
        self.engine.say(text)
        self.speaking_phrase = text
        if blocking:
            self.wait_until_done_speaking()

    def draw(self, ax):
        plot_robot(ax, pan=self.joint_angles['pan'], tilt=self.joint_angles['tilt'])

    def enable_mutual_gaze(self):
        self.mutual_gaze_flag = True

    def disable_mutual_gaze(self):
        self.mutual_gaze_flag = False

    def mutual_gaze_loop(self, webcam_image):
        if not self.mutual_gaze_flag:
            return

        results = self.face_detector.process(webcam_image)
        if not results.detections:
            return  

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_x = bbox.xmin + bbox.width / 2
        face_y = bbox.ymin + bbox.height / 2

        delta_x = face_x - 0.5
        delta_y = face_y - 0.5

        self.joint_angles['pan'] = int(-delta_x * 90)
        self.joint_angles['tilt'] = int(delta_y * 90)

    def interaction_logic_seperate_thread(self):
        self.enable_mutual_gaze()
        time.sleep(5)
        self.disable_mutual_gaze()

        self.speak("I am a robot.", blocking=True)
        self.speak("look at that picture.", blocking=True)
        time.sleep(1)
        self.joint_angles['pan'] = 0

        duration = np.random.normal(3.54, 1.26)
        start_delay = np.random.normal(-1.32, 0.47)

        aversion_thread = threading.Thread(
            target=self.perform_gaze_aversion,
            args=(start_delay, duration)
        )
        aversion_thread.start()

        self.speak("I think it is very pretty.", blocking=True)
        self.speak("It reminds me of the fun times I had when I was a baby robot.", blocking=True)

        self.completed = True

    def start(self):
        cap = cv2.VideoCapture(0)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.canvas.mpl_connect('close_event', lambda event: setattr(self, 'completed', True))

        self.engine.startLoop(False)
        first_loop = True
        interaction_thread = None

        while cap.isOpened() and not self.completed:
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if first_loop:
                interaction_thread = threading.Thread(target=self.interaction_logic_seperate_thread)
                interaction_thread.start()
                first_loop = False

            self.mutual_gaze_loop(image)
            self.engine.iterate()

            if (
                self.current_word
                and self.current_word.lower().strip(string.punctuation) == "picture"
                and not self.deictic_done
            ):
                picture_pos = self.objects_in_env["picture"]
                head_pos = (0, 0, 3)

                dx = picture_pos[0] - head_pos[0]
                dz = picture_pos[2] - head_pos[2]
                angle_deg = np.degrees(np.arctan2(dx, -dz))

                self.joint_angles['pan'] = int(angle_deg)
                print(f"[Deictic Gaze] pan={int(angle_deg)}°")
                self.deictic_done = True

            self.draw(ax)
            time.sleep(0.1)

        if interaction_thread:
            interaction_thread.join()
        self.engine.endLoop()
        plt.close(fig)

    def perform_gaze_aversion(self, start_delay, duration):
        if start_delay > 0:
            time.sleep(start_delay)
        else:
            time.sleep(max(0, start_delay + np.random.normal(0, 0.05)))

        original_pan = self.joint_angles['pan']
        original_tilt = self.joint_angles['tilt']

        raw_side = np.random.normal(5, 1)
        side_angle = np.clip(raw_side, 3, 7) * np.random.choice([-1, 1])
        down_angle = np.clip(np.random.normal(20, 3), 15, 25)

        self.joint_angles['pan'] = original_pan + side_angle
        self.joint_angles['tilt'] = original_tilt + abs(down_angle)

        print(
            f"[Gaze Aversion] → pan={self.joint_angles['pan']:.2f}°, "
            f"tilt={self.joint_angles['tilt']:.2f}° "
            f"(down by {down_angle:.2f}°, side by {side_angle:.2f}°)"
        )

        time.sleep(max(0.5, duration))

        self.joint_angles['pan'] = 0
        self.joint_angles['tilt'] = 0
        print("[Gaze Aversion] Returned to user.")


if __name__ == "__main__":
    robot = Robot()
    robot.start()