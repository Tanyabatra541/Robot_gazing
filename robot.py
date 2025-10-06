import matplotlib.pyplot as plt
import threading 
import time
import cv2
import pyttsx3
from draw_util import *

class Robot():

    def __init__(self):
        self.joint_angles = {
            'pan': 0,
            'tilt': 0
        }

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

    def _on_speech_start(self, name):
        self.is_speaking = True
        print("Speech started")

    def _on_speech_end(self, name, completed):
        self.is_speaking = False
        self.current_word = None
        print("Speech ended")
    
    def _on_word_start(self, name, location, length):
        if self.speaking_phrase:
            self.current_word = self.speaking_phrase[location:location+length]
            print(f"Speaking: {self.current_word}")

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
        if self.mutual_gaze_flag:
            self.joint_angles['pan'] = 0 # degrees
            self.joint_angles['tilt'] = 0 # degrees

    def interaction_logic_seperate_thread(self):

        self.enable_mutual_gaze()
        time.sleep(5)
        self.disable_mutual_gaze()
        self.joint_angles['pan'] = 30 # degrees
        self.speak("I am a robot.", blocking=True)
        self.speak("look at that picture.", blocking=True)
        print(self.objects_in_env['picture'])
        time.sleep(1)
        self.joint_angles['pan'] = 0 # degrees
        self.speak("I think it is very pretty.", blocking=True)
        self.speak("It reminds me of the fun times I had when I was a baby robot.", blocking=True)

        # this tells the main control loop to stop
        self.completed = True


    def start(self):
        cap = cv2.VideoCapture(0) # 0 for the default webcam
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.canvas.mpl_connect('close_event', lambda event: setattr(self, 'completed', True))
        self.engine.startLoop(False)
        first_loop = True
        interaction_thread = None

        while cap.isOpened() and not self.completed:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # make it easier to process.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if first_loop:
                interaction_thread = threading.Thread(target=self.interaction_logic_seperate_thread)
                interaction_thread.start()
                first_loop = False

            # do the mutual gaze thing later to help with code
            self.mutual_gaze_loop(image)
            # iterate the TTS engine            
            self.engine.iterate()
            self.draw(ax)
            time.sleep(0.1)

        # clean up        
        if interaction_thread:
            interaction_thread.join()
        self.engine.endLoop()
        plt.close(fig)



if __name__ == "__main__":
    robot = Robot()
    robot.start()
