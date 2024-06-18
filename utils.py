import cv2
import numpy as np
from collections import namedtuple, deque
import random
from screen_captor import grab_screen
import imageio
import os


def get_frame_and_state(region, target_size=(400, 300), normalize=True):
    # region: capture area in screen
    # target_size: resize the img
    # get a frame and the game state (state: 0 = win, 1 = dead, 2 = playing, 3 = not start
    frame = grab_screen(region)
    B = frame[24, 24][0]
    G = frame[24, 24][1]
    R = frame[24, 24][2]
    if R == 255 and G == 255 and B == 255:
        state = 3
    elif R > 0:
        state = 1
    elif G > 0:
        state = 0
    elif B > 0:
        state = 2
    else:
        state = -1

    frame = cv2.resize(cv2.cvtColor(grab_screen(region), cv2.COLOR_BGR2GRAY), target_size)
    # cv2.imshow('window1',frame)
    if normalize:
        return frame.astype(np.float32) / 255, state  # normalize
    else:
        return frame, state


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'surv_dur', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        return List[Transition]
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    if __name__ == '__main__':
        while True:
            avd_window = (780, 345, 1000, 750)
            frame, state = grab_screen(avd_window)
            print(state)
            cv2.imshow('w1', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cv2.waitKey()  # 视频结束后，按任意键退出
        cv2.destroyAllWindows()
