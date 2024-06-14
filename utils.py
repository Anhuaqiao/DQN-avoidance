import cv2
import numpy as np
from collections import namedtuple, deque
import random
from screen_captor import grab_screen
import imageio
import os


def get_screenshot(region, target_size=(400, 300), normalize=True):
    # region: capture area in screen
    # target_size: resize the img
    frame = cv2.resize(cv2.cvtColor(grab_screen(region), cv2.COLOR_BGR2GRAY), target_size)
    # cv2.imshow('window1',frame)
    if normalize:
        return frame.astype(np.float32) / 255  # normalize
    else:
        return frame

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        return List[Transition]
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


while True:
    print(1)
    avd_window = (640, 300, 1600, 1200)
    get_screenshot(avd_window)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.waitKey()  # 视频结束后，按任意键退出
cv2.destroyAllWindows()