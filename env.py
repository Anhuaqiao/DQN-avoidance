import action
import utils
import time
import torch


class Env:
    """
    act = 0: jump
        1: cancel jump
        2: left
        3: stop left
        4: right
        5: stop right
        6: do nothing
        7: restart
    """

    def __init__(self):
        self.avd_window = (780, 345, 1000, 750)
        self.epoch_start_time = time.time()
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.episode_start_time = 0
        self.survive_duration = 0
        self.pb = 0

    def step(self, act: int):
        # take a move, return the game state, reward, and next frame
        if act == 0:
            action.jump()
        elif act == 1:
            action.cancel_jump()
        elif act == 2:
            action.go_left()
        elif act == 3:
            action.stop_left()
        elif act == 4:
            action.go_right()
        elif act == 5:
            action.stop_right()
        # elif act == 6: do nothing
        elif act == 7:
            action.restart()
        frame, game_state = utils.get_frame_and_state(self.avd_window)
        self.survive_duration = time.time() - self.episode_start_time
        if game_state == 1 or game_state == 0:
            done = True
        else:
            done = False
        reward = self.get_reward(game_state)
        return frame, done, reward, self.survive_duration

    def restart(self):
        # 按R重开并返回第一帧图像
        action.release_all()
        action.restart()
        self.episode_start_time = time.time()
        print("new episode start...")
        while True:
            # 重开后会回到warp房，掉进warp前不记录状态
            frame, game_state = utils.get_frame_and_state(self.avd_window)
            if game_state != 3:
                break
        return frame

    def get_reward(self, state):
        r = 0
        if state == 2:
            r = 0
        if state == 0:
            r = 100
        if state == 1:
            r = (self.survive_duration - self.pb) * 100
            self.pb = max(self.pb,self.survive_duration)
            print(f"player dead, survive {self.survive_duration}s, pb {self.pb}s")
        return r


