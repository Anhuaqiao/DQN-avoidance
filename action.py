import ctypes
import time
import cv2

SendInput = ctypes.windll.user32.SendInput

LSHIFT = 0x2A
R = 0x13
LEFT = 0xCB
RIGHT = 0xCD
esc = 0x01

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def jump():
    PressKey(LSHIFT)


def cancel_jump():
    ReleaseKey(LSHIFT)
    # time.sleep(0.1)


def go_left():
    PressKey(LEFT)


def stop_left():
    ReleaseKey(LEFT)


def go_right():
    PressKey(RIGHT)


def stop_right():
    ReleaseKey(RIGHT)


def restart():
    PressKey(R)
    time.sleep(0.1)
    ReleaseKey(R)


def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)


def release_all():
    ReleaseKey(LSHIFT)
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)


if __name__ == '__main__':
        time.sleep(1)
        jump()
        time.sleep(0.08)
        cancel_jump()
        for i in range(2):
            go_left()
