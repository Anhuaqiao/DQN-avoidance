import ctypes
import ctypes.wintypes
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, width, height = region
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    return img


def get_window_rect(hwnd):
    try:
        f = ctypes.windll.dwmapi.DwmGetWindowAttribute
    except WindowsError:
        f = None
    if f:
        rect = ctypes.wintypes.RECT()

        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        f(ctypes.wintypes.HWND(hwnd),
          ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
          ctypes.byref(rect),
          ctypes.sizeof(rect)
          )
        print(rect.left, rect.top)
        return rect.left, rect.top, 800, 600


def capture_window(title):
    handles = []
    win32gui.EnumWindows(lambda handle, param: param.append(handle), handles)

    for handle in handles:
        handle_title = win32gui.GetWindowText(handle)
        if title in handle_title:
            hwnd = handle
    # 找到窗口句柄
    if hwnd == 0:
        print("未找到窗口")
        return None
    # return win32gui.GetWindowRect(hwnd)
    return get_window_rect(hwnd)


