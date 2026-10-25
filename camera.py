"""Camera input shared by the live app and the capture tool."""
import threading
import time

import cv2
import numpy as np

SCRCPY_TITLE = "NoteCam"


class Camera:
    """Reads frames on a background thread and always keeps only the newest one.

    source: a camera index ("0"), a stream URL, or "scrcpy" for the scrcpy window.
    """

    def __init__(self, source):
        self.frame = None
        target = self._scrcpy if source == "scrcpy" else self._capture
        threading.Thread(target=target, args=(source,), daemon=True).start()

    def _capture(self, source):
        if source.isdigit():
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera: {source}")
        while True:
            ok, frame = cap.read()
            if ok:
                self.frame = frame
            else:
                time.sleep(0.05)

    def _scrcpy(self, _):
        """Grab the scrcpy window. Start it with:
        scrcpy --video-source=camera --camera-facing=back --window-title=NoteCam
        """
        import ctypes
        from ctypes import wintypes

        import mss

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        with mss.mss() as screen:
            while True:
                hwnd = user32.FindWindowW(None, SCRCPY_TITLE)
                rect, corner = wintypes.RECT(), wintypes.POINT(0, 0)
                if hwnd:
                    user32.GetClientRect(hwnd, ctypes.byref(rect))
                    user32.ClientToScreen(hwnd, ctypes.byref(corner))
                if rect.right < 10 or rect.bottom < 10:  # not open yet, or minimized
                    time.sleep(0.3)
                    continue
                area = {"left": corner.x, "top": corner.y, "width": rect.right, "height": rect.bottom}
                self.frame = np.ascontiguousarray(np.array(screen.grab(area))[:, :, :3])

    def wait(self, timeout=10):
        """Block until the first frame arrives."""
        end = time.time() + timeout
        while self.frame is None:
            if time.time() > end:
                raise SystemExit("No frames from the camera.")
            time.sleep(0.05)
        return self
