"""用真实 GLFW 窗口消息验证期末场景的标题、视角、交互和阴影切换。"""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import re
import subprocess
import sys
import time

EXE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
TITLE = "2024270223_王锦政_期末大作业"
user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def window_for(process):
    found = []

    @CALLBACK
    def visit(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == process.pid:
            title = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, title, len(title))
            if title.value == TITLE:
                found.append(hwnd)
        return True

    deadline = time.monotonic() + 5
    while not found and time.monotonic() < deadline and process.poll() is None:
        user32.EnumWindows(visit, 0)
        time.sleep(0.05)
    if not found:
        raise RuntimeError("The expected UTF-8 window title was not found")
    return found[0]


def key(hwnd, code, pressed, extended=False):
    scan = user32.MapVirtualKeyW(code, 0)
    flags = 1 | (scan << 16) | (0x01000000 if extended else 0) | (0 if pressed else 0xC0000000)
    if not user32.PostMessageW(hwnd, 0x100 if pressed else 0x101, code, flags):
        raise ctypes.WinError(ctypes.get_last_error())


def run_interaction(name, change):
    target = OUT / (name + ".ppm")
    process = subprocess.Popen([str(EXE), "--capture=" + str(target), "--capture-delay=3.5", "--view=1"],
                               cwd=str(OUT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        hwnd = window_for(process)
        time.sleep(.6)
        if change == "objects":
            key(hwnd, 0x72, True); key(hwnd, 0x72, False)  # F3: south library
            key(hwnd, 0x27, True, True); key(hwnd, ord("L"), True)
            time.sleep(.65)
            key(hwnd, 0x27, False, True); key(hwnd, ord("L"), False)
        elif change == "shadow":
            key(hwnd, 0x54, True); key(hwnd, 0x54, False)  # T: shadows off
        else:
            # 正向一格鼠标滚轮，测试 GLFW 滚动回调和相机缩放。
            if not user32.PostMessageW(hwnd, 0x020A, 120 << 16, 0):
                raise ctypes.WinError(ctypes.get_last_error())
        stdout, stderr = process.communicate(timeout=10)
        output = stdout.decode("utf-8", "replace")
        if process.returncode or not target.is_file():
            raise RuntimeError(output + stderr.decode("utf-8", "replace"))
        match = re.search(r"Capture state: yaw=([\d.\-]+) pitch=([\d.\-]+) radius=([\d.\-]+) sculpture=([\d.\-]+) busX=([\d.\-]+) shadows=(\d)", output)
        if not match:
            raise RuntimeError("Missing capture state: " + output)
        yaw, pitch, radius, sculpture, bus, shadows = match.groups()
        result = dict(title=TITLE, yaw=float(yaw), pitch=float(pitch), radius=float(radius),
                      sculpture=float(sculpture), bus_x=float(bus), shadows=int(shadows), image=str(target))
        if change == "objects" and not (result["yaw"] == 195 and result["sculpture"] > 10 and result["bus_x"] > -9):
            raise AssertionError(result)
        if change == "shadow" and result["shadows"] != 0:
            raise AssertionError(result)
        if change == "wheel" and abs(result["radius"] - 85.5) > .1:
            raise AssertionError(result)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


results = {"objects": run_interaction("interaction", "objects"),
           "shadow_toggle": run_interaction("shadow-off", "shadow"),
           "mouse_wheel": run_interaction("mouse-wheel", "wheel")}
(OUT / "interaction-tests.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
