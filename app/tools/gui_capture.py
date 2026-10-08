"""Capture only the application's own window for UI acceptance checks."""
import ctypes as c
from ctypes import wintypes as w
from PIL import Image


def capture_own_window(widget, target):
    user, gdi = c.windll.user32, c.windll.gdi32
    user.GetAncestor.restype = w.HWND
    user.GetWindowDC.restype = w.HDC
    gdi.CreateCompatibleDC.restype = w.HDC
    gdi.CreateCompatibleBitmap.restype = w.HBITMAP
    gdi.SelectObject.restype = w.HGDIOBJ
    hwnd = user.GetAncestor(w.HWND(widget.winfo_id()), 2)
    rect = w.RECT()
    user.GetWindowRect(hwnd, c.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    dc = user.GetWindowDC(hwnd)
    memory = gdi.CreateCompatibleDC(w.HDC(dc))
    bitmap = gdi.CreateCompatibleBitmap(w.HDC(dc), width, height)
    previous = gdi.SelectObject(w.HDC(memory), w.HGDIOBJ(bitmap))
    try:
        if not user.PrintWindow(hwnd, w.HDC(memory), 2):
            raise RuntimeError('Window rendering failed')
        class Header(c.Structure):
            _fields_ = [('size', w.DWORD), ('width', w.LONG), ('height', w.LONG),
                        ('planes', w.WORD), ('bits', w.WORD), ('compression', w.DWORD),
                        ('image_size', w.DWORD), ('x', w.LONG), ('y', w.LONG),
                        ('used', w.DWORD), ('important', w.DWORD)]
        info = Header(c.sizeof(Header), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        buffer = c.create_string_buffer(width * height * 4)
        if not gdi.GetDIBits(w.HDC(memory), w.HBITMAP(bitmap), 0, height, buffer, c.byref(info), 0):
            raise RuntimeError('Window pixels unavailable')
        Image.frombuffer('RGB', (width, height), buffer.raw, 'raw', 'BGRX', 0, 1).save(target)
    finally:
        gdi.SelectObject(w.HDC(memory), w.HGDIOBJ(previous))
        gdi.DeleteObject(w.HGDIOBJ(bitmap))
        gdi.DeleteDC(w.HDC(memory))
        user.ReleaseDC(hwnd, w.HDC(dc))
