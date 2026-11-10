"""Interfejs czatu PyGPT — ciemny, minimalistyczny motyw z czerwonym akcentem."""

import datetime
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

import voices as voice_catalog

from macros import SAMPLE_RATE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ---------------------------------------------------------------- pallete ----
BG          = "#0f1011"
BG_PANEL    = "#141517"
BG_ELEV     = "#1a1c1f"
BG_HOVER    = "#23262a"
BG_USER     = "#2a1113"
BORDER      = "#26282c"
BORDER_SOFT = "#1d1f22"
ACCENT      = "#c1121f"
ACCENT_DIM  = "#7d0f16"
ACCENT_GLOW = "#3a0d12"
TEXT        = "#e7e7e8"
TEXT_DIM    = "#9ca0a6"
TEXT_MUTED  = "#6b7075"
OK_GREEN    = "#3fb950"
REC_RED     = "#ff4d4d"

FONT_UI     = "Segoe UI"
COL_MAX     = 760          # maks. mess column width


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M")


def _round_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    points = (
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    )
    return canvas.create_polygon(points, smooth=True, **kwargs)


def _load_icon(name: str, size: int):
    path = os.path.join(ASSETS_DIR, name)
    try:
        return tk.PhotoImage(file=path)
    except Exception:
        return None


_AVATAR_CACHE = {}


def _circle_crop(image):
    from PIL import Image, ImageDraw

    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, side - 1, side - 1), fill=255)
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(image, (0, 0), mask)
    return out


def _avatar_photo(voice_id, size):
    if not voice_id:
        return None
    path = voice_catalog.image_path(voice_id)
    if not path:
        return None
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None

    key = (path, stamp, size)
    cached = _AVATAR_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from PIL import Image, ImageTk

        image = Image.open(path).convert("RGBA")
    except Exception:
        return None

    if image.getchannel("A").getextrema()[0] > 16:
        image = _circle_crop(image)

    image = image.resize((size, size), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    _AVATAR_CACHE[key] = photo
    return photo


class ThinScrollbar(tk.Canvas):


    WIDTH = 8

    def __init__(self, parent, command):
        super().__init__(parent, width=self.WIDTH, bg=BG,
                         highlightthickness=0, bd=0, cursor="arrow")
        self._command = command
        self._first, self._last = 0.0, 1.0
        self._drag = None
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))

    def set(self, first, last):
        self._first, self._last = float(first), float(last)
        self._draw()

    def _draw(self):
        self.delete("all")
        height = self.winfo_height()
        if height <= 2 or self._last - self._first >= 1.0:
            return
        top = self._first * height
        bottom = self._last * height
        bottom = min(height - 2, max(bottom, top + 22))
        _round_rect(self, 1, top, self.WIDTH - 1, bottom, 4,
                    fill="#2f3237", outline="")

    def _press(self, event):
        self._drag = (event.y, self._first, self.winfo_height())
        if self._last - self._first < 1.0:
            relative = event.y / max(self.winfo_height(), 1)
            self._command("moveto", max(0.0, min(1.0, relative - 0.1)))

    def _motion(self, event):
        if not self._drag:
            return
        start_y, start_first, height = self._drag
        delta = (event.y - start_y) / max(height, 1)
        self._command("moveto", max(0.0, min(1.0, start_first + delta)))


class Avatar(tk.Canvas):
    def __init__(self, parent, voice_id=None, letter="P", size=28):
        super().__init__(parent, width=size, height=size,
                         bg=BG, highlightthickness=0, bd=0)
        self._photo = _avatar_photo(voice_id, size)
        if self._photo is not None:
            self.create_image(size / 2, size / 2, image=self._photo)
        else:
            self.create_oval(0, 0, size - 1, size - 1,
                             fill=ACCENT_GLOW, outline=ACCENT_DIM, width=1)
            self.create_text(size / 2, size / 2 + 1, text=letter,
                             fill=ACCENT, font=(FONT_UI, 9, "bold"))


class Bubble(tk.Canvas):
    def __init__(self, parent, text, max_width=430, pad_x=14, pad_y=10,
                 radius=14, fill=BG_USER, outline="#3d161a", fg=TEXT):
        super().__init__(parent, bg=BG, highlightthickness=0, bd=0)
        self._text = self.create_text(
            pad_x, pad_y, text=text, width=max_width, anchor="nw",
            font=(FONT_UI, 10), fill=fg, justify="left")
        self.update_idletasks()
        x1, y1, x2, y2 = self.bbox(self._text)
        width = x2 + pad_x
        height = y2 + pad_y
        _round_rect(self, 1, 1, width - 1, height - 1, radius,
                    fill=fill, outline=outline)
        self.tag_raise(self._text)
        self.config(width=width, height=height)


class MicButton(tk.Canvas):
    SIZE = 40

    def __init__(self, parent, on_press, on_release):
        super().__init__(parent, width=self.SIZE, height=self.SIZE,
                         bg=BG_ELEV, highlightthickness=0, bd=0,
                         cursor="hand2")
        self._on_press = on_press
        self._on_release = on_release
        self._active = False
        self._pulse = 0
        self._icon = _load_icon("mic_24.png", 24)
        self._draw()

        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda _: (not self._active) and self._draw(hover=True))
        self.bind("<Leave>", lambda _: (not self._active) and self._draw())

    def _draw(self, hover=False):
        self.delete("all")
        c = self.SIZE / 2
        fill = REC_RED if self._active else (BG_HOVER if hover else BG_ELEV)
        outline = REC_RED if self._active else (BORDER if not hover else ACCENT_DIM)
        self.create_oval(1, 1, self.SIZE - 1, self.SIZE - 1,
                         fill=fill, outline=outline, width=1)
        if self._active:
            ring = 20 + (self._pulse % 6)
            self.create_oval(c - ring, c - ring, c + ring, c + ring,
                             outline=REC_RED, width=1)
        if self._icon is not None:
            self.create_image(c, c, image=self._icon)
        else:
            self.create_text(c, c, text="mic", fill=TEXT_DIM, font=(FONT_UI, 7))

    def _press(self, _=None):
        if self._active:
            return
        self._active = True
        self._pulse = 0
        self._draw()
        self._animate()
        self._on_press()

    def _release(self, _=None):
        if not self._active:
            return
        self._active = False
        self._draw(hover=True)
        self._on_release()

    def _animate(self):
        if not self._active:
            return
        self._pulse += 1
        self._draw()
        self.after(120, self._animate)


class SendButton(tk.Canvas):
    SIZE = 40

    def __init__(self, parent, on_click):
        super().__init__(parent, width=self.SIZE, height=self.SIZE,
                         bg=BG_ELEV, highlightthickness=0, bd=0,
                         cursor="hand2")
        self._on_click = on_click
        self._enabled = False
        self._draw()
        self.bind("<Button-1>", lambda _: self._on_click())
        self.bind("<Enter>", lambda _: self._draw(hover=True))
        self.bind("<Leave>", lambda _: self._draw())

    def set_enabled(self, enabled: bool):
        if enabled != self._enabled:
            self._enabled = enabled
            self._draw()

    def _draw(self, hover=False):
        self.delete("all")
        c = self.SIZE / 2
        if self._enabled:
            fill = "#d4141f" if hover else ACCENT
            self.create_oval(1, 1, self.SIZE - 1, self.SIZE - 1,
                             fill=fill, outline="", width=0)
            self.create_text(c, c - 1, text="→", fill="#ffffff",
                             font=(FONT_UI, 13, "bold"))
        else:
            self.create_oval(1, 1, self.SIZE - 1, self.SIZE - 1,
                             fill=BG_ELEV, outline=BORDER, width=1)
            self.create_text(c, c - 1, text="→", fill=TEXT_MUTED,
                             font=(FONT_UI, 13, "bold"))


class StatusBar(tk.Frame):
    STATES = {
        "ready":     (OK_GREEN,  "ready"),
        "recording": (REC_RED,   "recording"),
        "thinking":  (TEXT_MUTED, "thinking"),
        "rendering": (TEXT_MUTED, "generating voice"),
        "speaking":  (OK_GREEN,  "speaking"),
        "error":     (REC_RED,   "error"),
    }

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._busy = False
        self._phase = 0
        self._label = tk.Label(self, text="● ready", font=(FONT_UI, 8),
                               fg=OK_GREEN, bg=BG, anchor="w")
        self._label.pack(side="left", padx=4)

        self._hint = tk.Label(self, text="Enter — send   ·   Shift+Enter — new line",
                              font=(FONT_UI, 8), fg=TEXT_MUTED, bg=BG,
                              anchor="e")
        self._hint.pack(side="right", padx=4)

        self._timer = tk.Label(self, text="", font=(FONT_UI, 8),
                               fg=TEXT_MUTED, bg=BG, anchor="e")
        self._timer.pack(side="right", padx=4)

    def set(self, state: str, extra: str = ""):
        color, text = self.STATES.get(state, (TEXT_MUTED, state))
        self._label.config(fg=color)
        self._busy = state in ("thinking", "rendering", "speaking")
        self._phase = 0
        self._text = text + ("  " + extra if extra else "")
        if self._busy:
            self._animate()
        else:
            self._label.config(text="● " + self._text if state != "error" else "✕ " + self._text)

    def _animate(self):
        if not self._busy:
            return
        dots = "." * (self._phase % 4)
        self._label.config(text="● " + self._text + dots)
        self._phase += 1
        self.after(380, self._animate)

    def set_timer(self, text: str):
        self._timer.config(text=text)
        if text:
            self._hint.pack_forget()
        elif not self._hint.winfo_ismapped():
            self._hint.pack(side="right", padx=4)


class Waveform(tk.Canvas):

    HEIGHT = 54
    FRAME_MS = 33
    SPAN_SAMPLES = 1024        # ok. 43 ms przy 24 kHz
    LINE      = (0, 232, 118)
    GLOW      = (0, 96, 52)

    PANEL_BG = "#050506"
    GRID     = "#111114"
    BASE     = "#1c1c20"

    def __init__(self, parent):
        super().__init__(parent, height=self.HEIGHT, bg=BG,
                         highlightthickness=0, bd=0)
        self._samples = None
        self._samplerate = 0
        self._gain = None
        self._duration = 0.0
        self._t0 = 0.0
        self._active = False
        self._job = None
        self._size = (0, 0)
        self._surface = None
        self._photo = None
        self._item = None
        self.bind("<Configure>", lambda _e: self._redraw(self._elapsed()))
        self._draw_idle()

    # -------------------------------------------------------------                                                                                                                                                                                                 --
    def play(self, samples, samplerate):
                                                                        
        self.stop()
        if samples is None or samplerate <= 0:
            return
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return

        self._samples = data
        self._samplerate = int(samplerate)
        self._gain = None
        self._duration = len(data) / float(samplerate)
        self._t0 = time.perf_counter()
        self._active = True
        self._tick()

    def stop(self):
        self._active = False
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        self._draw_idle()

    def _trigger(self, position):

        limit = min(len(self._samples) - 2, position + self.SPAN_SAMPLES)
        samples = self._samples
        for i in range(max(1, position), limit):
            if samples[i - 1] < 0.0 <= samples[i]:
                return i
        return position

    def _elapsed(self):
        return time.perf_counter() - self._t0 if self._active else 0.0

    def _tick(self):
        if not self._active:
            return
        elapsed = self._elapsed()
        if elapsed > self._duration + 0.3:
            self.stop()
            return
        self._redraw(elapsed)
        self._job = self.after(self.FRAME_MS, self._tick)

    def _ensure_surface(self, width, height):
        from PIL import Image, ImageTk

        if self._surface is not None and self._size == (width, height):
            return
        self._size = (width, height)
        self._surface = Image.new("RGB", (width, height), self.PANEL_BG)
        self._photo = ImageTk.PhotoImage(self._surface)
        if self._item is None:
            self._item = self.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.itemconfig(self._item, image=self._photo)

    def _draw_idle(self):
        if not self.winfo_exists():
            return
        width = self.winfo_width()
        if width <= 8:
            return
        self._ensure_surface(width, self.HEIGHT)
        self._render(None)

    def _render(self, elapsed):
        from PIL import ImageDraw

        width, height = self._size
        self._surface.paste(self.PANEL_BG, (0, 0, width, height))
        draw = ImageDraw.Draw(self._surface)

        for part in (1, 2, 3):
            y = int(height * part / 4)
            draw.line([(3, y), (width - 4, y)], fill=self.GRID)
        draw.rectangle([0, 0, width - 1, height - 1], outline="#191c20")
        draw.line([(3, height - 2), (width - 4, height - 2)], fill=self.BASE)

        if elapsed is not None and self._active:
            self._draw_wave(draw, width, height)

        self._photo.paste(self._surface)

    def _draw_wave(self, draw, width, height):
        if self._samples is None or self._samplerate <= 0:
            return

        position = int(self._elapsed() * self._samplerate)
        start = self._trigger(position)
        block = self._samples[start:start + self.SPAN_SAMPLES]
        if block.size < 16:
            return

        peak = float(np.abs(block).max())
        headroom = (height / 2 - 6) * 0.86
        target = headroom / max(peak, 0.05)
        if self._gain is None or target < self._gain:
            self._gain = target       
        else:
            self._gain += (target - self._gain) * 0.25
        gain = min(self._gain, headroom / 0.05)

        centre = height / 2
        usable_w = width - 8
        count = len(block)
        points = []
        for i in range(count):
            x = 4 + usable_w * i / (count - 1)
            y = centre - float(block[i]) * gain
            points.append((x, max(3.0, min(height - 4.0, y))))

        draw.line(points, fill=self.GLOW, width=3, joint="curve")
        draw.line(points, fill=self.LINE, width=1, joint="curve")

    def _redraw(self, elapsed):
        if not self.winfo_exists():
            return
        width = self.winfo_width()
        if width <= 8:
            return
        self._ensure_surface(width, self.HEIGHT)
        self._render(elapsed)


class InputCard(tk.Canvas):
    HEIGHT = 100

    def __init__(self, parent, on_send, on_press_mic, on_release_mic, on_typing):
        super().__init__(parent, bg=BG, highlightthickness=0, bd=0,
                         height=self.HEIGHT)
        self._on_send = on_send
        self._focused = False

        self._inner = tk.Frame(self, bg=BG_ELEV)
        self._win = self.create_window(0, 0, anchor="nw", window=self._inner)

        self.mic = MicButton(self._inner, on_press_mic, on_release_mic)
        self.mic.pack(side="right", padx=(8, 6), pady=10)

        self.send = SendButton(self._inner, on_send)
        self.send.pack(side="right", padx=(0, 0), pady=10)

        self.text = tk.Text(self._inner, height=3, bg=BG_ELEV, fg=TEXT,
                            relief="flat", bd=0, highlightthickness=0,
                            insertbackground=ACCENT, font=(FONT_UI, 10),
                            wrap="word", padx=6, pady=4, spacing1=1,
                            spacing3=2)
        self.text.pack(side="left", fill="both", expand=True, padx=(14, 8))

        self.text.insert("1.0", "Write a message…")
        self.text.config(fg=TEXT_MUTED)

        self.text.bind("<FocusIn>", self._focus_in)
        self.text.bind("<FocusOut>", self._focus_out)
        self.text.bind("<KeyRelease>", lambda _e: on_typing())
        # Enter wysyła, Shift+Enter wstawia nową linię.
        self.text.bind("<Return>", self._on_return)
        self.text.bind("<KP_Enter>", self._on_return)
        self._on_typing = on_typing

        self.bind("<Configure>", self._redraw)

    def _on_return(self, event):
        if event.state & 0x0001:      
            return None
        if self.value():
            self._on_send()
        return "break"

    def _redraw(self, event=None):
        width = self.winfo_width()
        height = self.HEIGHT
        self.delete("bg")
        _round_rect(self, 1, 1, width - 1, height - 1, 20,
                    fill=BG_ELEV,
                    outline=ACCENT_DIM if self._focused else BORDER, tags="bg")
        self.tag_lower("bg")
        self.coords(self._win, 14, 12)
        self.itemconfig(self._win, width=max(width - 28, 10),
                        height=height - 24)

    def _focus_in(self, _=None):
        self._focused = True
        if self.text.get("1.0", "end-1c") == "Write a message…":
            self.text.delete("1.0", "end")
            self.text.config(fg=TEXT)
        self._redraw()

    def _focus_out(self, _=None):
        self._focused = False
        if not self.text.get("1.0", "end-1c").strip():
            self.text.insert("1.0", "Write a message…")
            self.text.config(fg=TEXT_MUTED)
        self._redraw()

    # -- API ----------------------------------------------------------
    def value(self) -> str:
        text = self.text.get("1.0", "end-1c").strip()
        return "" if text == "Write a message…" else text

    def clear(self):
        self.text.delete("1.0", "end")
        self.send.set_enabled(False)
        self._redraw()


class PyGPT_GUI:
    def __init__(self, pygpt_instance=None):
        self.pygpt = pygpt_instance
        self._is_busy = False
        self._recording_active = False
        self._wrap_labels = []

        self.root = tk.Tk()
        self.root.title("PyGPT")
        self.root.configure(bg=BG)
        self.root.geometry("860x700")
        self.root.minsize(620, 520)

        self._style_ttk()
        self._build_ui()
        self._bind_keys()

        self._ui_queue = queue.Queue()
        self.root.after(40, self._drain_ui)

        self._greet()

   
    def _post(self, func):
        self._ui_queue.put(func)

    def _drain_ui(self):
        try:
            while True:
                func = self._ui_queue.get_nowait()
                try:
                    func()
                except Exception:
                    import traceback
                    traceback.print_exc()
        except queue.Empty:
            pass
        try:
            if self.root.winfo_exists():
                self.root.after(40, self._drain_ui)
        except tk.TclError:
            pass


    def _style_ttk(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")


    def _build_ui(self):
        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(body, bg=BG, highlightthickness=0, bd=0)
        self._scroll = ThinScrollbar(body, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._scroll.pack(side="right", fill="y", padx=(2, 8), pady=8)
        self._canvas.pack(side="left", fill="both", expand=True)

        self._feed = tk.Frame(self._canvas, bg=BG)
        self._feed_id = self._canvas.create_window((0, 0), window=self._feed,
                                                   anchor="n")
        self._feed.bind("<Configure>", self._on_feed_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_composer()

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        logo = tk.Canvas(header, width=30, height=30, bg=BG_PANEL,
                         highlightthickness=0)
        photo = _avatar_photo(self._voice_id(), 30)
        if photo is not None:
            self._logo_photo = photo     
            logo.create_image(15, 15, image=photo)
        else:
            _round_rect(logo, 0, 0, 29, 29, 9, fill=ACCENT, outline="")
            logo.create_text(15, 16, text="P", fill="#ffffff",
                             font=(FONT_UI, 11, "bold"))
        logo.pack(side="left", padx=(18, 10), pady=18)

        titles = tk.Frame(header, bg=BG_PANEL)
        titles.pack(side="left", pady=14)
        tk.Label(titles, text="PyGPT", font=(FONT_UI, 11, "bold"),
                 fg=TEXT, bg=BG_PANEL, anchor="w").pack(anchor="w")
        tk.Label(titles, text=self._subtitle(), font=(FONT_UI, 8),
                 fg=TEXT_MUTED, bg=BG_PANEL, anchor="w").pack(anchor="w")

        chips = tk.Frame(header, bg=BG_PANEL)
        chips.pack(side="right", padx=(0, 16))
        self._chip(chips, self._model_text())
        self._chip(chips, self._voice_text())

        tk.Frame(self.root, bg=BORDER_SOFT, height=1).pack(fill="x")

    def _chip(self, parent, text, width=None):
        canvas = tk.Canvas(parent, height=24, bg=BG_PANEL,
                           highlightthickness=0, bd=0)
        font = (FONT_UI, 8)
        tmp = canvas.create_text(0, 0, text=text, font=font)
        x1, _y1, x2, _y2 = canvas.bbox(tmp)
        canvas.delete(tmp)
        w = (x2 - x1) + 24
        canvas.config(width=w)
        _round_rect(canvas, 0, 0, w - 1, 23, 11, fill=BG_ELEV, outline=BORDER)
        canvas.create_text(w / 2, 12, text=text, font=font, fill=TEXT_DIM)
        canvas.pack(side="right", padx=(6, 0))
        return canvas

    def _build_composer(self):
        composer = tk.Frame(self.root, bg=BG)
        composer.pack(fill="x", side="bottom")

        self._card = InputCard(
            composer,
            on_send=self._send_text,
            on_press_mic=self._start_recording,
            on_release_mic=self._stop_recording,
            on_typing=self._on_typing,
        )
        self._card.pack(fill="x", padx=16, pady=(4, 2))

        self._status = StatusBar(composer)
        self._status.pack(fill="x", padx=20, pady=(0, 2))

        self._waveform = Waveform(composer)
        self._waveform.pack(fill="x", padx=16, pady=(0, 6))

    def _voice_id(self):
        if not self.pygpt:
            return None
        return self.pygpt.voice or voice_catalog.GTTS

    def _subtitle(self) -> str:
        if not self.pygpt:
            return "demo mode"
        return "voice assistant · {}".format(self.pygpt.lang.upper())

    def _model_text(self) -> str:
        if not self.pygpt:
            return "no backend"
        import ai_models as model_catalog
        return model_catalog.label_for(self.pygpt.model).upper()

    def _voice_text(self) -> str:
        if not self.pygpt or not self.pygpt.voice:
            return "GTTS"
        return voice_catalog.label(self.pygpt.voice).upper()

    # --------------------------------------------------------- zdarzenia --
    def _bind_keys(self):
        self.root.bind("<space>", self._space_ptt)
        self.root.bind("<Escape>", lambda _: self._card.text.focus_set())

    def _on_typing(self):
        self._card.send.set_enabled(bool(self._card.value()))

    def _space_ptt(self, event):
        if self.root.focus_get() is not self._card.text:
            self._card.mic._press()
            self.root.after(3000, self._card.mic._release)
            return "break"

    def _on_feed_configure(self, _=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        width = min(event.width, COL_MAX)
        self._canvas.itemconfig(self._feed_id, width=width)
        self._canvas.coords(self._feed_id, event.width // 2, 0)
        wrap = max(width - 110, 220)
        for label in self._wrap_labels:
            if label.winfo_exists():
                label.config(wraplength=wrap)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _scroll_bottom(self):
        self.root.after(40, lambda: self._canvas.yview_moveto(1.0))

    # ---------------------------------------------------------- wiadomości --
    def _greet(self):
        if self.pygpt and self.pygpt.lang in ("pl", "PL"):
            text = ("Cześć! Jestem gotowy. Napisz wiadomość albo przytrzymaj "
                    "mikrofon, żeby mówić.")
        else:
            text = "Hi! I'm ready. Type a message or hold the mic to talk."
        self._add_message("assistant", text, _ts())

    def _add_message(self, role: str, text: str, timestamp: str = None):
        timestamp = timestamp or _ts()
        if role == "user":
            self._add_user(text, timestamp)
        else:
            self._add_assistant(text, timestamp)
        self._scroll_bottom()

    def _add_assistant(self, text: str, timestamp: str):
        holder = tk.Frame(self._feed, bg=BG)
        holder.pack(fill="x", pady=(14, 0), padx=(6, 6))

        row = tk.Frame(holder, bg=BG)
        row.pack(fill="x", anchor="w")

        Avatar(row, voice_id=self._voice_id(), size=30).pack(
            side="left", anchor="n", pady=(1, 0), padx=(0, 10))

        body = tk.Frame(row, bg=BG)
        body.pack(side="left", fill="x", expand=True)

        meta = tk.Label(body, text="PyGPT   {}".format(timestamp),
                        font=(FONT_UI, 8), fg=TEXT_MUTED, bg=BG, anchor="w")
        meta.pack(anchor="w", pady=(0, 3))

        wrap = max(min(self.root.winfo_width(), COL_MAX) - 110, 220)
        label = tk.Label(body, text=text, font=(FONT_UI, 10), fg=TEXT,
                         bg=BG, justify="left", anchor="w", wraplength=wrap)
        label.pack(anchor="w")
        self._wrap_labels.append(label)

    def _add_user(self, text: str, timestamp: str):
        holder = tk.Frame(self._feed, bg=BG)
        holder.pack(fill="x", pady=(14, 0), padx=(6, 6))

        meta = tk.Label(holder, text="You   {}".format(timestamp),
                        font=(FONT_UI, 8), fg=TEXT_MUTED, bg=BG, anchor="e")
        meta.pack(anchor="e", padx=(0, 6), pady=(0, 3))

        width = max(min(self.root.winfo_width(), COL_MAX) - 120, 220)
        Bubble(holder, text, max_width=min(460, width)).pack(anchor="e",
                                                             padx=(0, 6))

    # --------------------------------------------------------- wskaźniki --
    def _add_typing_indicator(self):
        self._typing = tk.Frame(self._feed, bg=BG)
        self._typing.pack(fill="x", padx=(16, 6), pady=(12, 0), anchor="w")
        self._typing_dots = tk.Label(self._typing, text="PyGPT is writing",
                                     font=(FONT_UI, 9), fg=TEXT_MUTED, bg=BG)
        self._typing_dots.pack(side="left")
        self._typing_phase = 0
        self._animate_typing()
        self._scroll_bottom()

    def _animate_typing(self):
        if not getattr(self, "_typing", None) or not self._typing.winfo_exists():
            return
        self._typing_dots.config(text="PyGPT is writing" + "." * (self._typing_phase % 4))
        self._typing_phase += 1
        self.root.after(380, self._animate_typing)

    def _remove_typing_indicator(self):
        widget = getattr(self, "_typing", None)
        if widget is not None and widget.winfo_exists():
            widget.destroy()
        self._typing = None

    # -------------------------------------------------------------- akcje --
    def _send_text(self):
        if self._is_busy:
            return
        text = self._card.value()
        if not text:
            return

        self._card.clear()
        self._add_message("user", text)
        self._is_busy = True
        threading.Thread(target=self._query_gpt, args=(text,), daemon=True).start()

    def _query_gpt(self, text: str):
        self._post(lambda: self._status.set("thinking"))
        self._post(self._add_typing_indicator)

        try:
            if self.pygpt and self.pygpt.is_chat:
                self.pygpt.sendMessage(text)
                response = self.pygpt.response.choices[0].message.content
            else:
                time.sleep(0.9)
                response = ("Demo mode — no backend is connected. You wrote: "
                            "\"{}\".").format(text)
        except Exception as exc:
            response = "Error: {}".format(exc)
            self._post(lambda: self._status.set("error"))
        else:
            self._post(lambda: self._status.set("ready"))

        self._post(self._remove_typing_indicator)
        self._post(lambda: self._add_message("assistant", response))

        if self.pygpt and (self.pygpt.voice or str(self.pygpt.tts).lower() == "gtts"):
            self._post(lambda: self._status.set("rendering"))
            threading.Thread(target=self._speak, args=(response,), daemon=True).start()
        else:
            self._post(self._unlock_ui)

    def _speak(self, text: str):
        try:
            import sounddevice as sd

            self._post(lambda: self._status.set("rendering"))

            if self.pygpt.voice:
                audio = self.pygpt.text2speech.synthesize(text)
                samplerate = SAMPLE_RATE
            else:
                audio, samplerate = self.pygpt.gtts_audio(text)

            self._post(lambda: self._status.set("speaking"))
            self._post(lambda: self._waveform.play(audio, samplerate))
            sd.play(audio, samplerate)
            sd.wait()
            self._post(self._waveform.stop)
            self._post(lambda: self._status.set("ready"))
        except Exception as exc:
            print("[speak error] {}".format(exc))
            self._post(self._waveform.stop)
            self._post(lambda: self._status.set("error"))
        finally:
            self._post(self._unlock_ui)

    def _unlock_ui(self):
        self._is_busy = False

    # -------------------------------------------------------------- nagranie --
    def _start_recording(self):
        if self._recording_active or self._is_busy:
            return
        self._is_busy = True
        self._recording_active = True
        self._status.set("recording")
        self._rec_start = time.time()
        self._update_rec_timer()
        threading.Thread(target=self._record_audio, daemon=True).start()

    def _stop_recording(self):
        self._recording_active = False

    def _update_rec_timer(self):
        if not getattr(self, "_recording_active", False):
            self._status.set_timer("")
            return
        self._status.set_timer("{:.1f}s".format(time.time() - self._rec_start))
        self.root.after(100, self._update_rec_timer)

    def _record_audio(self):
        recorder = None
        started_query = False
        try:
            if not self.pygpt:
                time.sleep(1.5)
                return
            from pvrecorder import PvRecorder
            recorder = PvRecorder(device_index=0, frame_length=512)

            recording = []
            recorder.start()
            while self._recording_active and len(recording) < 680448:
                recording.extend(recorder.read())
            recorder.stop()

            if not recording:
                return

            self._post(lambda: self._status.set("thinking"))
            arr = np.array(recording, dtype=float)
            arr /= max(np.max(np.abs(arr)), 1e-9)
            text = self.pygpt.speech2text(arr)
            if not text.strip():
                return

            self._post(lambda: self._add_message("user", text))
            self._post(lambda: self._status.set("ready"))
            started_query = True
            threading.Thread(target=self._query_gpt, args=(text,), daemon=True).start()
        except Exception as exc:
            print("[recording error] {}".format(exc))
            self._post(lambda: self._status.set("error"))
        finally:
            self._recording_active = False
            try:
                if recorder is not None:
                    recorder.stop()
            except Exception:
                pass
            self._post(lambda: self._status.set_timer(""))
            if not started_query:
                self._post(self._unlock_ui)

    def run(self):
        self.root.mainloop()
