import tkinter as tk
from tkinter import ttk, messagebox
import queue
import threading
import traceback

import ai_models as model_catalog
import voices as voice_catalog
from gui import PyGPT_GUI
from gui import _avatar_photo as voice_avatar

# ---------------------------------------------------------------- pallete ----
BG        = "#0f1011"
PANEL     = "#141517"
CARD      = "#1a1c1f"
HOVER     = "#23262a"
BORDER    = "#26282c"
RED       = "#c1121f"
RED_DIM   = "#8f0d16"
RED_GHOST = "#3a0d12"
FG        = "#e7e7e8"
DIM       = "#6b7075"
MUTED     = "#9ca0a6"
FONT      = "Segoe UI"


def _rounded(canvas, x1, y1, x2, y2, r, **kwargs):
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    canvas.create_polygon(
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        smooth=True, **kwargs)


class LoadingBar(tk.Canvas):
    STEP_MS = 16
    SPEED = 0.018
    SEGMENT = 0.26

    def __init__(self, parent, width=364, height=8):
        super().__init__(parent, width=width, height=height, bg=BG,
                         highlightthickness=0, bd=0)
        self._bw = width
        self._bh = height
        self._pos = 0.0
        self._job = None
        self.draw()

    def start(self):
        if self._job is None:
            self._step()

    def stop(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        self.delete("all")

    def draw(self):
        self.delete("all")
        _rounded(self, 0, 0, self._bw, self._bh, self._bh / 2,
                 fill=CARD, outline="")

        span = self._bw * self.SEGMENT
        left = -span + (self._bw + span) * self._pos
        right = left + span
        x1 = max(1.0, left)
        x2 = min(self._bw - 1.0, right)
        if x2 - x1 > 2:
            _rounded(self, x1, 1, x2, self._bh - 1, (self._bh - 2) / 2,
                     fill=RED, outline="")

    def _step(self):
        if not self.winfo_exists():
            return
        self._pos = (self._pos + self.SPEED) % 1.0
        self.draw()
        self._job = self.after(self.STEP_MS, self._step)


class AccentButton(tk.Canvas):
    def __init__(self, parent, text, command, width=320, height=42):
        super().__init__(parent, width=width, height=height, bg=BG,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._text = text
        self._command = command
        self._enabled = True
        self._hover = False
        self._cw, self._ch = width, height
        self.bind("<Button-1>", lambda _: self._enabled and self._command())
        self.bind("<Enter>", lambda _: self._set_hover(True))
        self.bind("<Leave>", lambda _: self._set_hover(False))
        self._draw()

    def _set_hover(self, value):
        self._hover = value
        self._draw()

    def set_enabled(self, enabled):
        self._enabled = enabled
        self._draw()

    def _draw(self):
        self.delete("all")
        if not self._enabled:
            fill, outline, fg = RED_GHOST, BORDER, DIM
        elif self._hover:
            fill, outline, fg = "#d4141f", "", "#ffffff"
        else:
            fill, outline, fg = RED, "", "#ffffff"
        _rounded(self, 1, 1, self._cw - 1, self._ch - 1, 12,
                 fill=fill, outline=outline)
        self.create_text(self._cw / 2, self._ch / 2, text=self._text,
                         fill=fg, font=(FONT, 10, "bold"))


class StartScreen:
    def __init__(self):
        self.backend = None

        self.root = tk.Tk()
        self.root.title("PyGPT")
        self.root.geometry("420x444")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self._style()
        self._build_form()
        self._build_footer()

        self._ui_queue = queue.Queue()
        self.root.after(50, self._drain_ui)

        threading.Thread(target=self._refresh_models, daemon=True).start()


    def _post(self, func):

        self._ui_queue.put(func)

    def _drain_ui(self):
        try:
            while True:
                func = self._ui_queue.get_nowait()
                try:
                    func()
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        try:
            if self.root.winfo_exists():
                self.root.after(50, self._drain_ui)
        except tk.TclError:
            pass

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("PyGPT.TCombobox",
                        fieldbackground=CARD,
                        background=CARD,
                        foreground=FG,
                        arrowcolor=MUTED,
                        bordercolor=BORDER,
                        lightcolor=CARD,
                        darkcolor=CARD,
                        selectbackground=CARD,
                        selectforeground=FG,
                        padding=6,
                        borderwidth=0,
                        relief="flat")
        style.map("PyGPT.TCombobox",
                  fieldbackground=[("readonly", CARD), ("active", CARD)],
                  background=[("readonly", CARD), ("active", CARD)],
                  foreground=[("readonly", FG)],
                  bordercolor=[("focus", RED_DIM), ("active", BORDER)],
                  arrowcolor=[("active", FG)])

    def _build_form(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=28, pady=(26, 18))

        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text="PyGPT", font=(FONT, 15, "bold"),
                 fg=FG, bg=BG, anchor="w").pack(anchor="w")
        tk.Label(titles, text="voice assistant · launcher", font=(FONT, 8),
                 fg=DIM, bg=BG, anchor="w").pack(anchor="w")

        card = tk.Frame(self.root, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", padx=28)

        def row(text, factory, last=False):
            block = tk.Frame(card, bg=CARD)
            block.pack(fill="x", padx=16, pady=(14, 0))
            tk.Label(block, text=text, font=(FONT, 7, "bold"),
                     fg=DIM, bg=CARD, anchor="w").pack(fill="x", pady=(0, 5))
            widget = factory(block)
            widget.pack(fill="x")
            if last:
                tk.Frame(card, bg=CARD, height=14).pack()
            return widget

        self.lang_var = tk.StringVar(value="pl")
        row("LANGUAGE", lambda p: ttk.Combobox(
            p, textvariable=self.lang_var, values=["pl", "en"],
            state="readonly", style="PyGPT.TCombobox", height=4))

        # --- głosy: systemowy gTTS + wszystkie presety VALL-E X ---
        self._voice_choices = voice_catalog.choices()
        self._voice_map = {label: vid for label, vid in self._voice_choices}
        default_voice = next(
            (label for label, vid in self._voice_choices
             if vid == voice_catalog.GTTS),
            self._voice_choices[0][0])

        self.tts_var = tk.StringVar(value=default_voice)

        def voice_row(parent):
            box = tk.Frame(parent, bg=CARD)
            icon = tk.Canvas(box, width=26, height=26, bg=CARD,
                             highlightthickness=0, bd=0)
            icon.pack(side="left", padx=(0, 8))
            combo = ttk.Combobox(
                box, textvariable=self.tts_var,
                values=[label for label, _ in self._voice_choices],
                state="readonly", style="PyGPT.TCombobox", height=18)
            combo.pack(side="left", fill="x", expand=True)
            self._voice_icon = icon
            self._voice_combo = combo
            combo.bind("<<ComboboxSelected>>",
                       lambda _e: self._update_voice_icon())
            self._update_voice_icon()
            return box

        row("VOICE", voice_row)

        self._model_choices = model_catalog.choices()
        self._model_map = {label: mid for label, mid in self._model_choices}
        default_model = next(
            (label for label, mid in self._model_choices
             if mid == model_catalog.DEFAULT_MODEL),
            self._model_choices[0][0])

        self.model_var = tk.StringVar(value=default_model)

        def make_model_combo(parent):
            self._model_combo = ttk.Combobox(
                parent, textvariable=self.model_var,
                values=[label for label, _ in self._model_choices],
                state="readonly", style="PyGPT.TCombobox", height=12)
            return self._model_combo

        row("MODEL", make_model_combo, last=True)

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=28, pady=(16, 18))

        self._loader = tk.Frame(footer, bg=BG, height=34)
        self._loader.pack(fill="x", pady=(0, 16))
        self._loader.pack_propagate(False)

        self._prog_label = tk.Label(self._loader, text="", font=(FONT, 8),
                                    fg=DIM, bg=BG, anchor="center")
        self._prog_label.place(x=0, y=2, relwidth=1)

        self._progress = LoadingBar(self._loader, width=364, height=8)

        self._start_btn = AccentButton(footer, "START", self.start_app,
                                       width=364, height=44)
        self._start_btn.pack()

    # ------------------------------------------------------------ akcje --
    def _set_loading(self, loading: bool, msg: str = ""):
        if loading:
            self._start_btn.set_enabled(False)
            self._prog_label.config(text=msg)
            self._progress.place(x=0, y=24)
            self._progress.start()
        else:
            self._progress.stop()
            self._progress.place_forget()
            self._prog_label.config(text="")
            self._start_btn.set_enabled(True)

    def _refresh_models(self):
        """Dociąga aktualną listę darmowych modeli NagaAI (w tle)."""
        fetched = model_catalog.fetch_free_models()
        if not fetched:
            return
        choices = model_catalog.choices(fetched)
        if choices:
            self._post(lambda: self._apply_models(choices))

    def _apply_models(self, choices):
        current = self._model_map.get(self.model_var.get(),
                                      model_catalog.DEFAULT_MODEL)
        self._model_choices = choices
        self._model_map = {label: mid for label, mid in choices}
        self._model_combo.config(values=[label for label, _ in choices])
        for label, mid in choices:
            if mid == current:
                self.model_var.set(label)
                return

    def _update_voice_icon(self):
        """Pokazuje ikonę wybranej postaci obok listy głosów."""
        voice_id = self._voice_map.get(self.tts_var.get())
        photo = voice_avatar(voice_id, 26)
        self._voice_icon.delete("all")
        if photo is None:
            # np. gTTS — bez ikony, żeby combobox był wyrównany z resztą
            self._voice_icon.pack_forget()
            return
        if not self._voice_icon.winfo_ismapped():
            self._voice_icon.pack(side="left", padx=(0, 8),
                                  before=self._voice_combo)
        self._voice_icon._photo = photo       # referencja dla Tk
        self._voice_icon.create_image(13, 13, image=photo)

    def start_app(self):
        lang = self.lang_var.get().strip().lower()
        tts = self._voice_map.get(self.tts_var.get(), voice_catalog.GTTS)
        model = self._model_map.get(self.model_var.get(),
                                    model_catalog.DEFAULT_MODEL)

        if tts != voice_catalog.GTTS and lang != "en":
            messagebox.showerror(
                "Invalid configuration",
                "Głosy VALL-E X (Paimon, Bronya, Fu Xuan i pozostałe)\n"
                "obsługują tylko język angielski.\n\n"
                "Ustaw Language = 'en' albo wybierz głos 'gTTS (system)'.")
            return

        self.lang = lang
        self.tts = tts
        self.model = model

        threading.Thread(target=self._load_backend, daemon=True).start()

    def _load_backend(self):
        self._post(lambda: self._set_loading(True, "Initializing…"))

        try:
            from pvrecorder import PvRecorder
            self._post(lambda: self._prog_label.config(
                text="Loading speech recognition model…"))

            from pygpt import PyGPT

            print("Dostępne urządzenia nagrywające:")
            for i, d in enumerate(PvRecorder.get_available_devices()):
                print(f"  [{i}] {d}")

            if self.tts != voice_catalog.GTTS:
                self._post(lambda: self._prog_label.config(
                    text="Loading VALL-E X voice model… (this may take a while)"))
            else:
                self._post(lambda: self._prog_label.config(
                    text="Creating chat session…"))

            self.backend = PyGPT(
                model=self.model,
                lang=self.lang,
                tts=self.tts
            )
            self.backend.createChat()

        except Exception as e:
            traceback.print_exc()
            print(f"[ERROR] Backend: {e}")

            self.backend = None
            self._post(lambda: self._set_loading(False))
            self._post(lambda: messagebox.showwarning(
                "Backend Error",
                "Could not load backend: \n\nLaunching in demo mode."))

        self._post(self._launch_gui)

    def _launch_gui(self):
        self._set_loading(False)
        self.root.destroy()
        app = PyGPT_GUI(pygpt_instance=self.backend)
        app.run()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    StartScreen().run()
