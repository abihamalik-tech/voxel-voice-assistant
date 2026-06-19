"""
VOXEL — Voice Assistant (Groq edition)
======================================
- Rotating 3D glass orb
- Say "hey assistant" then speak your command
- STT stays LOCAL (Vosk), TTS stays LOCAL (pyttsx3)
- The "brain" now runs on Groq cloud (LPU inference, 500+ tokens/sec)

Install:
    pip install vosk sounddevice pyttsx3 numpy
    Linux: sudo apt install python3-tk portaudio19-dev espeak

Vosk model: https://alphacephei.com/vosk/models
    → vosk-model-small-en-us-0.15, extract next to this file

Groq API key (free, no card needed): https://console.groq.com/keys
    Then set it as an environment variable before running:
        Windows : setx GROQ_API_KEY "gsk_xxxxxxxx"   (reopen terminal)
        macOS/Linux: export GROQ_API_KEY="gsk_xxxxxxxx"

Run:
    python voxel_groq.py

NOTE: This version needs internet (the LLM call goes to Groq). Vosk + pyttsx3
      remain fully local, so listening and speaking still work without latency
      from the cloud — only the thinking step is online, and it is very fast.
"""

import os, sys, json, queue, re, math, time, datetime, subprocess, threading
import urllib.request, urllib.error
import tkinter as tk
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)

# ─── Config ───────────────────────────────────────────────────────────────────
VOSK_MODEL     = "vosk-model-small-en-us-0.15"
# Groq chat model. Fastest: "llama-3.1-8b-instant". Smarter: "llama-3.3-70b-versatile".
GROQ_MODEL     = "llama-3.1-8b-instant"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "pasteapikeyhere").strip()
SAMPLE_RATE    = 16000
BLOCK_SIZE     = 4000
SILENCE_NEEDED = 12

WAKE_WORDS = ["hey assistant", "hey computer", "wake up", "hey", "hi"]
EXIT_WORDS = ["goodbye", "exit", "quit", "stop assistant"]

SYSTEM_PROMPT = (
    "You are a helpful voice assistant for visually impaired users. "
    "Reply in plain spoken English only — no markdown, no bullets, no asterisks. "
    "Use natural sentences. Keep answers concise. "
    "For recipes or steps, say each step as a full sentence. "
    "Never say 'As an AI'."
)

# ─── Palette ──────────────────────────────────────────────────────────────────
BG         = "#000000"
BG2        = "#080008"
CYAN       = "#00f5ff"
CYAN_DIM   = "#001a1f"
PURPLE     = "#cc00ff"
PURPLE_DIM = "#1a0028"
ORANGE     = "#ff69b4"   # shocking pink replaces orange
WHITE      = "#f0e0ff"
DIM        = "#553366"
BORDER     = "#2a0040"
FONT       = "Franklin Gothic Medium"

# ─── TTS ──────────────────────────────────────────────────────────────────────
_tts = pyttsx3.init()
_tts.setProperty("rate", 155)
_tts.setProperty("volume", 1.0)
for v in _tts.getProperty("voices"):
    if any(k in (v.name + v.id).lower() for k in ["female","zira","hazel","samantha","karen"]):
        _tts.setProperty("voice", v.id)
        break

_speak_lock = threading.Lock()
_last_response = ""

def _clean(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\n\s*[-•\d]+[.)]\s*', '. ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def speak(text):
    text = _clean(text)
    with _speak_lock:
        _tts.say(text)
        _tts.runAndWait()

# ── Hover TTS — reuses main engine, only fires when IDLE ────────────────
_hover_queue = queue.Queue(maxsize=1)

def _hover_worker():
    while True:
        word, app_ref = _hover_queue.get()
        # Only speak if app is fully idle
        if getattr(app_ref, "voice_state", "IDLE") == "IDLE":
            with _speak_lock:
                try:
                    _tts.say(word)
                    _tts.runAndWait()
                except Exception:
                    pass

threading.Thread(target=_hover_worker, daemon=True).start()

def speak_hover(word, app_ref):
    """Only queues if app is IDLE — discards otherwise."""
    if getattr(app_ref, "voice_state", "") != "IDLE":
        return
    try:
        _hover_queue.put_nowait((word, app_ref))
    except queue.Full:
        pass

# ─── Intents ──────────────────────────────────────────────────────────────────
def fast_intent(t):
    global _last_response
    if any(w in t for w in ["what time","current time","time is it"]):
        return f"The time is {datetime.datetime.now().strftime('%I:%M %p')}."
    if any(w in t for w in ["what date","today","what day","date is it"]):
        return f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."
    if t.strip() in ("hello","hi","hey","how are you"):
        return "Hello! Ask me anything."
    if "volume up" in t:
        v = min(1.0, _tts.getProperty("volume") + 0.15)
        _tts.setProperty("volume", v)
        return f"Volume up to {int(v*100)} percent."
    if "volume down" in t or "lower volume" in t:
        v = max(0.1, _tts.getProperty("volume") - 0.15)
        _tts.setProperty("volume", v)
        return f"Volume down to {int(v*100)} percent."
    if "speak faster" in t:
        _tts.setProperty("rate", min(260, _tts.getProperty("rate") + 20))
        return "Speaking faster."
    if "speak slower" in t or "slow down" in t:
        _tts.setProperty("rate", max(90, _tts.getProperty("rate") - 20))
        return "Speaking slower."
    if any(w in t for w in ["repeat","say again"]):
        return _last_response or "Nothing to repeat."
    if "help" in t or "what can you do" in t:
        return "Say hey assistant then ask me anything — recipes, facts, time, or how-to questions."
    return None

_history = []

def ask_groq(user_text):
    if not GROQ_API_KEY:
        return ("No Groq API key found. Get a free key at console dot groq dot com, "
                "then set the GROQ underscore API underscore KEY environment variable.")

    _history.append({"role": "user", "content": user_text})
    # Keep last 6 turns of context, prepend the system prompt
    window = _history[-6:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + window

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 300,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                # Cloudflare (in front of api.groq.com) blocks the default
                # "Python-urllib/3.x" agent with a 403 / error code 1010.
                # Any normal-looking User-Agent gets through.
                "User-Agent": "VOXEL/1.0 (voice-assistant)",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            answer = data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        answer = f"Groq returned an error, code {e.code}. {body}"
    except urllib.error.URLError:
        answer = "I cannot reach Groq. Please check your internet connection."
    except Exception as e:
        answer = f"Error: {e}"

    _history.append({"role": "assistant", "content": answer})
    return answer

def get_response(text):
    global _last_response
    fast = fast_intent(text.lower().strip())
    if fast:
        _last_response = fast
        return fast
    answer = ask_groq(text)
    _last_response = answer
    return answer

# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────
class VoxelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VOXEL")
        self.root.configure(bg=BG)
        self.root.geometry("400x700")
        self.root.resizable(False, False)

        # Animation state
        self._angle      = 0.0
        self._orb_phase  = 0.0
        self._wave_phase = 0.0
        self._blink_on   = True
        self._blink_tick = 0

        # Voice state (set from voice thread, read by animation)
        self.voice_state = "IDLE"   # IDLE | LISTENING | THINKING | SPEAKING

        # Menu state
        self._menu_open   = False
        self._menu_panel  = None
        self._active_page = None   # None | "profile" | "settings"

        # User profile data
        self._prof_name  = tk.StringVar()
        self._prof_email = tk.StringVar()
        self._prof_phone = tk.StringVar()
        self._prof_loc   = tk.StringVar()

        # Voice settings state
        self._voice_gender = tk.StringVar(value="female")  # female | male
        self._conv_tone    = tk.StringVar(value="formal")  # formal | casual
        self._hf_clarity   = tk.BooleanVar(value=False)

        # TTS speed cycle: slow=120, normal=155, fast=210
        self._speed_labels = ["SLOW", "NORMAL", "FAST"]
        self._speed_rates  = [120, 155, 210]
        self._speed_idx    = 1   # default normal

        self._build_ui()
        self._animate()

        # Start voice engine
        threading.Thread(target=self._voice_engine, daemon=True).start()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Nav bar
        nav = tk.Frame(self.root, bg=BG, height=50)
        nav.pack(fill="x")
        nav.pack_propagate(False)
        ham = tk.Canvas(nav, width=24, height=18, bg=BG, highlightthickness=0)
        ham.place(x=18, y=16)
        for y in (1, 8, 15):
            ham.create_line(0, y, 22, y, fill=CYAN, width=1.5)
        ham.bind("<Button-1>", lambda e: self._toggle_menu())
        ham.bind("<Enter>", lambda e: speak_hover("Menu", self))
        tk.Label(nav, text="V O X E L", bg=BG, fg=CYAN,
                 font=("Franklin Gothic Medium", 15, "bold")).place(relx=0.5, rely=0.5, anchor="center")
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # Greeting
        self._greeting_var = tk.StringVar(value="HELLO, ABIHA")
        tk.Label(self.root, textvariable=self._greeting_var, bg=BG, fg=WHITE,
                 font=("Franklin Gothic Medium", 20, "bold")).pack(pady=(20, 4))
        self._subtext_var = tk.StringVar(value="say  'hey assistant'  to begin")
        tk.Label(self.root, textvariable=self._subtext_var,
                 bg=BG, fg=DIM, font=("Franklin Gothic Medium", 10)).pack()

        # Orb canvas
        self.canvas = tk.Canvas(self.root, width=280, height=280,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(pady=8)

        # Status bar
        self._status_frame = tk.Frame(self.root, bg=CYAN_DIM,
                                      highlightbackground=CYAN,
                                      highlightthickness=1)
        self._status_frame.pack(fill="x", padx=20, pady=(0, 10))
        self._status_var = tk.StringVar(value="[ NEURAL INTERFACE READY ]")
        self._status_lbl = tk.Label(self._status_frame,
                                    textvariable=self._status_var,
                                    bg=CYAN_DIM, fg=CYAN,
                                    font=("Franklin Gothic Medium", 11, "bold"), pady=9)
        self._status_lbl.pack()

        # Transcript box
        trans = tk.Frame(self.root, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1)
        trans.pack(fill="x", padx=20, pady=(0, 8))
        self._role_var  = tk.StringVar(value="VOXEL")
        self._trans_var = tk.StringVar(value="Say 'hey assistant' then ask anything.")
        tk.Label(trans, textvariable=self._role_var,
                 bg=BG2, fg=CYAN, font=("Franklin Gothic Medium", 8),
                 anchor="w").pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(trans, textvariable=self._trans_var,
                 bg=BG2, fg=WHITE, font=("Franklin Gothic Medium", 10),
                 wraplength=340, justify="left", anchor="w").pack(
                 fill="x", padx=10, pady=(2, 8))

        # Footer
        foot = tk.Frame(self.root, bg=BG)
        foot.pack(pady=(0, 6))
        self._dot_c = tk.Canvas(foot, width=10, height=10,
                                bg=BG, highlightthickness=0)
        self._dot_c.pack(side="left", padx=(0, 6))
        self._dot_c.create_oval(1, 1, 9, 9, fill=CYAN, outline="", tags="dot")
        tk.Label(foot, text=f"{GROQ_MODEL}  ·  GROQ LPU",
                 bg=BG, fg=DIM, font=("Franklin Gothic Medium", 9)).pack(side="left")

        # ── Floating action buttons ───────────────────────────────────────
        btn_bar = tk.Frame(self.root, bg=BG)
        btn_bar.pack(pady=(4, 10))

        def _make_btn(parent, icon, label, command, accent):
            """Build one floating pill button."""
            outer = tk.Frame(parent, bg=accent,
                             highlightbackground=accent, highlightthickness=1,
                             cursor="hand2")
            outer.pack(side="left", padx=10)
            inner = tk.Frame(outer, bg="#0a0012", padx=12, pady=6)
            inner.pack(padx=1, pady=1)
            tk.Label(inner, text=icon, bg="#0a0012", fg=accent,
                     font=("Franklin Gothic Medium", 13)).pack()
            tk.Label(inner, text=label, bg="#0a0012", fg=accent,
                     font=("Franklin Gothic Medium", 7, "bold")).pack()
            # hover
            def _on(e, lbl=label):
                inner.config(bg=accent)
                for w in inner.winfo_children(): w.config(bg=accent, fg="#000000")
                speak_hover(lbl.split()[0].capitalize(), self)
            def _off(e):
                inner.config(bg="#0a0012")
                for w in inner.winfo_children(): w.config(bg="#0a0012", fg=accent)
            for w in [outer, inner] + inner.winfo_children():
                w.bind("<Button-1>", lambda e, fn=command: fn())
                w.bind("<Enter>", _on)
                w.bind("<Leave>", _off)
            return outer

        # Button 1 — Settings (cyan)
        _make_btn(btn_bar, "⚙", "SETTINGS", self._show_voice_settings, CYAN)

        # Button 2 — Exit (shocking pink)
        def _exit_app():
            speak("Goodbye! Take care.")
            self.root.after(1200, self.root.destroy)
        _make_btn(btn_bar, "✕", "EXIT", _exit_app, "#ff69b4")

        # Button 3 — Speed (purple)
        self._speed_lbl_var = tk.StringVar(value="NORMAL")
        spd_outer = tk.Frame(btn_bar, bg=PURPLE,
                             highlightbackground=PURPLE, highlightthickness=1,
                             cursor="hand2")
        spd_outer.pack(side="left", padx=10)
        spd_inner = tk.Frame(spd_outer, bg="#0a0012", padx=12, pady=6)
        spd_inner.pack(padx=1, pady=1)
        tk.Label(spd_inner, text="⏩", bg="#0a0012", fg=PURPLE,
                 font=("Franklin Gothic Medium", 13)).pack()
        self._spd_txt = tk.Label(spd_inner, textvariable=self._speed_lbl_var,
                                 bg="#0a0012", fg=PURPLE,
                                 font=("Franklin Gothic Medium", 7, "bold"))
        self._spd_txt.pack()
        def _cycle_speed():
            self._speed_idx = (self._speed_idx + 1) % 3
            rate = self._speed_rates[self._speed_idx]
            label = self._speed_labels[self._speed_idx]
            import pyttsx3 as _p
            _tts.setProperty("rate", rate)
            self._speed_lbl_var.set(label)
        def _spd_on(e):
            spd_inner.config(bg=PURPLE)
            for w in spd_inner.winfo_children(): w.config(bg=PURPLE, fg="#000000")
            speak_hover("Speed", self)
        def _spd_off(e):
            spd_inner.config(bg="#0a0012")
            for w in spd_inner.winfo_children(): w.config(bg="#0a0012", fg=PURPLE)
        for w in [spd_outer, spd_inner] + spd_inner.winfo_children():
            w.bind("<Button-1>", lambda e: _cycle_speed())
            w.bind("<Enter>", _spd_on)
            w.bind("<Leave>", _spd_off)


    # ── Side menu ─────────────────────────────────────────────────────────
    def _toggle_menu(self):
        if self._menu_open:
            self._close_menu()
        else:
            self._open_menu()

    def _open_menu(self):
        self._menu_open = True
        panel = tk.Frame(self.root, bg="#080d14", width=270,
                         highlightbackground=CYAN, highlightthickness=1)
        panel.place(x=0, y=0, relheight=1.0)
        panel.lift()
        self._menu_panel = panel

        # Header
        tk.Label(panel, text="SYSTEM MENU", bg="#080d14", fg=CYAN,
                 font=("Franklin Gothic Medium", 10, "bold")).place(x=18, y=18)
        tk.Frame(panel, bg=BORDER, height=1).place(x=0, y=42, width=270)
        close_btn = tk.Label(panel, text="✕", bg="#080d14", fg=CYAN,
                             font=("Franklin Gothic Medium", 14), cursor="hand2")
        close_btn.place(x=238, y=12)
        close_btn.bind("<Button-1>", lambda e: self._close_menu())

        # Menu items
        items = [
            ("◆", "Main Interface",   "VOICE CORE  •  HOME",          self._go_home),
            ("👤", "Account Profile", "SECURE ACCESS  •  ID-882",      self._show_profile),
            ("🎙", "Voice Settings",  "GUIDE  •  TONE",                self._show_voice_settings),
        ]
        for i, (icon, title, sub, cmd) in enumerate(items):
            y = 60 + i * 78
            row = tk.Frame(panel, bg="#0a1018",
                           highlightbackground="#0d2535", highlightthickness=1)
            row.place(x=10, y=y, width=248, height=64)
            tk.Label(row, text=icon, bg="#0a1018", fg=CYAN,
                     font=("Franklin Gothic Medium", 13)).place(x=10, y=12)
            tk.Label(row, text=title, bg="#0a1018", fg=CYAN,
                     font=("Franklin Gothic Medium", 11, "bold")).place(x=38, y=10)
            tk.Label(row, text=sub, bg="#0a1018", fg=DIM,
                     font=("Franklin Gothic Medium", 8)).place(x=38, y=32)
            tk.Label(row, text=">", bg="#0a1018", fg=CYAN,
                     font=("Franklin Gothic Medium", 11)).place(x=226, y=18)
            row.bind("<Button-1>", lambda e, fn=cmd: fn())
            row.bind("<Enter>", lambda e, t=title: speak_hover(t.split()[0], self))
            for w in row.winfo_children():
                w.bind("<Button-1>", lambda e, fn=cmd: fn())
                w.bind("<Enter>", lambda e, t=title: speak_hover(t.split()[0], self))

    def _go_home(self):
        """Close any open panel and return to main interface."""
        if self._menu_panel:
            self._menu_panel.destroy()
            self._menu_panel = None
        self._menu_open   = False
        self._active_page = None

    def _close_menu(self):
        if self._menu_panel:
            self._menu_panel.destroy()
            self._menu_panel = None
        self._menu_open   = False
        self._active_page = None

    def _clear_panel_content(self):
        """Destroy all children below the fixed header row (y > 300)."""
        if not self._menu_panel:
            return
        for w in self._menu_panel.winfo_children():
            try:
                if w.winfo_y() > 280:
                    w.destroy()
            except Exception:
                pass

    def _show_profile(self):
        self._active_page = "profile"
        # Rebuild panel as profile view
        if self._menu_panel:
            self._menu_panel.destroy()
        panel = tk.Frame(self.root, bg="#080d14", width=400,
                         highlightbackground=CYAN, highlightthickness=1)
        panel.place(x=0, y=0, relheight=1.0)
        panel.lift()
        self._menu_panel = panel

        # Header
        tk.Label(panel, text="USER IDENTITY", bg="#080d14", fg=CYAN,
                 font=("Franklin Gothic Medium", 10, "bold")).place(x=18, y=18)
        tk.Frame(panel, bg=BORDER, height=1).place(x=0, y=42, width=400)
        back = tk.Label(panel, text="← BACK", bg="#080d14", fg=CYAN,
                        font=("Franklin Gothic Medium", 9), cursor="hand2")
        back.place(x=300, y=20)
        back.bind("<Button-1>", lambda e: self._open_menu())

        fields = [("NAME", self._prof_name), ("EMAIL", self._prof_email),
                  ("PHONE", self._prof_phone), ("LOCATION", self._prof_loc)]
        for i, (label, var) in enumerate(fields):
            y = 60 + i * 78
            tk.Label(panel, text=label, bg="#080d14", fg=DIM,
                     font=("Franklin Gothic Medium", 8)).place(x=20, y=y)
            entry = tk.Entry(panel, textvariable=var, bg="#0a1018", fg=WHITE,
                             insertbackground=CYAN, relief="flat",
                             font=("Franklin Gothic Medium", 11),
                             highlightbackground=BORDER, highlightthickness=1)
            entry.place(x=20, y=y+18, width=358, height=38)

        # Sync button
        sync = tk.Frame(panel, bg=CYAN, cursor="hand2")
        sync.place(x=20, y=380, width=358, height=44)
        tk.Label(sync, text="SYNCHRONIZE_PROFILE", bg=CYAN, fg=BG,
                 font=("Franklin Gothic Medium", 10, "bold")).place(relx=0.5, rely=0.5, anchor="center")
        sync.bind("<Button-1>", lambda e: self._sync_profile())
        for w in sync.winfo_children():
            w.bind("<Button-1>", lambda e: self._sync_profile())

    def _sync_profile(self):
        name = self._prof_name.get().strip()
        display = f"HELLO, {name.upper()}" if name else "HELLO, ABIHA"
        self._greeting_var.set(display)
        self._close_menu()

    def _show_voice_settings(self):
        self._active_page = "settings"
        if self._menu_panel:
            self._menu_panel.destroy()
        panel = tk.Frame(self.root, bg=BG, width=400,
                         highlightbackground=CYAN, highlightthickness=1)
        panel.place(x=0, y=0, relheight=1.0)
        panel.lift()
        self._menu_panel = panel

        # Header
        tk.Label(panel, text="Settings & Core", bg=BG, fg=WHITE,
                 font=("Franklin Gothic Medium", 14, "bold")).place(relx=0.5, y=18, anchor="n")
        tk.Frame(panel, bg=BORDER, height=1).place(x=0, y=50, width=400)
        back = tk.Label(panel, text="← BACK", bg=BG, fg=CYAN,
                        font=("Franklin Gothic Medium", 9), cursor="hand2")
        back.place(x=300, y=20)
        back.bind("<Button-1>", lambda e: self._open_menu())

        y = 64
        # ── GUIDE VOICE TYPE ─────────────────────────────────────────────
        tk.Label(panel, text="GUIDE VOICE TYPE", bg=BG, fg=CYAN,
                 font=("Franklin Gothic Medium", 8, "bold")).place(x=20, y=y)
        tk.Frame(panel, bg=BORDER, height=1).place(x=20, y=y+16, width=358)
        y += 24

        for gender in ("female", "male"):
            row = tk.Frame(panel, bg="#0a1018",
                           highlightbackground=BORDER, highlightthickness=1)
            row.place(x=20, y=y, width=358, height=44)
            tk.Label(row, text=gender.capitalize(), bg="#0a1018", fg=WHITE,
                     font=("Franklin Gothic Medium", 11)).place(x=14, y=10)
            dot = tk.Canvas(row, width=14, height=14, bg="#0a1018", highlightthickness=0)
            dot.place(x=330, y=14)
            fill = CYAN if self._voice_gender.get() == gender else ""
            dot.create_oval(1, 1, 13, 13, outline=CYAN, width=1.5, fill=fill)
            g = gender  # capture
            row.bind("<Button-1>", lambda e, g=g: self._set_voice_gender(g))
            for w in row.winfo_children():
                w.bind("<Button-1>", lambda e, g=g: self._set_voice_gender(g))
            y += 48

        y += 10
        # ── CONVERSATION TONE ────────────────────────────────────────────
        tk.Label(panel, text="CONVERSATION TONE", bg=BG, fg=CYAN,
                 font=("Franklin Gothic Medium", 8, "bold")).place(x=20, y=y)
        tk.Frame(panel, bg=BORDER, height=1).place(x=20, y=y+16, width=358)
        y += 24

        tone_row = tk.Frame(panel, bg="#0a1018",
                            highlightbackground=BORDER, highlightthickness=1)
        tone_row.place(x=20, y=y, width=358, height=44)
        tone_lbl = tk.Label(tone_row,
                            text="Formal" if self._conv_tone.get() == "formal" else "Casual",
                            bg="#0a1018", fg=WHITE, font=("Franklin Gothic Medium", 11))
        tone_lbl.place(x=14, y=10)
        sw = tk.Label(tone_row, text="SWITCH", bg=CYAN_DIM, fg=CYAN,
                      font=("Franklin Gothic Medium", 8, "bold"),
                      highlightbackground=CYAN, highlightthickness=1,
                      padx=6, pady=3, cursor="hand2")
        sw.place(x=278, y=9)
        sw.bind("<Button-1>", lambda e: self._toggle_tone(tone_lbl))
        y += 58

        # ── ACCESSIBILITY ────────────────────────────────────────────────
        tk.Label(panel, text="ACCESSIBILITY", bg=BG, fg=CYAN,
                 font=("Franklin Gothic Medium", 8, "bold")).place(x=20, y=y)
        tk.Frame(panel, bg=BORDER, height=1).place(x=20, y=y+16, width=358)
        y += 24

        acc_row = tk.Frame(panel, bg="#0a1018",
                           highlightbackground=BORDER, highlightthickness=1)
        acc_row.place(x=20, y=y, width=358, height=44)
        tk.Label(acc_row, text="High-Frequency Clarity", bg="#0a1018",
                 fg=WHITE, font=("Franklin Gothic Medium", 10)).place(x=14, y=10)
        on_col  = CYAN if self._hf_clarity.get() else DIM
        on_text = "ON" if self._hf_clarity.get() else "OFF"
        tog = tk.Label(acc_row, text=on_text, bg=on_col if self._hf_clarity.get() else "#0a1018",
                       fg=BG if self._hf_clarity.get() else DIM,
                       font=("Franklin Gothic Medium", 8, "bold"),
                       highlightbackground=on_col, highlightthickness=1,
                       padx=6, pady=3, cursor="hand2")
        tog.place(x=300, y=9)
        tog.bind("<Button-1>", lambda e: self._toggle_hf_clarity(tog))

    # ── Voice setting appliers ────────────────────────────────────────────
    def _set_voice_gender(self, gender):
        self._voice_gender.set(gender)
        voices = _tts.getProperty("voices")
        female_keys = ["female", "zira", "hazel", "samantha", "karen"]
        if gender == "female":
            for v in voices:
                if any(k in (v.name + v.id).lower() for k in female_keys):
                    _tts.setProperty("voice", v.id)
                    break
        else:
            for v in voices:
                if not any(k in (v.name + v.id).lower() for k in female_keys):
                    _tts.setProperty("voice", v.id)
                    break
        # Refresh panel to update dots
        self._show_voice_settings()

    def _toggle_tone(self, lbl):
        if self._conv_tone.get() == "formal":
            self._conv_tone.set("casual")
            _tts.setProperty("rate", min(220, _tts.getProperty("rate") + 30))
            lbl.config(text="Casual")
        else:
            self._conv_tone.set("formal")
            _tts.setProperty("rate", max(120, _tts.getProperty("rate") - 30))
            lbl.config(text="Formal")

    def _toggle_hf_clarity(self, tog):
        new_val = not self._hf_clarity.get()
        self._hf_clarity.set(new_val)
        if new_val:
            _tts.setProperty("volume", 1.0)
            _tts.setProperty("rate", max(110, _tts.getProperty("rate") - 15))
            tog.config(text="ON", bg=CYAN, fg=BG,
                       highlightbackground=CYAN)
        else:
            _tts.setProperty("rate", min(220, _tts.getProperty("rate") + 15))
            tog.config(text="OFF", bg="#0a1018", fg=DIM,
                       highlightbackground=DIM)

    # ── 3D Glass Orb ──────────────────────────────────────────────────────
    def _draw_orb(self):
        c = self.canvas
        c.delete("all")
        cx, cy = 140, 140
        state  = self.voice_state
        t      = self._angle
        pulse  = math.sin(self._orb_phase)

        # ── per-state colour palette ──────────────────────────────────────
        if state == "LISTENING":
            deep   = "#000d1a"   # darkest core — deep cyan-black
            mid1   = "#001a33"
            mid2   = "#002b55"
            rim1   = "#00f5ff"   # bright cyan rim
            rim2   = "#00aacc"
            shine  = "#aaffff"
            flare  = "#00ffee"
            glow   = "#000a12"
        elif state == "SPEAKING":
            deep   = "#0d0018"
            mid1   = "#1e0038"
            mid2   = "#3a0070"
            rim1   = "#cc00ff"   # shocking purple
            rim2   = "#8800cc"
            shine  = "#f0bbff"
            flare  = "#ff44ff"
            glow   = "#0a0015"
        elif state == "THINKING":
            deep   = "#1a0010"
            mid1   = "#330020"
            mid2   = "#660040"
            rim1   = "#ff69b4" if self._blink_on else DIM   # shocking pink
            rim2   = "#cc2288" if self._blink_on else DIM
            shine  = "#ffbbdd" if self._blink_on else DIM
            flare  = "#ff44aa" if self._blink_on else DIM
            glow   = "#0d0008"
        else:  # IDLE — deep purple-black glass
            deep   = "#050008"
            mid1   = "#0f0025"
            mid2   = "#1e0050"
            rim1   = "#cc00ff"   # vivid purple
            rim2   = "#7700bb"
            shine  = "#eeccff"
            flare  = "#ff69b4"   # shocking pink sparkles
            glow   = "#030006"

        R = 98 + int(3 * pulse)   # sphere radius, gently breathes

        # ── outer glow halo (fake ambient light) ─────────────────────────
        for i, gr in enumerate(range(R + 28, R + 6, -4)):
            alpha = ["#0a0018","#0c001e","#0e0022","#100026","#120028",
                     "#14002c"]
            gc = alpha[min(i, len(alpha)-1)]
            if state == "LISTENING":
                gc_map = ["#001020","#001428","#001830","#001c38","#002040","#002448"]
                gc = gc_map[min(i, len(gc_map)-1)]
            elif state == "SPEAKING":
                gc_map = ["#0a0018","#0e001e","#120026","#16002e","#1a0036","#1e003e"]
                gc = gc_map[min(i, len(gc_map)-1)]
            c.create_oval(cx-gr, cy-gr, cx+gr, cy+gr, fill=gc, outline="")

        # ── sphere body — 22 concentric ovals for depth gradient ──────────
        steps = 22
        for i in range(steps, 0, -1):
            frac = i / steps          # 1.0 = outermost, 0 = center
            r    = int(R * frac)

            # interpolate deep→mid1→mid2 from center outward
            if frac < 0.45:
                t2 = frac / 0.45
                ri = int(int(deep[1:3],16)*(1-t2) + int(mid1[1:3],16)*t2)
                gi = int(int(deep[3:5],16)*(1-t2) + int(mid1[3:5],16)*t2)
                bi = int(int(deep[5:7],16)*(1-t2) + int(mid1[5:7],16)*t2)
            else:
                t2 = (frac - 0.45) / 0.55
                ri = int(int(mid1[1:3],16)*(1-t2) + int(mid2[1:3],16)*t2)
                gi = int(int(mid1[3:5],16)*(1-t2) + int(mid2[3:5],16)*t2)
                bi = int(int(mid1[5:7],16)*(1-t2) + int(mid2[5:7],16)*t2)

            col = f"#{ri:02x}{gi:02x}{bi:02x}"
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline="")

        # ── rim glow — rotating arc around the sphere edge ────────────────
        rim_start = (t * 55) % 360
        c.create_arc(cx-R, cy-R, cx+R, cy+R,
                     start=rim_start, extent=160,
                     style="arc", outline=rim1, width=3)
        c.create_arc(cx-R, cy-R, cx+R, cy+R,
                     start=(rim_start+180)%360, extent=110,
                     style="arc", outline=rim2, width=2)
        # thin counter-rotating accent
        c.create_arc(cx-R+4, cy-R+4, cx+R-4, cy+R-4,
                     start=(-t*40+90)%360, extent=70,
                     style="arc", outline=rim1, width=1)

        # ── internal shimmer arcs (looks like light inside the glass) ─────
        ir = int(R * 0.72)
        c.create_arc(cx-ir, cy-ir, cx+ir, cy+ir,
                     start=(t*70+30)%360, extent=80,
                     style="arc", outline=rim2, width=1)
        c.create_arc(cx-ir, cy-ir, cx+ir, cy+ir,
                     start=(-t*50+200)%360, extent=55,
                     style="arc", outline=rim1, width=1)

        # ── floating bubble dots inside the sphere ────────────────────────
        import random
        rng = random.Random(42)   # fixed seed → stable positions
        for _ in range(10):
            bx = cx + rng.uniform(-0.65, 0.65) * R
            by = cy + rng.uniform(-0.65, 0.65) * R
            br = rng.uniform(2.5, 5.5)
            # only draw if inside sphere
            if (bx-cx)**2 + (by-cy)**2 < (R*0.78)**2:
                # shimmer: vary opacity via brightness cycle
                phase_offset = rng.uniform(0, 6.28)
                bright = 0.35 + 0.65 * abs(math.sin(self._orb_phase * 1.3 + phase_offset))
                ri2 = int(int(rim1[1:3],16) * bright)
                gi2 = int(int(rim1[3:5],16) * bright)
                bi2 = int(int(rim1[5:7],16) * bright)
                bcol = f"#{ri2:02x}{gi2:02x}{bi2:02x}"
                c.create_oval(bx-br, by-br, bx+br, by+br, fill=bcol, outline="")

        # ── pink/magenta flare dots (the bright sparkles in the image) ────
        rng2 = random.Random(99)
        for _ in range(5):
            fx = cx + rng2.uniform(-0.5, 0.5) * R
            fy = cy + rng2.uniform(-0.5, 0.5) * R
            fr = rng2.uniform(1.5, 3.5)
            phase2 = rng2.uniform(0, 6.28)
            bright2 = abs(math.sin(self._orb_phase * 2.1 + phase2))
            if bright2 > 0.45 and (fx-cx)**2+(fy-cy)**2 < (R*0.7)**2:
                c.create_oval(fx-fr, fy-fr, fx+fr, fy+fr, fill=flare, outline="")

        # ── glass highlight — top-left bright blob (key to 3D illusion) ──
        hx = cx - int(R * 0.38)
        hy = cy - int(R * 0.38)
        # soft outer highlight
        for hr in range(22, 6, -3):
            alpha_val = int(255 * (1 - hr/24) * 0.55)
            # fake it with stepping from shine toward mid2
            t3 = 1 - hr/22
            sr = int(int(shine[1:3],16)*t3 + int(mid2[1:3],16)*(1-t3))
            sg = int(int(shine[3:5],16)*t3 + int(mid2[3:5],16)*(1-t3))
            sb = int(int(shine[5:7],16)*t3 + int(mid2[5:7],16)*(1-t3))
            hcol = f"#{sr:02x}{sg:02x}{sb:02x}"
            c.create_oval(hx-hr, hy-hr, hx+hr, hy+hr, fill=hcol, outline="")
        # bright core of highlight
        c.create_oval(hx-5, hy-5, hx+5, hy+5, fill=shine, outline="")

        # ── small secondary highlight bottom-right edge ───────────────────
        hx2 = cx + int(R * 0.55)
        hy2 = cy + int(R * 0.48)
        if (hx2-cx)**2+(hy2-cy)**2 < R**2:
            c.create_oval(hx2-4, hy2-4, hx2+4, hy2+4, fill=rim1, outline="")

        # ── waveform bars when LISTENING (rendered on top) ────────────────
        if state == "LISTENING":
            bars, bw, sp = 9, 4, 6
            total = bars * (bw + sp) - sp
            bx0 = cx - total // 2
            by0 = cy
            for i in range(bars):
                h = 7 + 20 * abs(math.sin(self._wave_phase + i * 0.55))
                x0 = bx0 + i * (bw + sp)
                c.create_rectangle(x0, by0-h, x0+bw, by0+h,
                                   fill="#00ffee", outline="")

    # ── Animation loop ────────────────────────────────────────────────────
    def _animate(self):
        state = self.voice_state

        speeds = {"IDLE": 0.025, "LISTENING": 0.07, "THINKING": 0.018, "SPEAKING": 0.045}
        self._angle      += speeds.get(state, 0.025)
        self._orb_phase  += 0.06 if state == "IDLE" else 0.13
        self._wave_phase += 0.22

        self._blink_tick += 1
        if self._blink_tick >= 12:
            self._blink_on   = not self._blink_on
            self._blink_tick = 0

        self._draw_orb()

        # Footer dot heartbeat
        dot_col = CYAN if int(time.time()*1.5) % 2 == 0 else DIM
        self._dot_c.itemconfig("dot", fill=dot_col)

        self.root.after(40, self._animate)  # 25 fps

    # ── Thread-safe UI updates ────────────────────────────────────────────
    def _ui(self, fn):
        self.root.after(0, fn)

    def set_state(self, state, status_text):
        def _do():
            self.voice_state = state
            self._status_var.set(status_text)
            if state == "IDLE":
                self._status_frame.config(bg="#0a0015", highlightbackground=PURPLE)
                self._status_lbl.config(bg="#0a0015", fg=PURPLE)
                self._subtext_var.set("say  'hey assistant'  to begin")
            elif state == "LISTENING":
                self._status_frame.config(bg="#001a1f", highlightbackground=CYAN)
                self._status_lbl.config(bg="#001a1f", fg=CYAN)
                self._subtext_var.set("listening...")
            elif state == "THINKING":
                self._status_frame.config(bg="#1a0018", highlightbackground="#ff69b4")
                self._status_lbl.config(bg="#1a0018", fg="#ff69b4")
                self._subtext_var.set("thinking...")
            elif state == "SPEAKING":
                self._status_frame.config(bg=PURPLE_DIM, highlightbackground="#cc00ff")
                self._status_lbl.config(bg=PURPLE_DIM, fg="#cc00ff")
                self._subtext_var.set("speaking...")
        self._ui(_do)

    def set_transcript(self, role, text):
        def _do():
            self._role_var.set(role)
            self._trans_var.set(text[:180] + ("…" if len(text) > 180 else ""))
        self._ui(_do)

    # ── Voice engine ──────────────────────────────────────────────────────
    def _voice_engine(self):
        if not os.path.exists(VOSK_MODEL):
            self.set_transcript("ERROR",
                f"Vosk model not found: '{VOSK_MODEL}'. "
                "Download vosk-model-small-en-us-0.15 from alphacephei.com/vosk/models")
            return

        self.set_state("IDLE", "[ LOADING MODEL... ]")
        rec = KaldiRecognizer(Model(VOSK_MODEL), SAMPLE_RATE)
        rec.SetWords(True)
        self.set_state("IDLE", "[ NEURAL INTERFACE READY ]")
        self.set_transcript("VOXEL", "Say 'hey assistant' then ask anything.")

        threading.Thread(target=speak,
            args=("Voice assistant ready. Say hey assistant then ask your question.",),
            daemon=True).start()

        audio_q       = queue.Queue()
        mode          = "IDLE"
        collected     = ""
        silence_count = 0

        def audio_cb(indata, frames, time_info, status):
            audio_q.put(bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                               dtype="int16", channels=1, callback=audio_cb):
            while True:
                # BUSY: drain audio while TTS is speaking
                if mode == "BUSY":
                    try: audio_q.get_nowait()
                    except queue.Empty: pass
                    continue

                try:
                    data = audio_q.get(timeout=0.3)
                except queue.Empty:
                    # Silence timeout — fire command if we have one
                    if mode == "COLLECTING":
                        silence_count += 1
                        if silence_count >= SILENCE_NEEDED and collected.strip():
                            cmd = collected.strip()
                            collected = ""; silence_count = 0; mode = "BUSY"
                            self._handle_command(cmd, rec, audio_q)
                            mode = "IDLE"
                            self.set_state("IDLE","[ NEURAL INTERFACE READY ]")
                    continue

                is_final = rec.AcceptWaveform(data)
                text = (json.loads(rec.Result()) if is_final
                        else json.loads(rec.PartialResult()))
                text = text.get("text" if is_final else "partial", "").strip().lower()

                if not text:
                    if mode == "COLLECTING":
                        silence_count += 1
                        if silence_count >= SILENCE_NEEDED and collected.strip():
                            cmd = collected.strip()
                            collected = ""; silence_count = 0; mode = "BUSY"
                            self._handle_command(cmd, rec, audio_q)
                            mode = "IDLE"
                            self.set_state("IDLE","[ NEURAL INTERFACE READY ]")
                    continue

                silence_count = 0

                if mode == "IDLE":
                    if any(ww in text for ww in WAKE_WORDS):
                        remainder = text
                        for ww in WAKE_WORDS:
                            remainder = remainder.replace(ww, "").strip()
                        if remainder and len(remainder.split()) >= 2:
                            # Full command in same breath
                            mode = "BUSY"
                            self._handle_command(remainder, rec, audio_q)
                            mode = "IDLE"
                            self.set_state("IDLE","[ NEURAL INTERFACE READY ]")
                        else:
                            # Wake word only — now collect
                            collected = remainder
                            silence_count = 0
                            mode = "COLLECTING"
                            self.set_state("LISTENING","[ LISTENING — SPEAK NOW ]")
                            self.set_transcript("YOU","...")
                            threading.Thread(target=speak, args=("Yes?",),
                                             daemon=True).start()

                elif mode == "COLLECTING":
                    silence_count = 0
                    if is_final:
                        collected += " " + text
                        self.set_transcript("YOU", collected.strip())

    def _handle_command(self, command, rec, audio_q):
        self.set_transcript("YOU", command)
        self.set_state("THINKING","[ PROCESSING NEURAL INPUT ]")

        if any(w in command.lower() for w in EXIT_WORDS):
            speak("Goodbye! Take care.")
            self.root.after(1500, self.root.destroy)
            return

        response = get_response(command)
        self.set_state("SPEAKING","[ VOXEL IS SPEAKING ]")
        self.set_transcript("VOXEL", response)
        speak(response)
        rec.Reset()
        audio_q.queue.clear()


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.4)
    except Exception:
        pass
    app = VoxelApp(root)
    root.mainloop()
