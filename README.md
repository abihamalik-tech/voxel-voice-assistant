# VOXEL — AI Voice Assistant 🎙️

A multithreaded AI voice assistant built for **accessibility**, aimed at visually impaired users. VOXEL listens, thinks, and speaks — combining on-device speech recognition and synthesis with cloud-based LLM reasoning, so responses feel instant while the heavy thinking happens in the cloud.

It runs as a **desktop application** and as a **standalone hardware device** built on an ESP32-S3, with its own microphone, speaker, OLED display, and push-to-talk button.

<!-- Add a screenshot or short demo GIF here — it massively increases how impressive this looks. -->
<!-- ![VOXEL demo](demo.gif) -->

---

##  Features

- **4-state interaction model** — idle → listening → thinking → speaking, with wake-word / push-to-talk activation
- **Multithreaded core** — audio capture, transcription, LLM calls, and speech playback run concurrently, so the interface never freezes
- **Hybrid speech pipeline** — on-device recognition (Vosk) and synthesis (pyttsx3) paired with a cloud LLM (Groq / Llama) for low-latency responses
- **Offline fast-intent commands** — time, date, volume, speech rate, and repeat resolve instantly with no network call
- **Rolling conversation context** — remembers recent turns for natural follow-up questions
- **Accessibility-first UI** — live audio waveform, animated glass orb, and hover-to-speak labels
- **Custom hardware build** — a self-contained ESP32-S3 device (mic + amp + speaker + OLED + button)

---

##  Software Setup (Desktop)

**Requirements:** Python 3.9+

```bash
# Install dependencies
pip install vosk pyttsx3 groq sounddevice numpy
```

**Set your API key as an environment variable** (never hard-code it in the file):

```bash
# Windows
setx GROQ_API_KEY "your-key-here"

# macOS / Linux
export GROQ_API_KEY="your-key-here"
```

**Run it:**

```bash
python voxel1234_custom.py
```

> The code reads the key with `os.environ.get("GROQ_API_KEY")`, so your secret stays out of the source. Adjust the dependency list above to match the imports in your file.

---

## 🔌 Hardware Build (ESP32-S3)

VOXEL-S3 runs the assistant on dedicated hardware. Flash `voxel_esp32s3.ino` using the Arduino IDE.

**Components**

| Part | Role |
|------|------|
| ESP32-S3 DevKitC (N8R2 / N16R8) | Main controller (needs PSRAM for audio buffers) |
| INMP441 | I2S MEMS microphone (audio in) |
| MAX98357A | I2S amplifier (audio out) |
| SSD1306 0.91" OLED (128×32) | Status display, I2C addr `0x3C` |
| 3W 4/8Ω speaker | Output |
| Push button on BOOT (GPIO0) | Push-to-talk |
| USB-C | Power + flashing |

**Wiring**

| Module | Connection |
|--------|-----------|
| **INMP441** (3.3 V) | VDD→3V3, GND→GND, SCK→GPIO4, WS→GPIO5, SD→GPIO6, L/R→GND |
| **MAX98357A** (5 V) | VIN→5V, GND→GND, DIN→GPIO7, BCLK→GPIO15, LRC→GPIO16, SD→3V3, GAIN→float, SPK+/SPK−→speaker |
| **SSD1306 OLED** (3.3 V, I2C) | VCC→3V3, GND→GND, SDA→GPIO8, SCL→GPIO9 |

**Power:** USB-C powers the board; the 5V pin feeds the MAX98357A, 3V3 feeds the mic and OLED. All grounds are common.

> Full wiring diagram: `voxel_esp32s3_wiring.svg` (add the file to this repo to display it below).
<!-- ![Wiring diagram](voxel_esp32s3_wiring.svg) -->

**Arduino notes**
- Enable **PSRAM** in the board settings (audio buffers need it).
- Pins avoid strapping (0/3/45/46), USB (19/20), and PSRAM/flash (26–37). You can remap any GPIO in the `.ino` — the S3 routes I2S/I2C on most pins.

---

## Tech Stack

`Python` · `Vosk` · `pyttsx3` · `Groq / LLM API` · `Tkinter` · `multithreading` · `ESP32-S3` · `I2S` · `Arduino / C++`

---

## 📫 Contact

Built by **Abiha Ajam Malik** — abihajam@gmail.com
