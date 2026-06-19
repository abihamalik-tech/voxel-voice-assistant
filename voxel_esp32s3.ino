/* ============================================================================
   VOXEL-S3  —  ESP32-S3 Voice Assistant powered by Groq
   ============================================================================
   Push-to-talk voice assistant on a single ESP32-S3. The whole brain lives on
   Groq Cloud (all three stages run on Groq's LPU, so it is very fast):

       INMP441 mic  --I2S-->  ESP32-S3  --HTTPS-->  Groq Whisper   (speech -> text)
                                        --HTTPS-->  Groq LLM       (text  -> reply)
                                        --HTTPS-->  Groq PlayAI TTS (reply -> audio)
       ESP32-S3  --I2S-->  MAX98357A  -->  Speaker                 (audio out)
       SSD1306 OLED shows status + transcript.

   HOW TO USE:
     1. Hold the BOOT button (GPIO0) and speak. Release when done.
     2. It transcribes -> thinks -> speaks the answer through the speaker.

   ----------------------------------------------------------------------------
   BOARD / IDE SETUP (Arduino IDE)
   ----------------------------------------------------------------------------
   - Install "esp32" boards package (Espressif) v3.x  (gives the ESP_I2S library).
   - Board:  "ESP32S3 Dev Module"
   - PSRAM:  N16R8  -> "OPI PSRAM"      |   N8R2 -> "QSPI PSRAM"
             (PSRAM is REQUIRED — the audio buffers do not fit in internal RAM.)
   - USB CDC On Boot: "Enabled"  (so Serial works over the USB port)
   - Libraries to install via Library Manager:
         * ArduinoJson  v7.x   (Benoit Blanchon — v6 syntax will NOT compile)
         * Adafruit SSD1306
         * Adafruit GFX Library
     (WiFi, HTTPClient, WiFiClientSecure, Wire, ESP_I2S all ship with the core.)

   ----------------------------------------------------------------------------
   GROQ SETUP
   ----------------------------------------------------------------------------
   - Free key (no card): https://console.groq.com/keys
   - PlayAI TTS requires a one-time terms acceptance on the model page:
         https://console.groq.com/playground   (open a PlayAI TTS model once)
   - Fill in WIFI_SSID / WIFI_PASS / GROQ_API_KEY below.

   NOTE: I could not compile this for you in my sandbox (no ESP32 toolchain),
         so build + flash it from the Arduino IDE. The structure and the Groq
         endpoints/model names are current as of June 2026.
   ============================================================================ */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP_I2S.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ───────────────────────── USER CONFIG ──────────────────────────────────────
const char* WIFI_SSID    = "YOUR_WIFI_NAME";
const char* WIFI_PASS    = "YOUR_WIFI_PASSWORD";
const char* GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx";

const char* CHAT_MODEL = "llama-3.1-8b-instant";     // fastest; swap for llama-3.3-70b-versatile if you want smarter
const char* STT_MODEL  = "whisper-large-v3-turbo";   // Groq speech-to-text
const char* TTS_MODEL  = "playai-tts";               // Groq PlayAI TTS
const char* TTS_VOICE  = "Fritz-PlayAI";             // any PlayAI voice name

const char* SYSTEM_PROMPT =
  "You are VOXEL, a helpful voice assistant. Reply in plain spoken English, "
  "no markdown or symbols. Keep answers short, ideally one or two sentences.";

// ───────────────────────── PIN MAP (ESP32-S3 DevKitC) ───────────────────────
// Chosen to avoid strapping pins (0/3/45/46), USB pins (19/20) and the
// flash/PSRAM pins (26-37). All of these are safe GPIOs on the S3.
#define MIC_SCK   4    // INMP441 SCK  (bit clock)
#define MIC_WS    5    // INMP441 WS   (word select / LRCL)
#define MIC_SD    6    // INMP441 SD   (data out -> ESP in)

#define AMP_BCLK  15   // MAX98357 BCLK
#define AMP_LRC   16   // MAX98357 LRC (word select)
#define AMP_DIN   7    // MAX98357 DIN (data in  <- ESP out)

#define I2C_SDA   8    // SSD1306 SDA
#define I2C_SCL   9    // SSD1306 SCL

#define BTN_PIN   0    // BOOT button = push-to-talk (active LOW)

// ───────────────────────── AUDIO CONFIG ─────────────────────────────────────
#define SAMPLE_RATE     16000   // Whisper works best at 16 kHz mono
#define MAX_REC_SECONDS 6       // hard cap on a single recording
#define MIC_GAIN        4       // INMP441 is quiet; raise/lower if needed (1-12)

// ───────────────────────── OLED 0.91" 128x32 ────────────────────────────────
#define OLED_W 128
#define OLED_H 32
#define OLED_ADDR 0x3C
Adafruit_SSD1306 oled(OLED_W, OLED_H, &Wire, -1);

// ───────────────────────── I2S DEVICES ──────────────────────────────────────
I2SClass i2s_mic;   // input  (INMP441)
I2SClass i2s_amp;   // output (MAX98357)

WiFiClientSecure tls;   // shared TLS client for all Groq calls

// ─────────────────────────────────────────────────────────────────────────────
//  OLED helper
// ─────────────────────────────────────────────────────────────────────────────
void oledShow(const String& title, const String& body = "") {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println(title);
  if (body.length()) {
    oled.setCursor(0, 12);
    // wrap roughly 21 chars per line on a 128px wide font-size-1 display
    String w = body;
    int line = 12;
    while (w.length() && line <= 24) {
      oled.setCursor(0, line);
      oled.println(w.substring(0, 21));
      w = w.substring(min((int)w.length(), 21));
      line += 10;
    }
  }
  oled.display();
}

// ─────────────────────────────────────────────────────────────────────────────
//  WAV header writer (44-byte canonical PCM header)
// ─────────────────────────────────────────────────────────────────────────────
void writeWavHeader(uint8_t* h, uint32_t pcmBytes, uint32_t rate,
                    uint16_t ch, uint16_t bits) {
  uint32_t byteRate   = rate * ch * bits / 8;
  uint16_t blockAlign = ch * bits / 8;
  uint32_t chunkSize  = 36 + pcmBytes;
  memcpy(h,      "RIFF", 4);
  memcpy(h + 4,  &chunkSize, 4);
  memcpy(h + 8,  "WAVE", 4);
  memcpy(h + 12, "fmt ", 4);
  uint32_t sub1 = 16; memcpy(h + 16, &sub1, 4);
  uint16_t fmt  = 1;  memcpy(h + 20, &fmt, 2);
  memcpy(h + 22, &ch, 2);
  memcpy(h + 24, &rate, 4);
  memcpy(h + 28, &byteRate, 4);
  memcpy(h + 32, &blockAlign, 2);
  memcpy(h + 34, &bits, 2);
  memcpy(h + 36, "data", 4);
  memcpy(h + 40, &pcmBytes, 4);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RECORD: capture mic while the button is held -> 16-bit mono WAV in PSRAM
//  Returns malloc'd buffer (caller frees) and sets *outSize.
// ─────────────────────────────────────────────────────────────────────────────
uint8_t* recordWhileHeld(size_t* outSize) {
  const uint32_t maxSamples = SAMPLE_RATE * MAX_REC_SECONDS;
  const uint32_t maxPcm     = maxSamples * 2;            // 16-bit mono
  uint8_t* wav = (uint8_t*) ps_malloc(44 + maxPcm);
  if (!wav) { Serial.println("PSRAM alloc failed"); return nullptr; }
  int16_t* pcm = (int16_t*)(wav + 44);

  const size_t CH = 256;
  int32_t buf[CH];
  uint32_t got = 0;

  // read until the button is released or we hit the cap
  while (digitalRead(BTN_PIN) == LOW && got < maxSamples) {
    size_t want = min((size_t)(maxSamples - got), CH);
    size_t n = i2s_mic.readBytes((char*)buf, want * sizeof(int32_t));
    size_t s = n / sizeof(int32_t);
    for (size_t i = 0; i < s; i++) {
      // INMP441 puts 24-bit data left-justified in a 32-bit slot.
      // Take the top 16 bits, then apply a little gain and clamp.
      int32_t v = (buf[i] >> 16) * MIC_GAIN;
      if (v > 32767)  v = 32767;
      if (v < -32768) v = -32768;
      pcm[got++] = (int16_t)v;
    }
  }

  uint32_t pcmBytes = got * 2;
  writeWavHeader(wav, pcmBytes, SAMPLE_RATE, 1, 16);
  *outSize = 44 + pcmBytes;
  Serial.printf("Recorded %u samples (%.1f s)\n", got, got / (float)SAMPLE_RATE);
  return wav;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROQ STT: multipart upload the WAV to Whisper, return the transcript text.
// ─────────────────────────────────────────────────────────────────────────────
String groqTranscribe(uint8_t* wav, size_t wavLen) {
  const char* B = "----voxelS3boundary";
  String pre =
    String("--") + B + "\r\n"
    "Content-Disposition: form-data; name=\"model\"\r\n\r\n" + STT_MODEL + "\r\n"
    "--" + B + "\r\n"
    "Content-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n"
    "--" + B + "\r\n"
    "Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
    "Content-Type: audio/wav\r\n\r\n";
  String post = String("\r\n--") + B + "--\r\n";

  size_t total = pre.length() + wavLen + post.length();
  uint8_t* body = (uint8_t*) ps_malloc(total);
  if (!body) { Serial.println("STT body alloc failed"); return ""; }
  size_t o = 0;
  memcpy(body + o, pre.c_str(), pre.length());   o += pre.length();
  memcpy(body + o, wav, wavLen);                 o += wavLen;
  memcpy(body + o, post.c_str(), post.length()); o += post.length();

  HTTPClient http;
  http.begin(tls, "https://api.groq.com/openai/v1/audio/transcriptions");
  http.addHeader("Authorization", String("Bearer ") + GROQ_API_KEY);
  http.addHeader("Content-Type", String("multipart/form-data; boundary=") + B);
  int code = http.POST(body, total);
  String out;
  if (code == 200) {
    JsonDocument doc;
    if (!deserializeJson(doc, http.getStream()))
      out = doc["text"].as<String>();
  } else {
    Serial.printf("STT HTTP %d: %s\n", code, http.getString().c_str());
  }
  http.end();
  free(body);
  out.trim();
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROQ CHAT: send the transcript, return the assistant reply.
// ─────────────────────────────────────────────────────────────────────────────
String groqChat(const String& userText) {
  JsonDocument req;
  req["model"]       = CHAT_MODEL;
  req["temperature"] = 0.7;
  req["max_tokens"]  = 160;
  JsonArray msgs = req["messages"].to<JsonArray>();
  JsonObject sys = msgs.add<JsonObject>();
  sys["role"] = "system"; sys["content"] = SYSTEM_PROMPT;
  JsonObject usr = msgs.add<JsonObject>();
  usr["role"] = "user";   usr["content"] = userText;

  String body;
  serializeJson(req, body);

  HTTPClient http;
  http.begin(tls, "https://api.groq.com/openai/v1/chat/completions");
  http.addHeader("Authorization", String("Bearer ") + GROQ_API_KEY);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);

  String answer;
  if (code == 200) {
    JsonDocument res;
    DeserializationError err = deserializeJson(res, http.getStream());
    if (!err) answer = res["choices"][0]["message"]["content"].as<String>();
  } else {
    Serial.printf("CHAT HTTP %d: %s\n", code, http.getString().c_str());
  }
  http.end();
  answer.trim();
  return answer;
}

// ─────────────────────────────────────────────────────────────────────────────
//  GROQ TTS: stream the WAV response straight to the MAX98357 (low memory).
// ─────────────────────────────────────────────────────────────────────────────
void groqSpeak(const String& text) {
  JsonDocument req;
  req["model"]           = TTS_MODEL;
  req["voice"]           = TTS_VOICE;
  req["input"]           = text;
  req["response_format"] = "wav";
  String body;
  serializeJson(req, body);

  HTTPClient http;
  http.begin(tls, "https://api.groq.com/openai/v1/audio/speech");
  http.addHeader("Authorization", String("Bearer ") + GROQ_API_KEY);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code != 200) {
    Serial.printf("TTS HTTP %d: %s\n", code, http.getString().c_str());
    http.end();
    return;
  }

  // Buffer the whole WAV into PSRAM, then let the library parse + play it.
  // (More robust than hand-skipping the header: handles any header length
  //  and chunked transfer encoding.)
  WiFiClient* s = http.getStreamPtr();
  int   contentLen = http.getSize();                 // -1 if chunked
  size_t cap = (contentLen > 0) ? (size_t)contentLen : 65536;
  uint8_t* buf = (uint8_t*) ps_malloc(cap);
  if (!buf) { Serial.println("TTS buffer alloc failed"); http.end(); return; }

  size_t len = 0;
  uint32_t t0 = millis();
  while ((http.connected() || s->available()) && millis() - t0 < 30000) {
    size_t avail = s->available();
    if (avail) {
      if (len + avail > cap) {                        // grow if needed
        cap = len + avail + 16384;
        uint8_t* nb = (uint8_t*) ps_realloc(buf, cap);
        if (!nb) { free(buf); http.end(); return; }
        buf = nb;
      }
      len += s->readBytes(buf + len, avail);
      t0 = millis();
    } else if (!http.connected()) {
      break;
    } else {
      delay(2);
    }
  }
  http.end();

  if (len > 44) {
    // Sample rate lives at byte 24 of the canonical WAV header.
    uint32_t rate = buf[24] | (buf[25] << 8) | (buf[26] << 16) | ((uint32_t)buf[27] << 24);
    if (rate < 8000 || rate > 48000) rate = 24000;   // sane fallback
    i2s_amp.setPins(AMP_BCLK, AMP_LRC, AMP_DIN, -1, -1);
    i2s_amp.begin(I2S_MODE_STD, rate, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
    i2s_amp.playWAV(buf, len);                        // library finds the data chunk
    i2s_amp.end();                                    // release channel for the mic
  }
  free(buf);
}

// ─────────────────────────────────────────────────────────────────────────────
//  One full interaction
// ─────────────────────────────────────────────────────────────────────────────
void runConversation() {
  oledShow("Listening...", "hold BOOT & speak");
  size_t wavLen = 0;
  uint8_t* wav = recordWhileHeld(&wavLen);
  if (!wav || wavLen < 44 + 8000) {            // need > ~0.25 s of audio
    if (wav) free(wav);
    oledShow("Too short", "try again");
    delay(800);
    return;
  }

  oledShow("Transcribing...");
  String heard = groqTranscribe(wav, wavLen);
  free(wav);
  Serial.println("YOU: " + heard);
  if (heard.length() == 0) { oledShow("Didn't catch that"); delay(900); return; }
  oledShow("You said:", heard);

  oledShow("Thinking...");
  String reply = groqChat(heard);
  Serial.println("VOXEL: " + reply);
  if (reply.length() == 0) reply = "Sorry, I had trouble answering.";
  oledShow("VOXEL:", reply);

  groqSpeak(reply);
  oledShow("Ready", "hold BOOT to talk");
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(BTN_PIN, INPUT_PULLUP);

  // OLED
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 not found");
  }
  oled.clearDisplay();
  oledShow("VOXEL-S3", "booting...");

  // Mic: 32-bit mono, LEFT slot (INMP441 L/R pin tied to GND = left).
  i2s_mic.setPins(MIC_SCK, MIC_WS, -1, MIC_SD, -1);   // bclk, ws, dout(none), din, mclk
  if (!i2s_mic.begin(I2S_MODE_STD, SAMPLE_RATE,
                     I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO,
                     I2S_STD_SLOT_LEFT)) {
    Serial.println("I2S mic init failed");
  }

  // WiFi
  oledShow("WiFi", "connecting...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t < 20000) { delay(250); Serial.print("."); }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) { oledShow("WiFi FAILED", "check creds"); }
  else { Serial.println("WiFi: " + WiFi.localIP().toString()); }

  tls.setInsecure();   // skip cert validation (fine for a hobby device)

  oledShow("Ready", "hold BOOT to talk");
}

void loop() {
  // Push-to-talk: a press starts a conversation.
  if (digitalRead(BTN_PIN) == LOW) {
    delay(30);                                   // debounce
    if (digitalRead(BTN_PIN) == LOW) {
      runConversation();
      while (digitalRead(BTN_PIN) == LOW) delay(10);   // wait for release
    }
  }
  delay(10);
}
