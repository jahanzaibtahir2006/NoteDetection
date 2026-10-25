"""Speech output (Microsoft neural voices via edge-tts) and speech input (faster-whisper)."""
import asyncio
import io
import os
import queue
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import edge_tts
import numpy as np
import pygame
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK = 480            # 30 ms of audio
END_SILENCE = 26       # ~0.8 s of quiet ends a sentence
MIN_SPEECH = 12        # ignore sounds shorter than ~0.4 s
MAX_SPEECH = 333       # cut very long sentences at ~10 s
HALLUCINATIONS = ("thank you for watching", "subscribe", "thanks for watching")


class Speaker:
    """Speaks one sentence at a time. Newer announcements replace queued ones."""

    def __init__(self, voice="en-US-AndrewMultilingualNeural"):
        self.voice = voice
        self.jobs = queue.Queue()
        self.speaking = threading.Event()
        pygame.mixer.init()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text, interrupt=False):
        while interrupt and not self.jobs.empty():
            self.jobs.get_nowait()
        self.jobs.put(text)

    async def _synthesize(self, text):
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        return bytes(audio)

    def _run(self):
        while True:
            text = self.jobs.get()
            self.speaking.set()
            try:
                pygame.mixer.music.load(io.BytesIO(asyncio.run(self._synthesize(text))), "mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:  # keep the app running without internet
                print("Speech output failed:", e)
            time.sleep(0.3)  # let the room go quiet so the mic does not hear us
            self.speaking.clear()


class Listener:
    """Listens on the microphone and passes each finished sentence, in English, to on_text.

    Whisper's translate task turns Urdu, Punjabi or English speech into English text,
    so the user can talk in whatever language is natural to them.
    """

    PROMPT = "How much money is this? What's the total? Which note is this? Say that again."

    def __init__(self, on_text, speaker, model="small", device="cpu"):
        self.on_text = on_text
        self.speaker = speaker
        # Runs on the CPU so the 4 GB GPU stays free for YOLO
        self.whisper = WhisperModel(model, device=device, compute_type="int8")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        blocks = queue.Queue()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                            callback=lambda data, *_: blocks.put(data[:, 0].copy())):
            # Measure one second of room noise to set the speech threshold
            noise = np.mean([np.sqrt(np.mean(blocks.get() ** 2)) for _ in range(33)])
            threshold = max(noise * 3, 0.01)
            speech, silent = [], 0
            while True:
                block = blocks.get()
                if self.speaker and self.speaker.speaking.is_set():
                    speech, silent = [], 0
                    continue
                loud = np.sqrt(np.mean(block ** 2)) > threshold
                if loud or speech:
                    speech.append(block)
                    silent = 0 if loud else silent + 1
                if speech and (silent > END_SILENCE or len(speech) > MAX_SPEECH):
                    if len(speech) - silent > MIN_SPEECH:
                        self._transcribe(np.concatenate(speech))
                    speech, silent = [], 0

    def _transcribe(self, audio):
        segments, _ = self.whisper.transcribe(
            audio, task="translate", beam_size=1, vad_filter=True, initial_prompt=self.PROMPT,
            condition_on_previous_text=False)
        text = " ".join(s.text for s in segments if s.no_speech_prob < 0.5).strip()
        if text and not any(h in text.lower() for h in HALLUCINATIONS):
            self.on_text(text)
