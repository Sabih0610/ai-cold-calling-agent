import webrtcvad
import wave
import os

# Path to your WAV file
audio_path = r"F:\ai cold calling agent\piper_output_16k.wav"

# Make sure file exists
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"File not found: {audio_path}")

# Open the WAV file
with wave.open(audio_path, "rb") as wf:
    sample_rate = wf.getframerate()
    num_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {num_channels}")
    print(f"Sample width: {sample_width} bytes")

    # WebRTC VAD requires 16 kHz, mono, 16-bit PCM audio
    if sample_rate != 16000 or num_channels != 1 or sample_width != 2:
        print("⚠️ Audio must be 16-bit mono 16kHz PCM for VAD to work correctly.")
    else:
        vad = webrtcvad.Vad(2)  # aggressiveness: 0–3
        frames = []
        # Each frame = 20ms (320 samples at 16kHz)
        frame_duration = 20  # ms
        frame_size = int(sample_rate * frame_duration / 1000) * sample_width
        wf.rewind()

        # Read and test frames
        speech_frames = 0
        total_frames = 0
        while True:
            frame = wf.readframes(frame_size // sample_width)
            if len(frame) < frame_size:
                break
            total_frames += 1
            if vad.is_speech(frame, sample_rate):
                speech_frames += 1

        print(f"✅ Speech frames detected: {speech_frames} / {total_frames}")
        speech_ratio = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
        print(f"🧠 Estimated speech ratio: {speech_ratio:.1f}%")
