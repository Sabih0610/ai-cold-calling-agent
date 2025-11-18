from faster_whisper import WhisperModel
model = WhisperModel('small', device='cpu', compute_type='int8')
segments, info = model.transcribe('data/recordings/cid4020_lid9300000_1762906691.wav', vad_filter=False, beam_size=1, without_timestamps=True)
for seg in segments:
    print(seg.text.strip())
