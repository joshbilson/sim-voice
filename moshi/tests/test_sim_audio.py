# SPDX-License-Identifier: MIT
"""Voice samples (moshi.sim_audio): WAV decoding, 24 kHz resampling, the 4.0 s window. No torch."""

import io
import os
import struct
import tempfile
import unittest
import wave

import numpy as np

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
import voice_fixtures as F
from moshi import sim_audio as A

RATE = A.SAMPLE_RATE


def peak_frequency(x: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.fft.rfftfreq(x.size, 1.0 / rate)[int(np.argmax(spectrum))])


class DecodeTests(unittest.TestCase):
    def test_pcm16_matches_the_stdlib_wave_module(self):
        samples = F.tone(440, 0.5, 0.6)
        data = F.pcm_wav_bytes(samples)
        decoded, rate = A.decode_wav(data)
        with wave.open(io.BytesIO(data), "rb") as reference:
            expected = np.frombuffer(reference.readframes(reference.getnframes()), "<i2").astype(np.float32) / 32768.0
        self.assertEqual(rate, RATE)
        self.assertTrue(np.array_equal(decoded, expected))

    def test_integer_pcm_widths_decode_within_one_step(self):
        samples = F.tone(330, 0.2, 0.7)
        for width, step in ((1, 1 / 128), (2, 1 / 32768), (3, 1 / 8388608), (4, 1 / 2147483648)):
            with self.subTest(width=width):
                decoded, rate = A.decode_wav(F.pcm_wav_bytes(samples, width=width))
                self.assertEqual(rate, RATE)
                self.assertEqual(decoded.size, samples.size)
                self.assertLessEqual(float(np.max(np.abs(decoded - samples))), 2 * step + 1e-6)

    def test_float32_and_float64_decode_exactly(self):
        samples = F.tone(220, 0.3, 0.5)
        for width in (4, 8):
            with self.subTest(width=width):
                decoded, rate = A.decode_wav(F.float_wav_bytes(samples, width=width))
                self.assertEqual(rate, RATE)
                self.assertTrue(np.allclose(decoded, samples, atol=1e-6))

    def test_extensible_pcm_and_float_are_read_through_the_subformat(self):
        samples = F.tone(500, 0.2, 0.4)
        decoded, _ = A.decode_wav(F.extensible_wav_bytes(samples, float_format=True))
        self.assertTrue(np.allclose(decoded, samples, atol=1e-6))
        decoded, _ = A.decode_wav(F.extensible_wav_bytes(samples, float_format=False))
        self.assertTrue(np.allclose(decoded, samples, atol=2 / 32768))

    def test_multichannel_is_averaged_to_mono(self):
        left = F.tone(300, 0.1, 0.8)
        data = F.pcm_wav_bytes(left, channels=2)  # both channels carry the same tone
        decoded, _ = A.decode_wav(data)
        self.assertEqual(decoded.size, left.size)
        self.assertTrue(np.allclose(decoded, left, atol=2 / 32768))

    def test_truncated_and_streaming_data_sizes_read_what_is_there(self):
        samples = F.tone(200, 0.2, 0.5)
        cut = F.pcm_wav_bytes(samples)[:-100]  # the data chunk claims more bytes than the file holds
        decoded, _ = A.decode_wav(cut)
        self.assertGreater(decoded.size, samples.size - 60)
        body = np.asarray(samples, dtype="<f4").tobytes()
        fmt = struct.pack("<HHIIHH", 3, 1, RATE, RATE * 4, 4, 32)
        decoded, rate = A.decode_wav(F.riff(fmt, body, data_size=0xFFFFFFFF))  # streaming writers
        self.assertEqual((rate, decoded.size), (RATE, samples.size))

    def test_unusable_files_are_refused_with_audio_error(self):
        good = F.pcm_wav_bytes(F.tone(100, 0.1, 0.5))
        cases = {
            "not a wav": b"ID3" + good[3:],
            "too short": b"RIFF",
            "no data chunk": good[:12] + good[12:36],
            "mu-law": F.riff(struct.pack("<HHIIHH", 7, 1, RATE, RATE, 1, 8), b"\x00" * 100),
            "zero channels": F.riff(struct.pack("<HHIIHH", 1, 0, RATE, RATE * 2, 2, 16), b"\x00" * 100),
            "nine channels": F.riff(struct.pack("<HHIIHH", 1, 9, RATE, RATE * 18, 18, 16), b"\x00" * 180),
            "2 kHz rate": F.riff(struct.pack("<HHIIHH", 1, 1, 2000, 4000, 2, 16), b"\x00" * 100),
            "24 bit float": F.riff(struct.pack("<HHIIHH", 3, 1, RATE, RATE * 3, 3, 24), b"\x00" * 99),
            "no samples": F.riff(struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16), b""),
            "damaged fmt": F.riff(struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)[:10], b"\x00" * 10),
        }
        for label, data in cases.items():
            with self.subTest(case=label), self.assertRaises(A.AudioError) as ctx:
                A.decode_wav(data)
            self.assertTrue(str(ctx.exception).isprintable())


class ResampleTests(unittest.TestCase):
    def test_a_tone_keeps_its_frequency_and_level_from_common_rates(self):
        for rate in (8000, 16000, 22050, 44100, 48000, 96000):
            with self.subTest(rate=rate):
                x = F.tone(1000, 1.0, 1.0, rate=rate)
                y = A.resample(x, rate)
                self.assertEqual(y.size, RATE)
                middle = y[RATE // 4:3 * RATE // 4].astype(np.float64)
                self.assertAlmostEqual(peak_frequency(middle, RATE), 1000.0, delta=1.0)
                self.assertAlmostEqual(float(np.max(np.abs(middle))), 1.0, delta=0.01)

    def test_content_above_the_new_nyquist_is_removed_not_aliased(self):
        x = F.tone(15000, 1.0, 1.0, rate=44100)  # above 12 kHz: would alias to 9 kHz unfiltered
        y = A.resample(x, 44100)[2000:-2000]
        level_db = 20 * np.log10(float(np.max(np.abs(y))) + 1e-12)
        self.assertLess(level_db, -60.0)

    def test_speech_band_below_11_khz_passes(self):
        x = F.tone(11000, 1.0, 1.0, rate=44100)
        y = A.resample(x, 44100)[2000:-2000]
        self.assertGreater(float(np.max(np.abs(y))), 0.9)

    def test_identity_at_the_target_rate(self):
        x = F.tone(440, 0.1, 0.3)
        self.assertTrue(np.array_equal(A.resample(x, RATE), x))

    def test_output_length_follows_the_rate_ratio(self):
        for rate in (44100, 48000, 16000):
            with self.subTest(rate=rate):
                x = np.zeros(rate, dtype=np.float32)
                self.assertLessEqual(abs(A.resample(x, rate).size - RATE), 1)

    def test_dc_gain_is_unity(self):
        y = A.resample(np.full(44100, 0.5, dtype=np.float32), 44100)
        self.assertAlmostEqual(float(np.mean(y[500:-500])), 0.5, places=4)


class SpeechWindowTests(unittest.TestCase):
    def test_silence_and_faint_noise_have_no_speech(self):
        rng = np.random.default_rng(7)
        for label, x in (("silence", F.silence(5)), ("noise at -60 dB", (0.001 * rng.standard_normal(5 * RATE)).astype(np.float32))):
            with self.subTest(case=label), self.assertRaises(A.AudioError):
                A.prepare_voice_sample(F.pcm_wav_bytes(x))

    def test_less_than_two_seconds_of_speech_is_refused(self):
        x = F.concat(F.speech_like(1.5), F.silence(5))
        with self.assertRaises(A.AudioError) as ctx:
            A.prepare_voice_sample(F.pcm_wav_bytes(x))
        self.assertIn("2 s of speech", str(ctx.exception))

    def test_window_sits_inside_the_loudest_dense_speech(self):
        # 3 s silence, a quiet 1.5 s burst, 4 s silence, a loud 6 s burst, 2 s silence.
        x = F.concat(F.silence(3), F.speech_like(1.5, 0.05), F.silence(4), F.speech_like(6, 0.4), F.silence(2))
        sample = A.prepare_voice_sample(F.pcm_wav_bytes(x))
        loud_start, loud_end = 8.5, 14.5
        self.assertGreaterEqual(sample.start_seconds, loud_start - 0.15)
        self.assertLessEqual(sample.start_seconds + A.VOICE_SECONDS, loud_end + 0.15)
        self.assertGreater(sample.speech_seconds, 6.5)

    def test_window_edges_snap_to_a_quiet_point(self):
        # Silence to 1.0 s, a 60 ms gap at 1.03 s, then speech. The best window starts at the onset;
        # its start moves to the gap, the quietest point within 100 ms.
        x = F.concat(F.silence(1.0), F.speech_like(0.03, 0.3), F.silence(0.06), F.speech_like(6, 0.3))
        energy = A.frame_energies_db(x.astype(np.float64))
        start = A.window_start(x.size, energy, A.speech_frames(energy)) / RATE
        self.assertGreaterEqual(start, 1.02)
        self.assertLessEqual(start, 1.08)

    def test_speech_shorter_than_the_window_is_centred_on_it(self):
        x = F.concat(F.silence(3.5), F.speech_like(2.5, 0.3), F.silence(4))
        sample = A.prepare_voice_sample(F.pcm_wav_bytes(x))
        self.assertAlmostEqual(sample.start_seconds, 3.5 + 1.25 - 2.0, delta=0.12)
        self.assertGreaterEqual(sample.speech_seconds, 2.2)

    def test_output_is_a_normalised_four_second_window_with_faded_edges(self):
        sample = A.prepare_voice_sample(F.pcm_wav_bytes(F.speech_like(6, 0.05)))
        self.assertEqual(sample.pcm.dtype, np.dtype("<i2"))
        self.assertEqual(sample.pcm.size, A.VOICE_SAMPLES)
        x = sample.pcm.astype(np.float64) / 32767.0
        rms_dbfs = 20 * np.log10(float(np.sqrt(np.mean(x * x))))
        self.assertAlmostEqual(rms_dbfs, A.TARGET_RMS_DBFS, delta=0.5)
        self.assertLessEqual(float(np.max(np.abs(x))), 0.9)
        self.assertLess(abs(int(sample.pcm[0])), 2)
        self.assertLess(abs(int(sample.pcm[-1])), 2)

    def test_the_same_upload_gives_the_same_window(self):
        data = F.pcm_wav_bytes(F.concat(F.silence(1), F.speech_like(5, 0.3)))
        first = A.prepare_voice_sample(data)
        second = A.prepare_voice_sample(data)
        self.assertTrue(np.array_equal(first.pcm, second.pcm))
        self.assertEqual(first.start_seconds, second.start_seconds)

    def test_any_supported_format_becomes_the_same_kind_of_sample(self):
        def scene(rate):
            return F.concat(F.silence(0.5, rate), F.speech_like(5, 0.3, rate))
        uploads = {
            "44.1 kHz stereo 16-bit": F.pcm_wav_bytes(scene(44100), rate=44100, channels=2),
            "48 kHz 24-bit mono": F.pcm_wav_bytes(scene(48000), rate=48000, width=3),
            "24 kHz float32": F.float_wav_bytes(scene(24000)),
            "16 kHz extensible float": F.extensible_wav_bytes(scene(16000), float_format=True, rate=16000),
        }
        found = []
        for label, data in uploads.items():
            with self.subTest(upload=label):
                sample = A.prepare_voice_sample(data)
                self.assertEqual(sample.pcm.size, A.VOICE_SAMPLES)
                found.append(sample.speech_seconds)
        # The same scene in every encoding finds the same speech (synthetic envelope dips included).
        self.assertLess(max(found) - min(found), 0.05)
        self.assertGreaterEqual(min(found), 4.0)

    def test_write_wav16_is_24k_mono_16_bit(self):
        sample = A.prepare_voice_sample(F.pcm_wav_bytes(F.speech_like(5, 0.3)))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "c-test.wav")
            A.write_wav16(path, sample.pcm)
            with wave.open(path, "rb") as back:
                self.assertEqual((back.getnchannels(), back.getsampwidth(), back.getframerate()), (1, 2, RATE))
                self.assertEqual(back.getnframes(), A.VOICE_SAMPLES)
                self.assertEqual(back.readframes(back.getnframes()), sample.pcm.tobytes())


if __name__ == "__main__":
    unittest.main()
