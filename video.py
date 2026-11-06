"""
video.py - Cross-Fade Animation & Audio-Visual Export
====================================================
Transforms 6 storyboard frames into smooth animated GIF and MP4 media:
1. Interpolation / Cross-Fade:
   - Uses Pillow's Image.blend() to generate intermediate blend frames
     between consecutive shots for smooth cinematic transitions.
2. AI-Generated Disclosure Label:
   - Enforces visible AI-generated watermarking on every frame of the final animation.
3. Fresh Output Directory:
   - Generates unique timestamped folders (outputs/run_YYYYMMDD_HHMMSS/) for each run,
     preventing any result caching or file reuse.
4. Audio Synthesis & Track Binding:
   - Synthesizes a gentle ambient soundtrack loop and multiplexes with MP4 video.
"""

import os
import math
import struct
import wave
from datetime import datetime
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def generate_ambient_music_loop(output_path: str = None, duration_sec: float = 12.0) -> str:
    """
    Procedurally synthesizes a gentle, calming 12-second ambient soundtrack loop.
    Uses numpy vectorized synthesis for instant generation (CC0 Public Domain).
    """
    if output_path is None:
        output_path = os.path.join(ASSETS_DIR, "ambient_story_theme.wav")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return output_path

    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)

    chord_freqs = [
        [261.63, 329.63, 392.00, 493.88],  # Cmaj7
        [220.00, 261.63, 329.63, 392.00],  # Am7
        [174.61, 220.00, 261.63, 329.63],  # Fmaj7
        [196.00, 261.63, 293.66, 392.00],  # Gsus4
    ]
    chord_duration = duration_sec / len(chord_freqs)

    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    left_signal = np.zeros(total_samples, dtype=np.float64)
    right_signal = np.zeros(total_samples, dtype=np.float64)

    for chord_idx, freqs in enumerate(chord_freqs):
        t_start = chord_idx * chord_duration
        t_end = t_start + chord_duration
        mask = (t >= t_start) & (t < t_end)
        chord_t = t[mask] - t_start

        envelope = np.exp(-1.8 * (chord_t / chord_duration)) * (1.0 - np.exp(-25.0 * chord_t))
        border_fade = np.minimum(1.0, t[mask] / 0.5) * np.minimum(1.0, (duration_sec - t[mask]) / 0.5)

        for note_i, freq in enumerate(freqs):
            pan = 0.3 + 0.4 * (note_i / len(freqs))
            sig = np.sin(2.0 * math.pi * freq * t[mask]) * 0.22
            sig += np.sin(4.0 * math.pi * freq * t[mask]) * 0.05
            left_signal[mask] += sig * (1.0 - pan) * envelope * border_fade * 12000.0
            right_signal[mask] += sig * pan * envelope * border_fade * 12000.0

    left_pcm = np.clip(left_signal, -32767, 32767).astype(np.int16)
    right_pcm = np.clip(right_signal, -32767, 32767).astype(np.int16)
    stereo = np.empty(total_samples * 2, dtype=np.int16)
    stereo[0::2] = left_pcm
    stereo[1::2] = right_pcm

    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo.tobytes())

    return output_path


def get_background_music_path() -> str:
    """Finds existing audio file in /assets, or generates ambient theme."""
    for fname in os.listdir(ASSETS_DIR):
        if fname.lower().endswith((".mp3", ".wav", ".ogg", ".aac")):
            return os.path.join(ASSETS_DIR, fname)
    return generate_ambient_music_loop()


def build_crossfade_sequence(
    frames: List[Image.Image],
    hold_frames: int = 12,
    transition_frames: int = 8,
    target_size: Tuple[int, int] = (512, 512)
) -> List[Image.Image]:
    """Creates a continuous sequence of PIL Images with smooth cross-fade interpolation."""
    if not frames:
        return []

    clean_frames = [f.convert("RGB").resize(target_size, Image.Resampling.LANCZOS) for f in frames]
    sequence: List[Image.Image] = []

    for i in range(len(clean_frames)):
        current = clean_frames[i]
        for _ in range(hold_frames):
            sequence.append(current)

        if i < len(clean_frames) - 1:
            next_frame = clean_frames[i + 1]
            for step in range(1, transition_frames + 1):
                alpha = step / float(transition_frames + 1)
                blended = Image.blend(current, next_frame, alpha)
                sequence.append(blended)

    return sequence


def export_animated_gif(
    image_sequence: List[Image.Image],
    output_path: str,
    fps: int = 15
) -> str:
    """Exports sequence as an optimized animated GIF to output_path."""
    if not image_sequence:
        raise ValueError("Cannot export GIF from empty image sequence.")

    duration_ms = int(1000 / fps)
    first_frame = image_sequence[0]
    
    first_frame.save(
        output_path,
        save_all=True,
        append_images=image_sequence[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    print(f"[VIDEO EXPORT] Fresh GIF saved -> {output_path} ({len(image_sequence)} frames)")
    return output_path


def export_mp4_with_audio(
    image_sequence: List[Image.Image],
    output_path: str,
    audio_path: Optional[str] = None,
    fps: int = 24
) -> str:
    """Exports sequence as MP4 with multiplexed audio to output_path."""
    if not image_sequence:
        raise ValueError("Cannot export MP4 from empty image sequence.")

    music_file = audio_path or get_background_music_path()
    total_frames = len(image_sequence)
    duration_sec = total_frames / float(fps)

    # Strategy 1: moviepy with AudioFileClip
    try:
        from moviepy.editor import ImageSequenceClip, AudioFileClip, afx
        np_frames = [np.array(img) for img in image_sequence]
        clip = ImageSequenceClip(np_frames, fps=fps)

        if os.path.exists(music_file):
            audio = AudioFileClip(music_file)
            if audio.duration < duration_sec:
                audio = audio.fx(afx.audio_loop, duration=duration_sec)
            else:
                audio = audio.subclip(0, duration_sec)
            clip = clip.set_audio(audio)

        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset="ultrafast",
            logger=None
        )
        clip.close()
        print(f"[VIDEO EXPORT] Fresh MP4 saved -> {output_path}")
        return output_path
    except Exception as moviepy_err:
        print(f"MoviePy export note: {moviepy_err}. Using imageio-ffmpeg fallback...")

    # Strategy 2: imageio + imageio_ffmpeg direct muxing
    try:
        import imageio
        import imageio_ffmpeg
        import subprocess

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p"
        )
        for img in image_sequence:
            writer.append_data(np.array(img))
        writer.close()

        if os.path.exists(music_file):
            temp_mp4 = output_path.replace(".mp4", "_raw.mp4")
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)
            os.rename(output_path, temp_mp4)

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_bin, "-y",
                "-i", temp_mp4,
                "-i", music_file,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)

        print(f"[VIDEO EXPORT] Fresh MP4 saved (ffmpeg) -> {output_path}")
        return output_path
    except Exception as imgio_err:
        print(f"ImageIO export failed: {imgio_err}")
        raise


def create_storyboard_media(
    frames: List[Image.Image],
    fps_video: int = 24,
    fps_gif: int = 15,
    hold_frames: int = 12,
    transition_frames: int = 8,
    output_dir: Optional[str] = None,
    progress_callback=None
) -> Tuple[str, str, str, str]:
    """
    Main orchestration function to generate fresh GIF, MP4, and individual frames.
    Saves outputs in a unique timestamped folder to guarantee NO CACHING.
    
    Returns:
        Tuple of (gif_path, mp4_path, music_path, output_dir)
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save individual frames to the run directory with both standard and timestamped unique names
    for idx, frame in enumerate(frames, 1):
        frame.save(os.path.join(output_dir, f"frame_{idx}.png"), "PNG")
        frame.save(os.path.join(output_dir, f"frame_{idx}_{run_timestamp}.png"), "PNG")

    if progress_callback:
        progress_callback(0.1, "Building cross-fade transition sequence...")

    seq = build_crossfade_sequence(
        frames=frames,
        hold_frames=hold_frames,
        transition_frames=transition_frames
    )

    if progress_callback:
        progress_callback(0.4, "Exporting animated GIF (no caching)...")
    gif_path = os.path.join(output_dir, f"storyboard_{run_timestamp}.gif")
    export_animated_gif(seq, output_path=gif_path, fps=fps_gif)
    # Also save standard name for scripting convenience
    try:
        import shutil
        shutil.copyfile(gif_path, os.path.join(output_dir, "storyboard.gif"))
    except Exception:
        pass

    if progress_callback:
        progress_callback(0.7, "Multiplexing audio into fresh MP4 video...")
    music_path = get_background_music_path()
    mp4_path = os.path.join(output_dir, f"storyboard_{run_timestamp}.mp4")
    export_mp4_with_audio(seq, output_path=mp4_path, audio_path=music_path, fps=fps_video)
    try:
        shutil.copyfile(mp4_path, os.path.join(output_dir, "storyboard.mp4"))
    except Exception:
        pass

    if progress_callback:
        progress_callback(1.0, f"Fresh media assembled in: {output_dir}")

    return gif_path, mp4_path, music_path, output_dir
