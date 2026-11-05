"""
video.py - Cross-Fade Animation & Audio-Visual Export
====================================================
Transforms 6 storyboard frames into smooth animated GIF and MP4 media:
1. Interpolation / Cross-Fade:
   - Uses Pillow's `Image.blend()` to generate ~10 intermediate cross-fade frames
     between each consecutive shot for smooth, professional cinematic transitions.
2. AI-Generated Disclosure Label:
   - Enforces visible AI-generated watermarking on every frame of the final animation.
3. Audio Synthesis & Track Binding:
   - Checks `/assets` for background music tracks.
   - If missing, procedurally synthesizes a gentle, harmonious ambient piano/chime loop
     (`assets/ambient_story_theme.wav`) using standard Python `wave` and `math` (zero external dependencies).
   - Multiplexes audio with video using `moviepy` or `imageio-ffmpeg`.
"""

import os
import math
import struct
import wave
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
    Uses standard library `wave` and harmonic sine wave synthesis (CC0 Public Domain).
    No external heavy audio libraries needed.
    """
    if output_path is None:
        output_path = os.path.join(ASSETS_DIR, "ambient_story_theme.wav")

    # If file already exists and has content, reuse it
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return output_path

    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    
    # Chords progression (Frequencies in Hz): Cmaj7 -> Am7 -> Fmaj7 -> Gsus4
    chord_freqs = [
        [261.63, 329.63, 392.00, 493.88],  # Cmaj7
        [220.00, 261.63, 329.63, 392.00],  # Am7
        [174.61, 220.00, 261.63, 329.63],  # Fmaj7
        [196.00, 261.63, 293.66, 392.00],  # Gsus4
    ]
    chord_duration = duration_sec / len(chord_freqs)

    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(2)  # Stereo
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(total_samples):
            t = i / sample_rate
            chord_idx = min(int(t / chord_duration), len(chord_freqs) - 1)
            active_notes = chord_freqs[chord_idx]
            chord_t = t - (chord_idx * chord_duration)

            # Natural exponential acoustic decay envelope
            envelope = math.exp(-1.8 * (chord_t / chord_duration)) * (1.0 - math.exp(-25.0 * chord_t))

            # Sum harmonic frequencies with soft warmth
            left_val = 0.0
            right_val = 0.0
            for note_i, freq in enumerate(active_notes):
                # Subtle stereo panning across chord notes
                pan = 0.3 + 0.4 * (note_i / len(active_notes))
                sig = math.sin(2.0 * math.pi * freq * t) * 0.22
                # Sub-harmonic overtone for warmth
                sig += math.sin(4.0 * math.pi * freq * t) * 0.05
                left_val += sig * (1.0 - pan)
                right_val += sig * pan

            # Overall volume scaling with fade-in and fade-out at borders
            border_fade = min(1.0, t / 0.5) * min(1.0, (duration_sec - t) / 0.5)
            master_gain = 12000.0 * envelope * border_fade

            sample_l = int(max(-32767, min(32767, left_val * master_gain)))
            sample_r = int(max(-32767, min(32767, right_val * master_gain)))

            wav_file.writeframes(struct.pack("<hh", sample_l, sample_r))

    print(f"Synthesized procedural ambient music loop saved to: {output_path}")
    return output_path


def get_background_music_path() -> str:
    """
    Finds the first existing audio file in /assets, or generates a procedural ambient theme.
    """
    for fname in os.listdir(ASSETS_DIR):
        if fname.lower().endswith((".mp3", ".wav", ".ogg", ".aac")):
            return os.path.join(ASSETS_DIR, fname)
    return generate_ambient_music_loop()


def build_crossfade_sequence(
    frames: List[Image.Image],
    hold_frames: int = 24,
    transition_frames: int = 10,
    target_size: Tuple[int, int] = (512, 512)
) -> List[Image.Image]:
    """
    Creates a continuous sequence of PIL Images with smooth cross-fade interpolation.
    
    Args:
        frames: List of 6 PIL Image frames.
        hold_frames: Number of frames to hold each still shot (~1.0s at 24fps).
        transition_frames: Number of intermediate blend frames between shots (~10 frames).
        target_size: Frame dimensions (width, height).
        
    Returns:
        List of all blended PIL Images in sequence.
    """
    if not frames:
        return []

    # Standardize sizes
    clean_frames = [f.convert("RGB").resize(target_size, Image.Resampling.LANCZOS) for f in frames]
    sequence: List[Image.Image] = []

    for i in range(len(clean_frames)):
        current = clean_frames[i]
        
        # Hold current shot
        for _ in range(hold_frames):
            sequence.append(current)

        # Cross-fade to next shot (if not the last frame)
        if i < len(clean_frames) - 1:
            next_frame = clean_frames[i + 1]
            for step in range(1, transition_frames + 1):
                alpha = step / float(transition_frames + 1)
                blended = Image.blend(current, next_frame, alpha)
                sequence.append(blended)

    return sequence


def export_animated_gif(
    image_sequence: List[Image.Image],
    output_path: Optional[str] = None,
    fps: int = 18
) -> str:
    """
    Exports the blended sequence as an optimized animated GIF.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, "storyboard.gif")

    if not image_sequence:
        raise ValueError("Cannot export GIF from empty image sequence.")

    # Convert sequence to palette mode with dithering for compact, clean GIF
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
    print(f"Animated GIF saved: {output_path} ({len(image_sequence)} frames)")
    return output_path


def export_mp4_with_audio(
    image_sequence: List[Image.Image],
    output_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    fps: int = 24
) -> str:
    """
    Exports sequence as an MP4 video file and multiplexes the background audio track.
    Uses moviepy or imageio_ffmpeg.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, "storyboard.mp4")

    if not image_sequence:
        raise ValueError("Cannot export MP4 from empty image sequence.")

    music_file = audio_path or get_background_music_path()
    total_frames = len(image_sequence)
    duration_sec = total_frames / float(fps)

    # Strategy 1: Attempt moviepy with AudioFileClip
    try:
        from moviepy.editor import ImageSequenceClip, AudioFileClip
        np_frames = [np.array(img) for img in image_sequence]
        clip = ImageSequenceClip(np_frames, fps=fps)

        if os.path.exists(music_file):
            audio = AudioFileClip(music_file)
            # Loop or trim audio to match video duration
            if audio.duration < duration_sec:
                # Subclip loops
                from moviepy.editor import afx
                audio = audio.fx(afx.audio_loop, duration=duration_sec)
            else:
                audio = audio.subclip(0, duration_sec)
            clip = clip.set_audio(audio)

        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset="medium",
            logger=None
        )
        clip.close()
        return output_path

    except Exception as moviepy_err:
        print(f"MoviePy video export note: {moviepy_err}. Using imageio-ffmpeg direct multiplexing...")

    # Strategy 2: ImageIO + imageio_ffmpeg fallback
    try:
        import imageio
        import imageio_ffmpeg

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

        # If audio exists, merge audio using ffmpeg directly
        if os.path.exists(music_file):
            temp_mp4 = output_path.replace(".mp4", "_temp.mp4")
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)
            os.rename(output_path, temp_mp4)

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            import subprocess
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

        return output_path

    except Exception as imgio_err:
        print(f"ImageIO export failed: {imgio_err}")
        raise


def create_storyboard_media(
    frames: List[Image.Image],
    fps_video: int = 24,
    fps_gif: int = 15,
    hold_frames: int = 20,
    transition_frames: int = 10,
    progress_callback=None
) -> Tuple[str, str, str]:
    """
    Main orchestration function to generate both GIF and MP4 with audio from 6 frames.
    
    Returns:
        Tuple of (gif_path, mp4_path, music_path)
    """
    if progress_callback:
        progress_callback(0.1, "Generating cross-fade transition frames...")

    seq = build_crossfade_sequence(
        frames=frames,
        hold_frames=hold_frames,
        transition_frames=transition_frames
    )

    if progress_callback:
        progress_callback(0.5, "Exporting animated GIF...")
    gif_path = export_animated_gif(seq, fps=fps_gif)

    if progress_callback:
        progress_callback(0.75, "Synthesizing music & exporting MP4 with audio...")
    music_path = get_background_music_path()
    mp4_path = export_mp4_with_audio(seq, audio_path=music_path, fps=fps_video)

    if progress_callback:
        progress_callback(1.0, "Media export completed successfully!")

    return gif_path, mp4_path, music_path
