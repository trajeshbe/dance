"""
Audio Extraction and Synchronization Service

Handles:
- Audio extraction from videos
- Beat detection
- Lip-sync alignment
- Audio-video synchronization
"""
import os
import logging
from typing import Dict, Any, Optional, List
import subprocess
import librosa
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


class AudioService:
    """Service for audio extraction and synchronization"""

    def __init__(self):
        """Initialize audio service"""
        logger.info("🎵 Initializing Audio Service")

    def extract_audio_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract audio from video file

        Args:
            video_path: Path to video file
            output_path: Optional output path for audio

        Returns:
            Dict with audio_path, duration, sample_rate
        """
        try:
            if output_path is None:
                output_path = video_path.replace('.mp4', '_audio.wav')

            logger.info(f"🎵 Extracting audio from: {video_path}")

            # Use ffmpeg to extract audio
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # WAV codec
                '-ar', '44100',  # 44.1kHz sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite
                output_path
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")

            # Load audio to get metadata
            audio, sr = librosa.load(output_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)

            logger.info(f"✅ Audio extracted: {duration:.2f}s at {sr}Hz")

            return {
                "success": True,
                "audio_path": output_path,
                "duration": duration,
                "sample_rate": sr,
                "channels": 2,
            }

        except Exception as e:
            logger.error(f"❌ Audio extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def detect_beats(
        self,
        audio_path: str,
    ) -> Dict[str, Any]:
        """
        Detect beats in audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with beat times and tempo
        """
        try:
            logger.info(f"🥁 Detecting beats in: {audio_path}")

            # Load audio
            y, sr = librosa.load(audio_path)

            # Detect tempo and beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            logger.info(f"✅ Detected {len(beat_times)} beats at {tempo:.1f} BPM")

            return {
                "success": True,
                "tempo": float(tempo),
                "beat_times": beat_times.tolist(),
                "num_beats": len(beat_times),
            }

        except Exception as e:
            logger.error(f"❌ Beat detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def add_audio_to_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        sync_offset: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Add audio track to video

        Args:
            video_path: Path to input video (silent)
            audio_path: Path to audio file
            output_path: Path for output video
            sync_offset: Time offset for synchronization (seconds)

        Returns:
            Dict with success and output_path
        """
        try:
            logger.info(f"🎬 Adding audio to video")
            logger.info(f"   Video: {video_path}")
            logger.info(f"   Audio: {audio_path}")
            logger.info(f"   Sync offset: {sync_offset}s")

            # Build ffmpeg command
            command = [
                'ffmpeg',
                '-i', video_path,  # Input video
                '-i', audio_path,  # Input audio
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # AAC audio codec
                '-strict', 'experimental',
                '-shortest',  # Match shortest stream
                '-y',  # Overwrite
                output_path
            ]

            # Add sync offset if needed
            if sync_offset != 0:
                command.insert(2, '-itsoffset')
                command.insert(3, str(sync_offset))

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")

            logger.info(f"✅ Audio added to video: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
            }

        except Exception as e:
            logger.error(f"❌ Adding audio failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def sync_to_beats(
        self,
        beat_times: List[float],
        video_fps: int,
        num_frames: int,
    ) -> List[int]:
        """
        Generate frame indices synchronized to beats

        Args:
            beat_times: List of beat timestamps
            video_fps: Video frame rate
            num_frames: Total number of frames

        Returns:
            List of frame indices to emphasize on beats
        """
        beat_frames = []

        for beat_time in beat_times:
            frame_idx = int(beat_time * video_fps)
            if 0 <= frame_idx < num_frames:
                beat_frames.append(frame_idx)

        logger.info(f"Synchronized {len(beat_frames)} beats to frames")
        return beat_frames

    def create_lip_sync_markers(
        self,
        audio_path: str,
        video_fps: int = 30,
    ) -> Dict[str, Any]:
        """
        Create markers for lip-sync animation

        Args:
            audio_path: Path to audio file
            video_fps: Video frame rate

        Returns:
            Dict with mouth opening values per frame
        """
        try:
            logger.info("👄 Creating lip-sync markers")

            # Load audio
            y, sr = librosa.load(audio_path)

            # Extract vocal-range energy
            # Focus on 80-300Hz (typical vocal range)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)

            # Get energy in vocal range
            vocal_mask = (freqs >= 80) & (freqs <= 300)
            vocal_energy = np.mean(S[vocal_mask, :], axis=0)

            # Normalize
            vocal_energy = vocal_energy / (np.max(vocal_energy) + 1e-8)

            # Convert to frame-based
            hop_length = sr // video_fps
            frame_energy = librosa.resample(
                vocal_energy,
                orig_sr=len(vocal_energy) / librosa.get_duration(y=y, sr=sr),
                target_sr=video_fps
            )

            # Convert to mouth opening values (0-1)
            mouth_openings = np.clip(frame_energy * 2, 0, 1).tolist()

            logger.info(f"✅ Created {len(mouth_openings)} lip-sync markers")

            return {
                "success": True,
                "mouth_openings": mouth_openings,
                "num_frames": len(mouth_openings),
            }

        except Exception as e:
            logger.error(f"❌ Lip-sync marker creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Singleton instance
_audio_service: Optional[AudioService] = None


def get_audio_service() -> AudioService:
    """Get singleton audio service instance"""
    global _audio_service
    if _audio_service is None:
        _audio_service = AudioService()
    return _audio_service
