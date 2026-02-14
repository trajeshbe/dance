"""
YouTube Video Download Service

Downloads videos from YouTube using yt-dlp
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeService:
    """Service for downloading YouTube videos"""

    def __init__(self, download_dir: str = "/tmp/youtube_downloads"):
        """
        Initialize YouTube service

        Args:
            download_dir: Directory to save downloaded videos
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def download_video(
        self,
        url: str,
        video_id: Optional[str] = None,
        max_duration: int = 300,  # 5 minutes
    ) -> Dict[str, Any]:
        """
        Download video from YouTube

        Args:
            url: YouTube video URL
            video_id: Optional custom video ID for filename
            max_duration: Maximum video duration in seconds

        Returns:
            Dict with video_path, duration, fps, resolution
        """
        try:
            # Generate filename
            if video_id is None:
                import uuid
                video_id = str(uuid.uuid4())

            output_path = os.path.join(self.download_dir, f"{video_id}.mp4")

            # yt-dlp options
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_path,
                'quiet': False,
                'no_warnings': False,
                'match_filter': yt_dlp.utils.match_filter_func(f"duration < {max_duration}"),
            }

            logger.info(f"📥 Downloading video from: {url}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)

                if info is None:
                    raise ValueError("Could not extract video information")

                duration = info.get('duration', 0)
                fps = info.get('fps', 30)
                width = info.get('width', 1920)
                height = info.get('height', 1080)
                title = info.get('title', 'Unknown')

                logger.info(f"Video: {title}")
                logger.info(f"Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")

                # Check duration
                if duration > max_duration:
                    raise ValueError(f"Video too long: {duration}s (max {max_duration}s)")

                # Download the video
                ydl.download([url])

            logger.info(f"✅ Video downloaded: {output_path}")

            return {
                "video_path": output_path,
                "duration": duration,
                "fps": fps,
                "width": width,
                "height": height,
                "title": title,
            }

        except Exception as e:
            logger.error(f"❌ Failed to download video: {e}")
            raise

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading

        Args:
            url: YouTube video URL

        Returns:
            Dict with video metadata
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if info is None:
                    raise ValueError("Could not extract video information")

                return {
                    "title": info.get('title', 'Unknown'),
                    "duration": info.get('duration', 0),
                    "fps": info.get('fps', 30),
                    "width": info.get('width', 1920),
                    "height": info.get('height', 1080),
                    "thumbnail": info.get('thumbnail'),
                    "uploader": info.get('uploader'),
                }

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise


# Singleton instance
_youtube_service: Optional[YouTubeService] = None


def get_youtube_service() -> YouTubeService:
    """Get singleton YouTube service instance"""
    global _youtube_service
    if _youtube_service is None:
        _youtube_service = YouTubeService()
    return _youtube_service
