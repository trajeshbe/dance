"""
Celery tasks for dance video generation
"""
from app.tasks.dance_generation_task import generate_dance_video_task

__all__ = ["generate_dance_video_task"]
