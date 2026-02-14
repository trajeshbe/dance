"""
Celery application for async video generation tasks
"""
from celery import Celery
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "dance_generator",
    broker=settings.CELERY_BROKER,
    backend=settings.CELERY_BACKEND,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time (for GPU)
    worker_max_tasks_per_child=1,  # Restart worker after each task (free GPU memory)
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.tasks"])


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task to test Celery"""
    print(f"Request: {self.request!r}")
    return "Celery is working!"
