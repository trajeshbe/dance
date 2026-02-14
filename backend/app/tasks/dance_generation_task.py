"""
Celery task for dance video generation

Supports:
- Local GPU processing
- Google Colab remote GPU processing (if configured)
"""
from celery import Task
from app.celery_app import celery_app
from app.services.colab_service import colab_manager
from app.core.config import settings
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)


class DanceGenerationTask(Task):
    """Base task with error handling"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        # TODO: Update database with failure status

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {task_id} completed successfully")
        # TODO: Update database with success status


@celery_app.task(
    bind=True,
    base=DanceGenerationTask,
    name="app.tasks.generate_dance_video"
)
def generate_dance_video_task(
    self,
    project_id: str,
    config: Dict
):
    """
    Generate dance video with facial expressions

    Args:
        project_id: Project UUID
        config: Generation configuration

    Returns:
        Final video URL
    """
    logger.info(f"Starting generation for project {project_id}")

    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'Initializing',
                'progress': 0,
                'logs': ['Starting video generation...']
            }
        )

        # ================================================================
        # CHECK IF SHOULD USE COLAB
        # ================================================================
        import asyncio

        # Check if Colab is available and preferred
        should_use_colab = asyncio.run(colab_manager.should_use_colab())

        if should_use_colab:
            logger.info("Using Google Colab for GPU processing")

            self.update_state(
                state='PROGRESS',
                meta={
                    'step': 'Routing to Colab GPU',
                    'progress': 5,
                    'logs': ['Using Google Colab GPU for processing...']
                }
            )

            # Submit to Colab and wait for result
            try:
                final_video_url = asyncio.run(
                    colab_manager.submit_to_colab(project_id, config)
                )

                self.update_state(
                    state='SUCCESS',
                    meta={
                        'step': 'Complete (Colab)',
                        'progress': 100,
                        'logs': ['Video generation complete on Colab!'],
                        'final_video_url': final_video_url
                    }
                )

                return {
                    'status': 'completed',
                    'final_video_url': final_video_url,
                    'processed_on': 'colab'
                }

            except Exception as e:
                logger.warning(f"Colab processing failed: {e}, falling back to local")
                # Fall through to local processing

        # ================================================================
        # LOCAL PROCESSING (if Colab not available or failed)
        # ================================================================
        logger.info("Using local GPU/CPU for processing")

        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'Processing locally',
                'progress': 5,
                'logs': ['Using local GPU/CPU...']
            }
        )

        # ================================================================
        # USE REAL PIPELINE
        # ================================================================
        from app.services.dance_generation_pipeline import get_dance_generation_pipeline
        import os

        pipeline = get_dance_generation_pipeline()

        # Set up output path
        output_path = f"/tmp/dance_generation/{project_id}_final.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Progress callback
        def progress_callback(step: str, progress: int):
            self.update_state(
                state='PROGRESS',
                meta={
                    'step': step,
                    'progress': progress,
                    'logs': [step]
                }
            )

        # Run pipeline
        result = asyncio.run(
            pipeline.generate_dance_video(
                photo_url=config['photo_url'],
                reference_video_url=config['reference_video_url'],
                output_path=output_path,
                scene_prompt=config.get('scene_prompt'),
                style_prompt=config.get('style_prompt', 'cinematic, 4k, dramatic lighting'),
                background_mode=config.get('background_mode', 'original'),
                enable_facial_expressions=config.get('enable_facial_expressions', True),
                enable_lip_sync=config.get('enable_lip_sync', True),
                expression_intensity=config.get('expression_intensity', 1.0),
                progress_callback=progress_callback,
            )
        )

        # Check result
        if result['success']:
            final_video_url = result['video_url']

            self.update_state(
                state='SUCCESS',
                meta={
                    'step': 'Complete',
                    'progress': 100,
                    'logs': ['Video generation complete!'],
                    'final_video_url': final_video_url
                }
            )

            logger.info(f"Generation complete for project {project_id}")

            return {
                'status': 'completed',
                'final_video_url': final_video_url,
                'metadata': result.get('metadata', {})
            }
        else:
            raise Exception(result.get('error', 'Unknown error'))

    except Exception as e:
        logger.error(f"Error generating video: {e}", exc_info=True)

        self.update_state(
            state='FAILURE',
            meta={
                'step': 'Failed',
                'progress': 0,
                'logs': [f'Error: {str(e)}'],
                'error': str(e)
            }
        )

        raise
