"""
Google Colab Integration Service

Allows using Google Colab notebooks as remote GPU workers for video generation.

Architecture:
1. Backend submits job to Colab via ngrok tunnel
2. Colab notebook processes video generation
3. Results uploaded to MinIO
4. Backend retrieves final video

Benefits:
- Free/paid GPU access (T4, V100, A100)
- No local GPU required
- Scalable processing
"""
import httpx
import asyncio
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ColabWorkerService:
    """Service for communicating with Colab worker notebooks"""

    def __init__(self, colab_url: Optional[str] = None):
        """
        Initialize Colab worker service

        Args:
            colab_url: Ngrok URL of running Colab notebook
                      Example: https://abc123.ngrok.io
        """
        self.colab_url = colab_url
        self.timeout = httpx.Timeout(timeout=3600.0)  # 1 hour for video generation

    async def check_colab_available(self) -> bool:
        """
        Check if Colab worker is online and available

        Returns:
            True if Colab is available
        """
        if not self.colab_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.colab_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Colab not available: {e}")
            return False

    async def submit_job(
        self,
        project_id: str,
        config: Dict
    ) -> str:
        """
        Submit video generation job to Colab worker

        Args:
            project_id: Project UUID
            config: Generation configuration

        Returns:
            Job ID from Colab
        """
        if not self.colab_url:
            raise ValueError("Colab URL not configured")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.colab_url}/generate",
                json={
                    "project_id": project_id,
                    "config": config
                }
            )
            response.raise_for_status()

            result = response.json()
            return result["job_id"]

    async def get_job_status(self, job_id: str) -> Dict:
        """
        Get job status from Colab worker

        Args:
            job_id: Job ID from Colab

        Returns:
            Status dict with progress, step, etc.
        """
        if not self.colab_url:
            raise ValueError("Colab URL not configured")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.colab_url}/status/{job_id}")
            response.raise_for_status()

            return response.json()

    async def get_gpu_info(self) -> Dict:
        """
        Get GPU information from Colab worker

        Returns:
            GPU info (name, memory, utilization)
        """
        if not self.colab_url:
            raise ValueError("Colab URL not configured")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.colab_url}/gpu_info")
            response.raise_for_status()

            return response.json()


class ColabJobManager:
    """Manages job routing between local GPU and Colab"""

    def __init__(
        self,
        colab_url: Optional[str] = None,
        prefer_colab: bool = True
    ):
        """
        Initialize job manager

        Args:
            colab_url: Colab worker URL
            prefer_colab: Prefer Colab over local GPU if available
        """
        self.colab_service = ColabWorkerService(colab_url)
        self.prefer_colab = prefer_colab

    async def should_use_colab(self) -> bool:
        """
        Decide whether to use Colab or local GPU

        Returns:
            True if should use Colab
        """
        # Check if Colab is available
        colab_available = await self.colab_service.check_colab_available()

        if not colab_available:
            logger.info("Colab not available, using local GPU")
            return False

        if self.prefer_colab:
            logger.info("Using Colab (preferred)")
            return True

        # TODO: Check local GPU availability/load
        # For now, always use Colab if available and preferred

        return True

    async def submit_to_colab(
        self,
        project_id: str,
        config: Dict
    ) -> str:
        """
        Submit job to Colab and monitor progress

        Args:
            project_id: Project UUID
            config: Generation configuration

        Returns:
            Final video URL
        """
        logger.info(f"Submitting job {project_id} to Colab")

        # Get GPU info
        gpu_info = await self.colab_service.get_gpu_info()
        logger.info(f"Colab GPU: {gpu_info.get('name')}, {gpu_info.get('memory_total')} GB")

        # Submit job
        job_id = await self.colab_service.submit_job(project_id, config)
        logger.info(f"Colab job started: {job_id}")

        # Monitor progress
        while True:
            await asyncio.sleep(5)  # Poll every 5 seconds

            status = await self.colab_service.get_job_status(job_id)

            logger.info(
                f"Colab job {job_id}: {status['step']} - {status['progress']}%"
            )

            if status['status'] == 'completed':
                logger.info(f"Colab job {job_id} completed!")
                return status['final_video_url']

            elif status['status'] == 'failed':
                error = status.get('error', 'Unknown error')
                logger.error(f"Colab job {job_id} failed: {error}")
                raise Exception(f"Colab job failed: {error}")


# Global instance (configured from environment)
colab_manager = ColabJobManager()
