"""
MinIO client service for object storage
"""
from minio import Minio
from minio.error import S3Error
from pathlib import Path
from typing import Optional
import logging
from datetime import timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """MinIO client for file storage"""

    def __init__(self):
        """Initialize MinIO client"""
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )

        # Ensure buckets exist
        self._ensure_buckets()

    def _ensure_buckets(self):
        """Create buckets if they don't exist"""
        buckets = [
            settings.MINIO_BUCKET_PHOTOS,
            settings.MINIO_BUCKET_VIDEOS,
            settings.MINIO_BUCKET_REFERENCES,
            settings.MINIO_BUCKET_AUDIO,
            settings.MINIO_BUCKET_POSES,
        ]

        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Error creating bucket {bucket}: {e}")

    def upload_file(
        self,
        file_path: str,
        bucket_name: str,
        object_name: Optional[str] = None
    ) -> str:
        """
        Upload file to MinIO

        Args:
            file_path: Local file path
            bucket_name: Destination bucket
            object_name: Object name in bucket (default: filename)

        Returns:
            MinIO object URL
        """
        if object_name is None:
            object_name = Path(file_path).name

        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path
            )
            logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}")

            # Return URL
            return f"minio://{bucket_name}/{object_name}"

        except S3Error as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str
    ):
        """
        Download file from MinIO

        Args:
            bucket_name: Source bucket
            object_name: Object name in bucket
            file_path: Local destination path
        """
        try:
            self.client.fget_object(
                bucket_name,
                object_name,
                file_path
            )
            logger.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")

        except S3Error as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1)
    ) -> str:
        """
        Get presigned URL for object

        Args:
            bucket_name: Bucket name
            object_name: Object name
            expires: Expiration time

        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=expires
            )
            return url

        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    def parse_minio_url(self, minio_url: str) -> tuple[str, str]:
        """
        Parse MinIO URL to bucket and object name

        Args:
            minio_url: URL in format "minio://bucket/object"

        Returns:
            (bucket_name, object_name)
        """
        if not minio_url.startswith("minio://"):
            raise ValueError(f"Invalid MinIO URL: {minio_url}")

        parts = minio_url.replace("minio://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MinIO URL format: {minio_url}")

        return parts[0], parts[1]

    def upload_bytes(
        self,
        data: bytes,
        bucket_name: str,
        object_name: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload bytes data to MinIO

        Args:
            data: Bytes data
            bucket_name: Destination bucket
            object_name: Object name
            content_type: MIME type

        Returns:
            MinIO object URL
        """
        from io import BytesIO

        try:
            self.client.put_object(
                bucket_name,
                object_name,
                BytesIO(data),
                length=len(data),
                content_type=content_type
            )
            logger.info(f"Uploaded bytes to {bucket_name}/{object_name}")

            return f"minio://{bucket_name}/{object_name}"

        except S3Error as e:
            logger.error(f"Error uploading bytes: {e}")
            raise


# Global instance
_minio_client: Optional[MinIOClient] = None


def get_minio_client() -> MinIOClient:
    """Get singleton MinIO client instance"""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinIOClient()
    return _minio_client
