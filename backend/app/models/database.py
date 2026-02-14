"""
Database models for dance video generation
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

Base = declarative_base()


class ProjectStatus(str, enum.Enum):
    """Project status enum"""
    PENDING = "pending"
    ANALYZING_REFERENCE = "analyzing_reference"
    PROCESSING_PHOTO = "processing_photo"
    GENERATING_MOTION = "generating_motion"
    GENERATING_EXPRESSIONS = "generating_expressions"
    COMPOSITING = "compositing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelProvider(str, enum.Enum):
    """AI model provider enum"""
    FOMM = "fomm"
    LIVEPORTRAIT = "liveportrait"
    MAGICANIMATE = "magicanimate"
    MAGICDANCE = "magicdance"
    MOORE_ANIMATEANYONE = "moore"
    SEEDANCE_API = "seedance_api"
    REPLICATE_API = "replicate_api"


class ChoreographyStrategy(str, enum.Enum):
    """Choreography mapping strategy"""
    ALL_SYNC = "all_sync"              # Everyone does same moves
    ONE_TO_N = "1_to_n"                # Divide reference into segments
    N_TO_N = "n_to_n"                  # Map dancers 1:1
    TIME_SEGMENTS = "time_segments"     # Custom mix


class DanceProject(Base):
    """Main dance video project"""
    __tablename__ = "dance_projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)  # TODO: Add User table

    # Input data
    photo_url = Column(Text, nullable=False)  # MinIO path to group/solo photo
    photo_type = Column(String(50), default="group")  # "solo" or "group"

    # Reference video
    reference_video_url = Column(Text)  # YouTube URL or MinIO path
    reference_video_source = Column(String(50))  # "youtube", "upload", "library"

    # Audio
    audio_url = Column(Text)  # Separate audio or extracted from reference
    audio_source = Column(String(50))  # "youtube", "upload", "reference_video"
    song_title = Column(String(255))

    # NEW: Text-to-Video Prompts (Sora/Kling style)
    scene_prompt = Column(Text)  # "dancing in a neon-lit nightclub"
    style_prompt = Column(Text)  # "cinematic, 4k, dramatic lighting"
    negative_prompt = Column(Text)  # "blurry, low quality, distorted"
    background_mode = Column(String(50), default="original")  # "original", "generated", "custom"
    background_url = Column(Text)  # Custom background image/video

    # Choreography configuration
    choreography_strategy = Column(String(50), default="all_sync")
    choreography_mapping = Column(JSONB)  # Detailed dancer-to-person mapping

    # Expression configuration
    enable_facial_expressions = Column(Boolean, default=True)
    enable_lip_sync = Column(Boolean, default=True)
    expression_intensity = Column(Float, default=1.0)  # 0.0 - 2.0

    # Model configuration
    body_motion_model = Column(String(50))  # Which model for body
    face_expression_model = Column(String(50))  # Which model for face

    # Processing status
    status = Column(String(50), default=ProjectStatus.PENDING.value)
    progress = Column(Integer, default=0)  # 0-100
    current_step = Column(String(100))

    # Output
    final_video_url = Column(Text)  # MinIO path to final video
    preview_url = Column(Text)  # Quick preview (low-res)
    thumbnail_url = Column(Text)

    # Metadata
    reference_metadata = Column(JSONB)  # Analyzed reference video data
    processing_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    persons = relationship("DancePerson", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("DanceJob", back_populates="project", cascade="all, delete-orphan")


class DancePerson(Base):
    """Detected person in group photo"""
    __tablename__ = "dance_persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("dance_projects.id", ondelete="CASCADE"), nullable=False)

    person_index = Column(Integer, nullable=False)  # 0, 1, 2, ...

    # Bounding box and segmentation
    bbox = Column(JSONB, nullable=False)  # {x, y, w, h}
    segmentation_mask_url = Column(Text)  # MinIO path to mask
    person_crop_url = Column(Text)  # Cropped person image

    # Face data
    face_data = Column(JSONB)  # {face_bbox, landmarks, alignment}

    # Assigned choreography
    dance_type = Column(String(50))  # "synchronized", "individual"
    reference_dancer_id = Column(String(50))  # "dancer_0", "dancer_1", etc.
    time_segments = Column(JSONB)  # [[start, end], ...]

    # Generated videos (intermediate)
    body_video_url = Column(Text)  # Body motion only
    face_video_url = Column(Text)  # Face expression only
    composite_video_url = Column(Text)  # Face + body combined

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("DanceProject", back_populates="persons")


class ReferenceVideo(Base):
    """Processed reference dance video"""
    __tablename__ = "reference_videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Source
    url = Column(Text, nullable=False)  # Original URL or MinIO path
    source = Column(String(50))  # "youtube", "upload", "library"
    title = Column(String(255))

    # Processed files
    processed_video_url = Column(Text)  # Cleaned/trimmed version
    audio_url = Column(Text)  # Extracted audio

    # Motion data
    pose_sequences_url = Column(Text)  # JSON file with pose keypoints
    face_sequences_url = Column(Text)  # JSON file with facial expressions

    # Metadata
    num_dancers = Column(Integer)
    duration_seconds = Column(Float)
    fps = Column(Integer)
    resolution = Column(JSONB)  # {width, height}

    # Analysis results
    has_audio = Column(Boolean, default=False)
    has_vocals = Column(Boolean, default=False)  # For lip-sync
    has_face_data = Column(Boolean, default=False)
    detected_emotions = Column(JSONB)  # Dominant emotions in video

    # Library fields
    is_public = Column(Boolean, default=False)  # Shareable template
    use_count = Column(Integer, default=0)
    tags = Column(JSONB)  # ["kpop", "energetic", "synchronized"]
    difficulty = Column(String(50))  # "easy", "medium", "hard"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DanceJob(Base):
    """Background job for video generation"""
    __tablename__ = "dance_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("dance_projects.id", ondelete="CASCADE"), nullable=False)

    celery_task_id = Column(String(255), unique=True)

    # Progress tracking
    step = Column(String(100))  # Current processing step
    progress = Column(Integer, default=0)  # 0-100
    logs = Column(JSONB, default=[])  # Array of log messages

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    project = relationship("DanceProject", back_populates="jobs")


class ChoreographyTemplate(Base):
    """Pre-defined choreography templates"""
    __tablename__ = "choreography_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name = Column(String(255), nullable=False)  # "BTS Dynamite Chorus"
    description = Column(Text)

    reference_video_id = Column(UUID(as_uuid=True), ForeignKey("reference_videos.id"))

    # Time segment for this template
    time_start = Column(Float)  # Start timestamp in reference video
    time_end = Column(Float)    # End timestamp

    # Metadata
    difficulty = Column(String(50))  # "easy", "medium", "hard"
    tags = Column(JSONB)  # ["kpop", "energetic"]

    # Recommended strategy
    recommended_strategy = Column(String(50))

    # Public/shareable
    is_public = Column(Boolean, default=True)
    use_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)


class StylePreset(Base):
    """Style presets for Sora/Kling-like generation"""
    __tablename__ = "style_presets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name = Column(String(255), nullable=False)  # "Cyberpunk Nightclub"
    description = Column(Text)

    # Prompt templates
    scene_prompt_template = Column(Text)  # "dancing in a {location} with {lighting}"
    style_prompt = Column(Text)  # "cinematic, 4k, neon lights, dramatic"
    negative_prompt = Column(Text)

    # Example image
    preview_url = Column(Text)

    # Settings
    recommended_background_mode = Column(String(50))  # "generated", "custom"

    # Metadata
    category = Column(String(100))  # "nightclub", "nature", "urban", "fantasy"
    is_public = Column(Boolean, default=True)
    use_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
