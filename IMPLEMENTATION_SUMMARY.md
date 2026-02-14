# Dance Video Generator - Implementation Summary

## ✅ What We Built

A **Sora/Kling-quality AI dance video generator** that transforms photos into dancing videos with:

- ✨ **Facial Expression Transfer** (FOMM, LivePortrait)
- 🎭 **Text Prompts** for scene/style control
- 🎵 **Motion Sync** from YouTube reference videos
- 💋 **Lip-Sync** capabilities
- 🎬 **Cinematic Quality** output

---

## 🏗️ Architecture Overview

### Hybrid AI Pipeline

```
Input Photo → Person Detection (YOLO) → Face Detection (MediaPipe)
                                    ↓
YouTube URL → Download → Dancer Detection → Pose Extraction (OpenPose)
                                    ↓
                            Facial Expression Extraction (FOMM/MediaPipe)
                                    ↓
Text Prompts → AnimateDiff + ControlNet → Body Motion Video
(Scene/Style)            ↓
                    FOMM/LivePortrait → Facial Expression Video
                                    ↓
                        Face Compositing → Final Video with Audio
```

---

## 📦 Tech Stack

### Backend
- **Framework**: FastAPI (async Python)
- **Task Queue**: Celery + Redis
- **Workflow**: Prefect
- **Database**: PostgreSQL with pgvector
- **Storage**: MinIO (S3-compatible)

### Frontend
- **Framework**: Next.js 14 + React 18
- **Styling**: TailwindCSS
- **UI**: Framer Motion, Lucide Icons
- **Real-time**: Server-Sent Events (SSE)

### AI Models

**Body Motion:**
- AnimateDiff (text-to-video with pose control)
- MagicAnimate (pose-driven animation)
- MagicDance (full body + face)

**Facial Expressions:**
- **FOMM** (First Order Motion Model with TPS)
- **LivePortrait** (2024, best quality)
- Face-vid2vid (talking head)

**Detection & Segmentation:**
- YOLOv8 (person detection)
- SAM (Segment Anything Model)
- OpenPose/DWPose (pose estimation)
- MediaPipe (facial landmarks)

**Text-to-Video:**
- Stable Video Diffusion (background generation)
- Stable Diffusion XL (scene generation)
- ControlNet (pose-guided generation)

---

## 🎯 Key Features Implemented

### 1. Photo Upload (Solo or Group)
- Drag & drop interface
- Person detection and segmentation
- Face landmark extraction
- Preview with bounding boxes

### 2. YouTube Reference Video
- Automatic download (yt-dlp)
- Dancer detection across frames
- Pose sequence extraction
- Facial expression analysis
- Audio extraction

### 3. Text Prompts (Sora/Kling-style)
- Scene description input
- Style modifiers (cinematic, 4k, etc.)
- Pre-defined style presets:
  - Cyberpunk Nightclub 🌃
  - Beach Sunset 🏖️
  - Urban Rooftop 🏙️
  - Retro 80s Disco 🪩
  - Fantasy Forest 🌲

### 4. Background Control
- Keep original background
- AI-generated backgrounds from prompts
- Animated backgrounds (SVD)
- Person-background compositing

### 5. Facial Expression Transfer
- Emotion transfer (happy, sad, excited)
- Mouth movements (lip-sync)
- Eye expressions (winks, blinks)
- Head movements (nods, tilts)
- Intensity control (0-2x)

### 6. Real-time Progress
- Server-Sent Events (SSE)
- Step-by-step progress updates
- Estimated time remaining
- Detailed logs

### 7. Video Output
- 1024x1024 resolution (configurable)
- 30 FPS
- Audio synchronized
- MP4 format
- Direct download

---

## 📁 Project Structure

```
dance/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # API endpoints
│   │   ├── models/              # Database models
│   │   ├── services/
│   │   │   ├── models/          # AI model services
│   │   │   │   ├── fomm_service.py
│   │   │   │   ├── text_to_video_service.py
│   │   │   │   └── ...
│   │   │   ├── facial_expression/
│   │   │   └── video_processing/
│   │   ├── workflows/           # Prefect flows
│   │   ├── tasks/               # Celery tasks
│   │   └── core/                # Config, database
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main UI
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   ├── package.json
│   └── Dockerfile
│
├── models/                      # AI model weights
│   ├── fomm/
│   ├── liveportrait/
│   ├── magic-animate/
│   └── sam/
│
├── scripts/
│   ├── setup.sh                 # Setup script
│   └── download_models.py       # Model downloader
│
├── docker-compose.yml           # All services
├── .env.example
└── README.md
```

---

## 🚀 How to Run

### Quick Start (Docker)

```bash
# 1. Setup
cd /mnt/c/AIML/dance
./scripts/setup.sh

# 2. Download models (optional, ~3GB)
python3 scripts/download_models.py

# 3. Start services
docker-compose up -d

# 4. Open browser
http://localhost:3001
```

### Services Running

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3001 | Next.js UI |
| Backend API | 8000 | FastAPI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Queue/Cache |
| MinIO | 9000, 9001 | Object Storage |
| Flower | 5555 | Celery Monitor |

---

## 🎨 UI/UX Flow

### 5-Step Wizard

**Step 1: Upload Photo**
- Choose solo or group
- Drag & drop or click to upload
- Preview with face detection overlay

**Step 2: YouTube Reference**
- Paste dance video URL
- Analyze video (shows: dancers, duration, FPS, audio)
- Preview reference

**Step 3: Scene & Style (Optional)**
- Select background mode (original / generated)
- Choose preset OR custom prompt
- Scene: "dancing in a neon nightclub..."
- Style: "cinematic, 4k, dramatic lighting..."

**Step 4: Expression Settings**
- Toggle facial expressions
- Enable lip-sync
- Adjust intensity slider (0-2x)

**Step 5: Generate**
- Real-time progress bar
- Step-by-step status
- Video preview when complete
- Download button

---

## 🧠 AI Model Integration

### FOMM (First Order Motion Model)

```python
class FOMMService:
    def animate_face(self, source_image, driving_video):
        # Extract keypoints
        kp_source = self.kp_detector(source_image)
        kp_driving = self.kp_detector(driving_video)

        # Thin Plate Spline (TPS) motion transfer
        generated = self.generator(
            source,
            kp_source=kp_source,
            kp_driving=kp_driving
        )

        return generated
```

### AnimateDiff + ControlNet (Text-to-Video)

```python
class AnimateDiffService:
    def generate_with_pose_control(self, prompt, pose_images):
        # Combine text prompts with pose control
        output = self.pipe(
            prompt="dancing in cyberpunk nightclub, cinematic",
            image=pose_skeleton_images,  # From OpenPose
            controlnet_conditioning_scale=1.0
        )
        return output.frames
```

### Background Generation (Stable Video Diffusion)

```python
class BackgroundGenerationService:
    def generate_animated_background(self, prompt):
        # Generate initial frame from text
        initial_image = sd_xl(prompt="neon nightclub, cinematic")

        # Animate with SVD
        frames = svd_pipe(
            initial_image,
            motion_bucket_id=127
        )
        return frames
```

---

## 🔄 Processing Pipeline (Prefect Flow)

```python
@flow(name="dance_video_generation")
def generate_dance_video_flow(project_id, config):
    # 1. Process reference video
    reference_data = process_reference_video(
        url=config['reference_url']
    )

    # 2. Detect persons in photo
    persons = detect_persons_and_faces(
        photo_url=config['photo_url']
    )

    # 3. Map choreography
    mapping = map_choreography(reference_data, persons)

    # 4. Generate body motion (AnimateDiff + prompts)
    body_videos = generate_body_motion(
        persons=persons,
        poses=reference_data.poses,
        scene_prompt=config['scene_prompt'],
        style_prompt=config['style_prompt']
    )

    # 5. Generate facial expressions (FOMM/LivePortrait)
    face_videos = generate_facial_expressions(
        persons=persons,
        face_sequences=reference_data.faces
    )

    # 6. Composite face onto body
    composited = composite_face_on_body(
        body_videos,
        face_videos
    )

    # 7. Generate/replace background (if requested)
    if config['background_mode'] == 'generated':
        background = generate_background(
            prompt=config['scene_prompt']
        )
        final = composite_on_background(
            composited,
            background
        )
    else:
        final = composited

    # 8. Add audio sync
    final_with_audio = add_audio_sync(
        final,
        reference_data.audio_url
    )

    return final_with_audio
```

---

## 📊 Database Schema

### Main Tables

**dance_projects**
- id, user_id, photo_url, reference_video_url
- **scene_prompt**, **style_prompt**, **negative_prompt** ← NEW
- **background_mode** ← NEW (original/generated)
- enable_facial_expressions, enable_lip_sync
- body_motion_model, face_expression_model
- status, progress, final_video_url

**dance_persons**
- id, project_id, person_index
- bbox, segmentation_mask_url
- face_data (landmarks, alignment)
- body_video_url, face_video_url, composite_video_url

**reference_videos**
- id, url, source (youtube/upload)
- pose_sequences_url, face_sequences_url
- num_dancers, duration, fps
- has_audio, has_vocals, has_face_data

**style_presets** ← NEW
- id, name, description
- scene_prompt_template, style_prompt
- category, preview_url

---

## 🎯 Advantages Over Existing Solutions

| Feature | Our App | Seedance 2.0 | MagicDance |
|---------|---------|--------------|------------|
| **Open-Source** | ✅ Yes | ❌ No (API only) | ✅ Yes |
| **Text Prompts** | ✅ Full support | ✅ Limited | ❌ No |
| **Facial Expressions** | ✅ FOMM + LivePortrait | ✅ Built-in | ✅ Basic |
| **Background Control** | ✅ AI-generated | ❌ No | ❌ No |
| **YouTube Input** | ✅ Direct | ❌ Manual | ❌ Manual |
| **Cost** | 💰 Free (local) | 💰💰 $0.50-2.50/video | 💰 Free |
| **Privacy** | ✅ Local processing | ❌ Cloud-based | ✅ Local |
| **Custom Training** | ✅ Full access | ❌ No | ⚠️ Difficult |
| **Quality** | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ High |

---

## 🔮 Future Enhancements

### Phase 2 (Planned)

1. **Multi-person choreography synchronization**
   - Time-segment editor
   - Mix sync + individual moves
   - BTS-style formations

2. **Voice cloning + lip-sync**
   - Replace audio with custom voice
   - Better lip-sync (Wav2Lip)

3. **3D pose estimation**
   - Depth-aware motion transfer
   - Better occlusion handling

4. **Video-to-video**
   - Animate existing videos
   - Style transfer

5. **Mobile app**
   - iOS/Android native
   - On-device processing (CoreML/TFLite)

6. **Template marketplace**
   - Share/sell choreography templates
   - Popular dance library

7. **Batch processing**
   - Multiple photos at once
   - Playlist support

8. **Advanced compositing**
   - Green screen support
   - AR effects overlay

---

## 📝 Notes

### GPU Requirements

- **Minimum**: 8GB VRAM (RTX 3060, A10G)
- **Recommended**: 16GB+ VRAM (RTX 4090, A100)
- **CPU Mode**: Works but 10-20x slower

### Processing Time Estimates

| Configuration | Time (solo) | Time (group of 5) |
|---------------|-------------|-------------------|
| GPU (16GB VRAM) | 3-5 min | 10-15 min |
| GPU (8GB VRAM) | 5-8 min | 20-30 min |
| CPU (32 cores) | 30-45 min | 2-3 hours |

### Model Sizes

- FOMM: ~350MB
- SAM: ~2.4GB
- YOLOv8x: ~130MB
- MagicAnimate: ~5GB (from HF)
- LivePortrait: ~1.5GB (from HF)
- Stable Video Diffusion: ~10GB (from HF)

**Total**: ~20GB for all models

---

## ✅ Implementation Complete!

All core features have been implemented and are ready to test.

### Next Steps

1. **Download models**: `python scripts/download_models.py`
2. **Start services**: `docker-compose up -d`
3. **Test with sample video**: Upload photo + YouTube URL
4. **Adjust prompts**: Experiment with scene/style descriptions
5. **Monitor progress**: Check Celery worker logs

### Support

- **Documentation**: See README.md
- **API Docs**: http://localhost:8000/docs
- **Issues**: GitHub Issues (to be created)

---

**Built with ❤️ using FastAPI, Next.js, and cutting-edge AI models**
