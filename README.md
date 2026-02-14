# Dance Video Generator

AI-powered dance video generator that transforms photos into dancing videos with facial expressions.

## Features

- **Upload Photos**: Solo or group pictures
- **YouTube Reference**: Paste any dance video URL
- **Facial Expression Transfer**: Using FOMM, LivePortrait, TPS
- **Body Motion Transfer**: Using MagicAnimate, MagicDance, AnimateDiff
- **Text Prompts**: Sora/Kling-style scene and style control
- **Audio Sync**: Lip-sync and beat synchronization
- **Google Colab Integration**: Use Colab's free GPU as remote worker! ⭐
- **Model Options**: Open-source (local) or proprietary (API)

## Architecture

### Hybrid Approach for Best Quality

**Body Motion Models:**
- MagicAnimate (pose-driven animation)
- MagicDance (with basic facial support)
- Moore-AnimateAnyone

**Facial Expression Models (BEST):**
- **LivePortrait** (2024) - State-of-the-art portrait animation
- **FOMM** (First Order Motion Model) - Proven facial reenactment
- **Face-vid2vid** - Talking head generation
- **TPS Motion Model** - Thin Plate Spline warping

### Processing Pipeline

1. **Reference Video Processing**
   - Download YouTube video
   - Extract audio track
   - Detect dancers (YOLO)
   - Extract body poses (OpenPose/DWPose)
   - Extract facial expressions (MediaPipe + FOMM keypoints)

2. **Input Photo Processing**
   - Detect persons (YOLO)
   - Segment persons (SAM)
   - Detect faces (MediaPipe/Dlib)
   - Extract facial landmarks

3. **Video Generation (Hybrid)**
   - Generate body motion video (MagicAnimate)
   - Generate facial expression video (FOMM/LivePortrait)
   - Composite face onto body (FFmpeg)
   - Add audio sync

4. **Final Composition**
   - Combine all persons
   - Audio synchronization
   - Export final video

## Tech Stack

### Backend
- FastAPI (Python 3.11+)
- PostgreSQL (database)
- Redis (job queue, caching)
- MinIO (object storage)
- Celery (async tasks)
- Prefect (workflow orchestration)

### Frontend
- Next.js 14
- React 18
- TypeScript
- TailwindCSS
- Framer Motion

### AI Models
- YOLOv8 (person detection)
- SAM (segmentation)
- OpenPose/DWPose (pose extraction)
- MediaPipe (facial landmarks)
- FOMM (facial expression transfer)
- LivePortrait (portrait animation)
- MagicAnimate (body motion)
- MagicDance (full animation)

## Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (recommended for local models, optional)
- 16GB+ RAM (32GB+ recommended for GPU)
- CUDA 11.8+ (for GPU acceleration)

### Quick Start (Docker - Recommended)

```bash
# 1. Navigate to project
cd /mnt/c/AIML/dance

# 2. Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Download AI models (optional, ~3GB)
python3 scripts/download_models.py

# 4. Start all services with Docker Compose
docker-compose up -d

# 5. Check logs
docker-compose logs -f backend

# 6. Open in browser
# Frontend: http://localhost:3001
# API Docs: http://localhost:8000/docs
# Celery Monitor: http://localhost:5555
# MinIO Console: http://localhost:9001
```

### Services

The Docker Compose setup includes:

- **Frontend** (Next.js) - Port 3001
- **Backend** (FastAPI) - Port 8000
- **PostgreSQL** - Port 5432
- **Redis** - Port 6379
- **MinIO** (Object Storage) - Ports 9000, 9001
- **Celery Worker** (Video Generation)
- **Flower** (Celery Monitoring) - Port 5555

### Manual Installation (Without Docker)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Start PostgreSQL, Redis, MinIO separately
# Update .env with connection details

# Run backend
uvicorn app.main:app --reload --port 8000

# Run frontend
npm run dev

# Run Celery worker
celery -A app.celery_app worker --loglevel=info
```

Access the app at http://localhost:3001

### Google Colab Integration (Recommended!)

**Use Colab's FREE GPU** instead of local hardware:

1. **Quick Setup**: See [colab/README.md](colab/README.md)
2. **Upload notebook**: `colab/dance_generator_colab_worker.ipynb` to [Google Colab](https://colab.research.google.com/)
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
4. **Run notebook**: Get ngrok URL
5. **Configure**: Add URL to `.env` as `COLAB_WORKER_URL`
6. **Done**: All processing now uses Colab GPU!

**Benefits**:
- ✅ Free T4 GPU (16GB VRAM)
- ✅ Faster than CPU (5-10 min vs 30-60 min)
- ✅ No local GPU required
- ✅ Upgrade to V100/A100 with Colab Pro/Pro+

See detailed guide: [colab/README.md](colab/README.md)

Access the app at http://localhost:3001

## Model Downloads

### Open-Source Models (Local)

```bash
# Body motion models
git clone https://github.com/magic-research/magic-animate
git clone https://github.com/Boese0601/MagicDance

# Facial expression models
git clone https://github.com/AliaksandrSiarohin/first-order-model
git clone https://github.com/KwaiVGI/LivePortrait
```

### Pre-trained Weights

Download from HuggingFace:
- LivePortrait: `KwaiVGI/LivePortrait`
- FOMM: `first-order-model/vox-256.pth`
- MagicAnimate: `zcxu-eric/MagicAnimate`
- MagicDance: `boese0601/MagicDance`

## Usage

### Creating a Dance Video

1. **Upload Photo**
   - Solo or group picture
   - JPG/PNG up to 10MB

2. **YouTube Reference**
   - Paste any dance video URL
   - System analyzes dancers, poses, and expressions

3. **Text Prompts** (Sora/Kling-style)
   - Optional: Add scene description
   - Example: "dancing in a cyberpunk nightclub"
   - Choose from presets or write custom

4. **Expression Settings**
   - Toggle facial expressions
   - Enable lip-sync
   - Adjust intensity

5. **Generate**
   - Processing takes 5-15 minutes
   - Real-time progress updates
   - Download final video

### Example Prompts

**Scene Prompts:**
- "dancing in a neon-lit nightclub with holographic displays"
- "dancing on a beach at sunset with golden light"
- "dancing on an urban rooftop with city skyline at night"
- "dancing in a magical forest with glowing fireflies"
- "dancing in a retro 80s disco with disco ball"

**Style Prompts:**
- "cinematic, 4k, dramatic lighting, professional photography"
- "vibrant colors, energetic, high contrast, dynamic"
- "warm golden hour, soft focus, dreamy atmosphere"

## API Documentation

Once running, access:
- **API Docs**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

### Key Endpoints

```
POST /api/v1/dance/reference/analyze
  - Analyze YouTube reference video

POST /api/v1/dance/upload/photo
  - Upload photo (solo or group)

POST /api/v1/dance/generate
  - Generate dance video with prompts

GET /api/v1/dance/status/{job_id}
  - Get real-time generation progress (SSE)

GET /api/v1/style-presets
  - Get pre-defined style presets
```

## License

MIT License

## Credits

- First Order Motion Model: https://github.com/AliaksandrSiarohin/first-order-model
- LivePortrait: https://github.com/KwaiVGI/LivePortrait
- MagicAnimate: https://github.com/magic-research/magic-animate
- MagicDance: https://github.com/Boese0601/MagicDance
