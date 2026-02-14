# Quick Start Guide

Get up and running with Dance Video Generator in **5 minutes**!

## Option 1: Docker (Recommended)

### Prerequisites
- Docker & Docker Compose installed
- (Optional) NVIDIA GPU for local processing
- (Optional) Google Colab account for remote GPU

### Steps

```bash
# 1. Navigate to project
cd /mnt/c/AIML/dance

# 2. Create .env file
cp .env.example .env

# 3. Start services
docker-compose up -d

# 4. Open browser
http://localhost:3001
```

**That's it!** The app is running.

---

## Option 2: With Google Colab GPU (Free!)

Use Colab's free GPU instead of local hardware.

### Setup (5 minutes)

**1. Start Backend**
```bash
cd /mnt/c/AIML/dance
docker-compose up -d
```

**2. Open Colab Notebook**
- Upload `colab/dance_generator_colab_worker.ipynb` to [Google Colab](https://colab.research.google.com/)
- Runtime → Change runtime type → GPU → T4
- Click Save

**3. Get Ngrok Token**
- Sign up: https://dashboard.ngrok.com/signup
- Copy token: https://dashboard.ngrok.com/get-started/your-authtoken

**4. Run Colab Cells**
- Cell 1-4: Install & setup (run all)
- Cell 5: **Paste your ngrok token** → Run
- Copy the URL: `https://abc123.ngrok.io`
- Cell 6: Start server (keep running!)

**5. Configure Backend**
```bash
# Edit .env
nano .env

# Add this line:
COLAB_WORKER_URL=https://abc123.ngrok.io
PREFER_COLAB=true

# Save (Ctrl+O, Enter, Ctrl+X)

# Restart
docker-compose restart backend celery_worker
```

**Done!** Now all video processing uses Colab's free GPU!

---

## Testing

### 1. Create Your First Video

**Open:** http://localhost:3001

**Upload:**
- Step 1: Upload a photo (solo or group)
- Step 2: Paste YouTube URL (e.g., BTS Dynamite: `https://youtube.com/watch?v=gdZLi9oWNZg`)
- Step 3: (Optional) Add scene prompt: "dancing in a neon nightclub"
- Step 4: Enable facial expressions
- Step 5: Click "Generate"

**Wait:** 5-15 minutes for processing

**Download:** Final video when complete!

### 2. Check Logs

```bash
# Backend logs
docker-compose logs -f backend

# Worker logs (see if using Colab or local)
docker-compose logs -f celery_worker

# Should see:
# "Using Google Colab for GPU processing"  ← if Colab configured
# OR
# "Using local GPU/CPU for processing"      ← if local only
```

### 3. Monitor Progress

- **Frontend**: Real-time progress bar
- **Celery**: http://localhost:5555 (Flower dashboard)
- **API**: http://localhost:8000/docs

---

## Troubleshooting

### Can't access frontend?

```bash
# Check if services are running
docker-compose ps

# Should see:
# dance_frontend    Up
# dance_backend     Up
# dance_postgres    Up (healthy)
# dance_redis       Up (healthy)
```

### Stuck at "Analyzing reference"?

- Check internet connection (needs to download YouTube video)
- Check backend logs: `docker-compose logs backend`

### Colab not working?

```bash
# Test Colab connection
curl https://YOUR-COLAB-URL.ngrok.io/health

# Should return:
# {"status":"healthy","gpu_available":true}

# If not, check:
# 1. Colab notebook Cell 6 is still running?
# 2. Ngrok URL correct in .env?
# 3. Backend restarted after changing .env?
```

### Videos processing slowly?

- **With Colab**: Should take 5-10 min
- **Without GPU**: Can take 30-60 min on CPU

Check which is being used:
```bash
docker-compose logs celery_worker | grep "Using"
```

---

## Next Steps

### Download Models (Optional)

For local processing without internet:

```bash
python3 scripts/download_models.py
```

This downloads ~20GB of AI models.

### Explore Features

- **Style Presets**: Try different scenes (cyberpunk, beach, etc.)
- **Expression Intensity**: Adjust from subtle (0.5x) to dramatic (2.0x)
- **Background Mode**: Generate AI backgrounds
- **Solo vs Group**: Works for 1 person or 10+ people

### Advanced Configuration

Edit `.env` to customize:

```bash
# Video quality
VIDEO_OUTPUT_RESOLUTION=1024  # 512, 1024, 2048
VIDEO_OUTPUT_FPS=30

# Model selection
FACIAL_EXPRESSION_MODEL=liveportrait  # or fomm
BODY_MOTION_MODEL=animatediff  # or magicanimate

# Parallel processing
MAX_CONCURRENT_JOBS=2
```

### Use Proprietary APIs (Optional)

For highest quality, add API keys to `.env`:

```bash
# Seedance 2.0 (best quality)
SEEDANCE_API_KEY=your_key_here

# Replicate (hosted models)
REPLICATE_API_TOKEN=your_token_here
```

---

## Common Workflows

### Workflow 1: Quick Test (2 min setup)

```bash
docker-compose up -d
# Open http://localhost:3001
# Upload photo + YouTube URL
# Generate!
```

**Best for**: Quick testing, CPU-only machines

---

### Workflow 2: Colab GPU (5 min setup)

```bash
# Backend
docker-compose up -d

# Colab
# 1. Open notebook
# 2. Add ngrok token
# 3. Run all cells
# 4. Copy URL to .env
# 5. Restart backend
```

**Best for**: No local GPU, free processing

---

### Workflow 3: Local GPU (production)

```bash
# Download models
python3 scripts/download_models.py

# Set .env
USE_GPU=true
CUDA_VISIBLE_DEVICES=0

# Start with GPU
docker-compose up -d
```

**Best for**: Privacy, offline use, best performance

---

## GPU Comparison

| Option | Speed (solo) | Cost | Setup |
|--------|-------------|------|-------|
| **Colab Free (T4)** | 5-8 min | $0 | 5 min |
| **Colab Pro (V100)** | 3-5 min | $10/mo | 5 min |
| **Local RTX 3060** | 5-8 min | One-time | Complex |
| **Local RTX 4090** | 2-3 min | One-time | Complex |
| **CPU Only** | 30-60 min | $0 | 2 min |

**Recommendation**: Start with **Colab Free** (best value for testing)

---

## Getting Help

### Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs backend
docker-compose logs celery_worker

# Follow (live)
docker-compose logs -f backend
```

### Status

```bash
# Service health
curl http://localhost:8000/health

# GPU info (if local)
docker exec dance_backend nvidia-smi

# Colab GPU info
curl https://YOUR-COLAB-URL.ngrok.io/gpu_info
```

### Reset

```bash
# Stop all
docker-compose down

# Remove volumes (⚠️ deletes database)
docker-compose down -v

# Start fresh
docker-compose up -d
```

---

## Summary

**Fastest Start**: Option 1 (Docker only) - 2 minutes
**Best Performance**: Option 2 (Docker + Colab) - 5 minutes
**Most Flexible**: Both options + download models

Pick your path and enjoy creating AI dance videos! 🎬✨

---

**Need help?** Check:
- Full README: `README.md`
- Colab Guide: `colab/README.md`
- Implementation Details: `IMPLEMENTATION_SUMMARY.md`
