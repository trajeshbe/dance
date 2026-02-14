# Google Colab Integration

Use Google Colab's free/paid GPUs as remote workers for video generation!

## Benefits

- **Free GPU Access**: T4 GPU (16GB VRAM) for free
- **Better GPUs**: V100 (Colab Pro) or A100 (Colab Pro+)
- **No Local GPU Required**: Process videos without local hardware
- **Scalable**: Run multiple Colab notebooks for parallel processing

## Setup Instructions

### 1. Open Colab Notebook

Upload `dance_generator_colab_worker.ipynb` to Google Colab or use this link:
- Click: File → Upload notebook → Select the .ipynb file

### 2. Enable GPU

- Runtime → Change runtime type
- Hardware accelerator: **GPU**
- GPU type: T4 (free) / V100 (Pro) / A100 (Pro+)
- Click **Save**

### 3. Get Ngrok Auth Token

1. Sign up at https://dashboard.ngrok.com/signup
2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy the token

### 4. Run Colab Notebook

Run all cells in order:

**Cell 1: Install Dependencies**
```python
!pip install fastapi uvicorn pyngrok torch ...
```

**Cell 2: Check GPU**
```python
# Should show: GPU Available: True
```

**Cell 3: Clone Models**
```python
# Clones FOMM, MagicAnimate, etc.
```

**Cell 4: FastAPI Server**
```python
# Creates the worker API
```

**Cell 5: Start Ngrok**
```python
# IMPORTANT: Replace YOUR_NGROK_TOKEN_HERE with your token!
ngrok_auth_token = "YOUR_ACTUAL_TOKEN"
```
After running, you'll see:
```
🌐 Colab Worker URL: https://abc123.ngrok.io
```
**Copy this URL!**

**Cell 6: Start Server**
```python
# Runs FastAPI server - keep this cell running!
```

### 5. Configure Backend

Edit `/mnt/c/AIML/dance/.env`:

```bash
# Add the ngrok URL from Colab
COLAB_WORKER_URL=https://abc123.ngrok.io

# Prefer Colab over local GPU
PREFER_COLAB=true
```

### 6. Restart Backend

```bash
docker-compose restart backend celery_worker
```

## Usage

Once configured, the system will automatically:

1. Check if Colab worker is available
2. Route jobs to Colab if available
3. Fall back to local GPU if Colab is down

You can monitor in logs:
```bash
docker-compose logs -f celery_worker
```

Look for:
```
Using Google Colab for GPU processing
Colab GPU: Tesla T4, 15.0 GB
Colab job started: abc-123
```

## GPU Comparison

| GPU | VRAM | Cost | Speed | Best For |
|-----|------|------|-------|----------|
| **T4** (Colab Free) | 16GB | Free | Good | Solo videos, testing |
| **V100** (Colab Pro) | 16GB | $10/mo | Better | Regular use |
| **A100** (Colab Pro+) | 40GB | $50/mo | Best | Group videos, batch |
| Local RTX 3060 | 12GB | One-time | Good | Privacy |
| Local RTX 4090 | 24GB | One-time | Excellent | Best local option |

## Monitoring

### Check Colab Status

```bash
curl https://YOUR-COLAB-URL.ngrok.io/health
```

Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "timestamp": "2026-02-13T12:00:00"
}
```

### Get GPU Info

```bash
curl https://YOUR-COLAB-URL.ngrok.io/gpu_info
```

Response:
```json
{
  "name": "Tesla T4",
  "memory_total": 15.0,
  "memory_allocated": 2.5,
  "memory_reserved": 3.0
}
```

## Troubleshooting

### Colab Disconnects

Colab free tier disconnects after 12 hours of inactivity. To keep alive:

1. Install Colab Pro ($10/mo)
2. Or use this browser extension: [Colab Alive](https://chrome.google.com/webstore/detail/colab-alive/)
3. Or click in the notebook periodically

### Ngrok Tunnel Expires

Free ngrok tunnels last 2 hours. Solutions:

1. Upgrade to ngrok Pro ($8/mo) for persistent URLs
2. Re-run the ngrok cell when it expires (get new URL)
3. Update backend .env with new URL

### Out of Memory

If getting OOM errors on Colab:

1. Reduce `VIDEO_OUTPUT_RESOLUTION` in .env (try 512 instead of 1024)
2. Process fewer people at once
3. Upgrade to V100 (Colab Pro) or A100 (Colab Pro+)

### Slow Processing

Colab free tier may throttle after heavy use. Solutions:

1. Upgrade to Colab Pro for faster GPUs
2. Use multiple Colab accounts (against ToS, not recommended)
3. Process during off-peak hours

## Advanced: Multiple Colab Workers

You can run multiple Colab notebooks for parallel processing:

1. Open 2-3 Colab tabs
2. Run the notebook in each
3. Get different ngrok URLs for each
4. Configure load balancing in backend (TODO: implement)

## Cost Breakdown

**Free Option:**
- Colab Free: $0/mo
- Ngrok Free: $0/mo
- **Total: $0/mo**

**Pro Option:**
- Colab Pro: $10/mo (V100 GPU)
- Ngrok Pro: $8/mo (persistent URLs)
- **Total: $18/mo**

**Pro+ Option:**
- Colab Pro+: $50/mo (A100 GPU, priority)
- Ngrok Pro: $8/mo
- **Total: $58/mo**

Compare to cloud GPU:
- AWS A10G: ~$0.50-1/hour = $360-720/mo continuous
- RunPod A100: ~$1.69/hour = $1200/mo continuous

**Colab is much cheaper for occasional use!**

## Tips

1. **Keep notebook tab open** - Colab needs browser connection
2. **Monitor GPU usage** - Check /gpu_info endpoint
3. **Test with small videos first** - Verify setup before long generations
4. **Use Colab Pro for production** - More reliable than free tier
5. **Clear output periodically** - Keeps notebook responsive

## Security

⚠️ **Never commit ngrok tokens to git!**

The ngrok URL is publicly accessible. Anyone with the URL can:
- Submit jobs to your Colab worker
- Check job status

To secure (optional):
1. Add API key authentication to the Colab worker
2. Use ngrok IP restrictions
3. Use VPN or private networks

For personal use, the risk is low since URLs are random and temporary.

---

**Questions?** See main README or create an issue on GitHub.
