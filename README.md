# AI Interview Backend

Node.js WebSocket server for AI-powered interview platform using OpenAI Whisper for transcription and GPT for interview generation.

## 🚀 Tech Stack

- **Runtime**: Node.js 20 LTS
- **Framework**: Express.js
- **WebSocket**: ws library
- **AI Services**: 
  - OpenAI GPT-4 (Interview questions & evaluation)
  - OpenAI Whisper (Speech-to-text transcription)
  - OpenAI TTS (Text-to-speech)
- **State Management**: LangGraph
- **Hosting**: Google Cloud Run

## 📋 Prerequisites

1. **Google Cloud Account** - Sign up at https://cloud.google.com
2. **Google Cloud SDK (gcloud CLI)** - Install from https://cloud.google.com/sdk/docs/install
3. **OpenAI API Key** - Get from https://platform.openai.com/account/api-keys
4. **Node.js 20+** (for local development)

## 🛠️ Local Development

### Installation

```bash
# Install dependencies
npm install
```

### Environment Variables

**Option 1: Using .env file (Recommended for local development)**

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
PORT=3001
NODE_ENV=development
```

**Option 2: Hardcode as fallback (For convenience)**

You can also set a fallback API key directly in `server.js` on line 51:

```javascript
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'sk-your-openai-api-key-here';
```

**Note**: The environment variable takes precedence over the fallback. The fallback is useful if you forget to set the environment variable.

### Run Locally

```bash
npm start
```

The server will start on `http://localhost:3001`

## ☁️ Google Cloud Deployment

### Initial Setup

1. **Install Google Cloud SDK**

   ```bash
   # macOS
   brew install --cask google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Login to Google Cloud**

   ```bash
   gcloud auth login
   ```

3. **Set Your Project**

   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Enable Required APIs**

   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

### Deploy to Cloud Run

#### Option 1: Deploy with Source (Recommended)

This builds and deploys directly from your source code:

```bash
# Navigate to backend directory
cd ai-interview-backends

# Deploy to Cloud Run
gcloud run deploy ai-interview-backend \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 3600 \
  --max-instances 10 \
  --set-env-vars "NODE_ENV=production,OPENAI_API_KEY=sk-your-openai-api-key-here"
```

**Important**: Replace `sk-your-openai-api-key-here` with your actual OpenAI API key.

#### Option 2: Deploy with Docker Build

If you prefer to build the Docker image separately:

```bash
# Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/ai-interview-backend .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/ai-interview-backend

# Deploy to Cloud Run
gcloud run deploy ai-interview-backend \
  --image gcr.io/YOUR_PROJECT_ID/ai-interview-backend \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 3600 \
  --max-instances 10 \
  --set-env-vars "NODE_ENV=production,OPENAI_API_KEY=sk-your-openai-api-key-here"
```

### After Deployment

You'll receive a URL like:
```
https://ai-interview-backend-xxxxx-uc.a.run.app
```

This is your backend WebSocket URL. Use `wss://` (secure WebSocket) for production.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes* | Fallback in server.js line 51 |
| `PORT` | Server port | No | 8080 (production) / 3001 (dev) |
| `NODE_ENV` | Environment | No | development |

\* You can set the API key either via environment variable or as a fallback in `server.js` line 51. The environment variable takes precedence.

### Update Environment Variables

```bash
gcloud run services update ai-interview-backend \
  --region us-central1 \
  --update-env-vars "OPENAI_API_KEY=new-key-here"
```

## 📊 Monitoring & Logs

### View Logs

```bash
# View recent logs
gcloud run services logs read ai-interview-backend --region us-central1 --limit 50

# Follow logs in real-time
gcloud run services logs tail ai-interview-backend --region us-central1
```

### View Service Details

```bash
gcloud run services describe ai-interview-backend --region us-central1
```

### Check Service Status

```bash
gcloud run services list --region us-central1
```

## 🔄 Updating the Deployment

### Push Code Changes

1. **Make your code changes**
2. **Commit to git** (if using version control)
3. **Redeploy**:

   ```bash
   gcloud run deploy ai-interview-backend \
     --source . \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --port 8080 \
     --memory 512Mi \
     --cpu 1 \
     --timeout 3600 \
     --max-instances 10 \
     --set-env-vars "NODE_ENV=production"
   ```

   Note: The API key is set as a fallback in the code, so you don't need to pass it every time unless you want to update it.

## 🌐 WebSocket Protocol

### Client → Server Messages

- `{type:"start", language, role, level, maxTurns, candidateName?, voice?}` - Start interview
- `{type:"answer_audio_start"}` - Begin audio recording
- `(binary audio chunks...)` - Audio data (WebM/Opus format)
- `{type:"answer_audio_end"}` - End audio recording
- `{type:"stop"}` - Stop interview early

### Server → Client Messages

- `{type:"session", sessionId}` - Session created
- `{type:"persona", text}` - Interviewer persona
- `(binary)` - TTS audio chunks
- `{type:"tts_done", format}` - TTS playback complete
- `{type:"transcript_update", transcript}` - Updated conversation transcript
- `{type:"done", summaryText, overallScore, rubric}` - Interview evaluation complete
- `{type:"error", error}` - Error occurred

## 📁 Project Structure

```
ai-interview-backends/
├── server.js          # Main server file
├── package.json       # Dependencies
├── Dockerfile         # Container configuration
├── .dockerignore     # Docker ignore file
├── .gcloudignore     # Cloud deployment ignore file
└── tmp/              # Temporary files (local only, uses /tmp in Cloud Run)
```

## 🔒 Security Notes

1. **API Keys**: Never commit API keys to git. Use environment variables or Google Cloud Secrets Manager.
2. **CORS**: Currently allows all origins. In production, restrict to your frontend domain.
3. **WebSocket**: Uses secure WebSocket (wss://) in production.

## 💰 Google Cloud Costs

- **Cloud Run**: 
  - Free tier: 2 million requests/month
  - Pay-as-you-go: $0.40 per million requests after free tier
  - Memory/CPU: Billed per second of usage
- **Container Registry**: Free for Cloud Run deployments

## 🐛 Troubleshooting

### Container Fails to Start

- Check logs: `gcloud run services logs read ai-interview-backend --region us-central1`
- Verify environment variables are set
- Ensure Node.js 20 is used (check Dockerfile)

### Transcription Not Working

- Verify OpenAI API key is correct
- Check that audio chunks are being received (check logs)
- Ensure `/tmp` directory is writable (Cloud Run requirement)

### WebSocket Connection Issues

- Verify the service URL uses `wss://` (not `ws://`)
- Check CORS settings
- Ensure service is publicly accessible (`--allow-unauthenticated`)

## 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

## 📝 License

ISC

