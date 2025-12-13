# 🚀 Deployment Guide - ED Bottleneck Engine

This guide covers deploying your application to production for customer demos.

## 📋 Quick Deployment Options

### Option 1: Railway (Recommended - Easiest)
- **Best for**: Quick demos, automatic deployments
- **Cost**: Free tier available, ~$5-20/month
- **Setup time**: 10 minutes
- **URL**: https://railway.app

### Option 2: Render
- **Best for**: Simple deployments, good free tier
- **Cost**: Free tier available, ~$7-25/month
- **Setup time**: 15 minutes
- **URL**: https://render.com

### Option 3: Fly.io
- **Best for**: Global edge deployment
- **Cost**: Free tier available, ~$5-15/month
- **Setup time**: 20 minutes
- **URL**: https://fly.io

### Option 4: AWS/GCP/Azure
- **Best for**: Enterprise, full control
- **Cost**: $20-100+/month
- **Setup time**: 1-2 hours
- **Best for**: Production with high traffic

---

## 🎯 Recommended: Railway Deployment (Fastest)

### Step 1: Prepare Your Code

1. **Push to GitHub** (if not already):
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

### Step 2: Deploy Backend

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add a new service → "Empty Service"
5. Configure:
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: `backend`
6. Add environment variables:
   - `OPENAI_API_KEY` (if using)
   - `PORT` (Railway sets this automatically)
7. Deploy!

### Step 3: Deploy Frontend

1. Add another service → "Empty Service"
2. Configure:
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Start Command**: `cd frontend && npm run preview -- --host 0.0.0.0 --port $PORT`
   - **Root Directory**: `frontend`
3. Add environment variable:
   - `VITE_API_URL` = `https://your-backend-url.railway.app/api`
4. Deploy!

### Step 4: Get Your URLs

Railway will give you:
- Backend URL: `https://your-app-backend.railway.app`
- Frontend URL: `https://your-app-frontend.railway.app`

---

## 🔧 Production Configuration

### Environment Variables

Create a `.env.production` file:

```bash
# Backend
OPENAI_API_KEY=your_key_here
CORS_ORIGINS=https://your-frontend-url.com
ENVIRONMENT=production

# Frontend
VITE_API_URL=https://your-backend-url.com/api
```

### Update CORS Settings

The backend needs to allow your production frontend URL. Update `backend/app/core/config.py`:

```python
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-frontend-url.com",  # Add your production URL
]
```

---

## 🐳 Docker Production Deployment

### Build Production Images

```bash
# Build backend
cd backend
docker build -t ed-backend:prod .

# Build frontend
cd ../frontend
docker build -t ed-frontend:prod -f Dockerfile.prod .
```

### Run with Docker Compose

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## 📊 Database Considerations

**Current**: SQLite (file-based, not ideal for production)

**For Production**, consider:
- **PostgreSQL** (recommended) - Free on Railway/Render
- **MySQL** - Alternative option
- Keep SQLite for demos if traffic is low

Migration guide available in `MIGRATION_GUIDE.md` (create if needed).

---

## 🔒 Security Checklist

- [ ] Set strong environment variables
- [ ] Enable HTTPS (automatic on Railway/Render)
- [ ] Update CORS to only allow your domain
- [ ] Set rate limiting (already configured)
- [ ] Use environment variables for secrets (never commit)
- [ ] Enable database backups (if using PostgreSQL)

---

## 🌐 Custom Domain (Optional)

### Railway
1. Go to your service → Settings → Domains
2. Add custom domain
3. Update DNS records as instructed

### Render
1. Go to your service → Settings → Custom Domains
2. Add domain
3. Update DNS

---

## 📈 Monitoring

### Recommended Tools:
- **Railway**: Built-in logs and metrics
- **Sentry**: Error tracking (free tier)
- **Uptime Robot**: Uptime monitoring (free)

---

## 🚨 Troubleshooting

### Backend not starting:
- Check logs: `railway logs` or in Railway dashboard
- Verify environment variables are set
- Check port binding (use `$PORT` variable)

### Frontend can't connect:
- Verify `VITE_API_URL` is correct
- Check CORS settings in backend
- Ensure backend is running

### Database issues:
- SQLite works for demos but has limitations
- Consider PostgreSQL for production
- Check file permissions if using SQLite

---

## 📞 Support

For deployment issues:
1. Check Railway/Render logs
2. Verify environment variables
3. Test locally first with production settings
4. Check GitHub issues or create one

---

## 🎉 Next Steps After Deployment

1. **Test the live URL** - Make sure everything works
2. **Share with customers** - Send them the frontend URL
3. **Monitor usage** - Watch logs for errors
4. **Set up alerts** - Get notified of issues
5. **Backup data** - Regular backups if using database

---

## 💰 Cost Estimates

### Railway (Recommended)
- Free tier: 500 hours/month
- Hobby: $5/month (unlimited)
- Pro: $20/month (better performance)

### Render
- Free tier: Spins down after inactivity
- Starter: $7/month per service
- Standard: $25/month per service

### Fly.io
- Free tier: 3 shared VMs
- Paid: ~$5-15/month depending on usage

---

**Ready to deploy? Start with Railway - it's the fastest!** 🚀
