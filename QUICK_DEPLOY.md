# ⚡ Quick Deploy to Railway (5 Minutes)

## Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

## Step 2: Deploy Backend

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect it's Python
5. Add environment variable:
   - `OPENAI_API_KEY` = your key (if using)
6. Click "Deploy"

**Backend URL**: `https://your-app.railway.app`

## Step 3: Deploy Frontend

1. In Railway, click "+ New" → "Empty Service"
2. Connect to same GitHub repo
3. Settings:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npx serve -s dist -l 3000`
4. Add environment variable:
   - `VITE_API_URL` = `https://your-backend-url.railway.app/api`
5. Deploy!

## Step 4: Update Backend CORS

1. Go to backend service → Variables
2. Add: `CORS_ORIGINS` = `https://your-frontend-url.railway.app`
3. Redeploy backend

## Done! 🎉

Your app is live at: `https://your-frontend-url.railway.app`

---

## Alternative: Render.com (Similar Process)

1. Go to https://render.com
2. New → Web Service
3. Connect GitHub repo
4. Settings:
   - **Root Directory**: `backend` (or `frontend`)
   - **Build**: `pip install -r requirements.txt` (or `npm install`)
   - **Start**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy!

---

## Cost: FREE on both platforms for demos! 🎉
