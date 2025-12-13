#!/bin/bash

# Deployment helper script for ED Bottleneck Engine

echo "🚀 ED Bottleneck Engine - Deployment Helper"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "⚠️  Git not initialized. Initializing..."
    git init
    git add .
    git commit -m "Initial commit for deployment"
    echo "✅ Git initialized"
    echo ""
fi

# Check if code is pushed to GitHub
echo "📋 Deployment Checklist:"
echo ""

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes:"
    git status --short
    echo ""
    read -p "Commit and push changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "Commit message: " commit_msg
        git commit -m "${commit_msg:-Prepare for deployment}"
        echo "✅ Changes committed"
    fi
fi

# Check if remote exists
if ! git remote | grep -q origin; then
    echo "⚠️  No GitHub remote found"
    read -p "Enter your GitHub repository URL (or press Enter to skip): " repo_url
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✅ Remote added"
    fi
fi

# Push to GitHub
if git remote | grep -q origin; then
    read -p "Push to GitHub? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push -u origin main || git push -u origin master
        echo "✅ Pushed to GitHub"
    fi
fi

echo ""
echo "=========================================="
echo "📚 Next Steps:"
echo ""
echo "1. Railway (Recommended - Easiest):"
echo "   → Go to https://railway.app"
echo "   → New Project → Deploy from GitHub"
echo "   → Select your repo"
echo "   → See QUICK_DEPLOY.md for details"
echo ""
echo "2. Render.com:"
echo "   → Go to https://render.com"
echo "   → New Web Service"
echo "   → Connect GitHub repo"
echo "   → See DEPLOYMENT_GUIDE.md for details"
echo ""
echo "3. Docker:"
echo "   → docker-compose -f docker-compose.prod.yml up -d"
echo ""
echo "📖 Full guide: DEPLOYMENT_GUIDE.md"
echo "⚡ Quick guide: QUICK_DEPLOY.md"
echo ""
