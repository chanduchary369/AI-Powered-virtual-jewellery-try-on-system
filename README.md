# Virtual Jewellery Try-On (Local Dev)

## Structure
- backend: Express server that proxies image removal requests to remove.bg
- frontend: Vite + React app that shows webcam and overlays earrings

## Quick start
1. Backend:
   ```
   cd backend
   npm install
   # add REMOVE_BG_API_KEY to .env
   node server.js
   ```
2. Frontend:
   ```
   cd frontend
   npm install
   npm run dev
   ```
3. Open http://localhost:5173
