# 🚀 React + Tailwind CSS Frontend - Installation Guide

## Complete Setup Instructions

Your new React frontend with Tailwind CSS has been created! Follow these steps to get started.

---

## 📋 Prerequisites

- **Node.js** 16 or higher
- **npm** 7 or higher
- **Python** & **Flask** (for backend)

Check your versions:
```bash
node --version
npm --version
```

---

## ⚙️ Installation Steps

### Step 1: Navigate to Frontend Directory
```bash
cd f:\project\web\frontend
```

### Step 2: Install Dependencies
```bash
npm install
```

This installs all required packages:
- React 18
- Vite (build tool)
- Tailwind CSS
- Marked (markdown)
- Highlight.js (code highlighting)
- And more...

### Step 3: Start Development Server
```bash
npm run dev
```

Output:
```
  VITE v5.0.8  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

Open `http://localhost:5173` in your browser.

---

## 🛠️ Development Workflow

### Live Reload
Changes to React files automatically reload the browser (HMR).

### Build for Production
```bash
npm run build
```

Creates optimized files in `../static/`:
- `index.html`
- `css/main.*.css`
- `js/main.*.js`
- Other assets

### Preview Production Build
```bash
npm run preview
```

---

## 🔗 Connecting to Flask Backend

### Option 1: Development with Proxy (Recommended)

The Vite dev server proxies API calls to Flask:

1. **Start Flask backend** on port 5000:
```bash
cd f:\project\web
python app.py
```

2. **Start React frontend** on port 5173:
```bash
cd f:\project\web\frontend
npm run dev
```

3. **The proxy is configured in `vite.config.js`:**
```js
server: {
  proxy: {
    '/api': 'http://localhost:5000',
    '/logout': 'http://localhost:5000'
  }
}
```

Now API calls from React go to Flask automatically!

### Option 2: Production

1. Build the React app:
```bash
npm run build
```

2. Flask serves static files:
```python
from flask import send_from_directory, render_template

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# SPA fallback
@app.errorhandler(404)
def not_found(e):
    return send_from_directory('static', 'index.html')
```

3. Run Flask:
```bash
python app.py
```

Visit `http://localhost:5000`

---

## 📂 Project Structure

```
frontend/
├── src/
│   ├── components/              # React components
│   │   ├── Sidebar.jsx          # Chat history sidebar
│   │   ├── TopBar.jsx           # Header bar
│   │   ├── MessageArea.jsx       # Message container
│   │   ├── MessageBubble.jsx     # Individual message
│   │   ├── TypingIndicator.jsx   # Loading animation
│   │   ├── InputArea.jsx         # Message input
│   │   └── WelcomeScreen.jsx     # Welcome screen
│   │
│   ├── App.jsx                  # Main component (state, API calls)
│   ├── main.jsx                 # React entry point
│   ├── index.css                # Tailwind + global styles
│   │
│   ├── api.js                   # API service (fetch wrapper)
│   ├── hooks.js                 # Custom React hooks
│   ├── utils.js                 # Utility functions
│   
├── index.html                   # HTML template
├── vite.config.js              # Vite configuration
├── tailwind.config.js          # Tailwind customization
├── postcss.config.js           # PostCSS configuration
├── package.json                # Dependencies & scripts
├── .gitignore                  # Git ignore
└── README.md                   # Frontend docs
```

---

## 🎨 Customization

### Colors
Edit `tailwind.config.js`:
```js
colors: {
  primary: '#10a37f',           // Green accent
  'primary-hover': '#0d8c6d',
  'primary-light': '#1ab89f',
  'bg-dark': '#0f172a',         // Main background
  'bg-darker': '#0a1628',
  'surface-dark': '#1a1f3a',    // Component background
  'border-dark': '#2a3f5f',     // Borders
  'text-secondary': '#a0a0a0',
  'text-tertiary': '#808080',
}
```

### Spacing
Edit `tailwind.config.js`:
```js
spacing: {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
}
```

### Adding New Components
1. Create `src/components/MyComponent.jsx`
2. Export as default:
```jsx
export default function MyComponent() {
  return <div className="...">Content</div>
}
```
3. Import in `App.jsx`

### Using Tailwind Classes
```jsx
<div className="flex gap-4 p-6 bg-surface-dark rounded-lg shadow-md hover:shadow-lg">
  <button className="px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-hover">
    Click me
  </button>
</div>
```

---

## 📡 API Endpoints

The frontend expects these Flask endpoints:

```
GET    /api/chats                          # List all chats
POST   /api/chats                          # Create new chat
GET    /api/chats/{id}                     # Get specific chat
PATCH  /api/chats/{id}                     # Rename chat
DELETE /api/chats/{id}                     # Delete chat
POST   /api/chats/{id}/messages            # Send message
POST   /api/chats/{id}/upload              # Upload file
DELETE /api/chats                          # Clear all
```

---

## 🧪 Testing

### Test Components in Browser
The dev server has hot reload. Just edit a component and save!

### Test API Calls
Add debug logs to `src/api.js`:
```js
const fetchAPI = async (endpoint, options = {}) => {
  console.log('API Call:', endpoint, options)
  // ...
}
```

### Browser DevTools
1. Open DevTools (F12)
2. Network tab: Monitor API calls
3. Console: View logs
4. React tab: Inspect components

---

## 🐛 Troubleshooting

### Port 5173 Already in Use
```bash
npm run dev -- --port 3000
```

### Module Not Found
```bash
# Clear cache and reinstall
rm -r node_modules package-lock.json
npm install
```

### Tailwind CSS Not Working
1. Check `content` in `tailwind.config.js`
2. Ensure files are being watched:
   ```js
   content: [
     "./index.html",
     "./src/**/*.{js,jsx}",
   ]
   ```
3. Rebuild:
   ```bash
   npm run build
   ```

### API Calls Failing
1. Check Flask is running on port 5000
2. Check proxy in `vite.config.js`
3. Check Flask CORS if needed:
   ```python
   from flask_cors import CORS
   CORS(app)
   ```

---

## 📦 Dependencies Explained

| Package | Purpose |
|---------|---------|
| **react** | UI library |
| **react-dom** | React rendering |
| **vite** | Fast build tool |
| **tailwindcss** | Utility CSS framework |
| **marked** | Markdown parser |
| **highlight.js** | Code syntax highlighting |
| **lucide-react** | Icon library |

---

## 🚢 Deployment

### Build for Production
```bash
npm run build
```

### Upload to Server
1. Built files are in `../static/`
2. Upload to web server
3. Update Flask to serve from `static/`

### Using Docker
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY frontend .
RUN npm install && npm run build

FROM python:3.10
COPY app.py .
COPY --from=build /app/static ./static
CMD ["python", "app.py"]
```

---

## 📚 Resources

- **React**: https://react.dev
- **Tailwind CSS**: https://tailwindcss.com
- **Vite**: https://vitejs.dev
- **Marked**: https://marked.js.org
- **Highlight.js**: https://highlightjs.org

---

## 🎯 Next Steps

1. ✅ Install dependencies: `npm install`
2. ✅ Start dev server: `npm run dev`
3. ✅ Open browser: http://localhost:5173
4. ⏳ Customize colors in `tailwind.config.js`
5. ⏳ Add more components in `src/components/`
6. ⏳ Build for production: `npm run build`
7. ⏳ Deploy to web server

---

## 💡 Tips

- Use React DevTools browser extension for debugging
- Check Network tab in DevTools to debug API calls
- Use `console.log()` to debug state and props
- Hot reload speeds up development significantly
- Build regularly to catch compilation errors

---

## 📝 Files Reference

📍 **Frontend Source**:
- `f:\project\web\frontend\`

📍 **Built Files** (after `npm run build`):
- `f:\project\web\static\`

📍 **Flask Backend**:
- `f:\project\web\app.py`

📍 **Old HTML UI** (backed up):
- `f:\project\web\templates\index_old_backup.html`

---

**You're all set! Happy coding! 🎉**
