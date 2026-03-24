# ⚡ React Frontend - Quick Reference Card

## 🚀 Commands Cheat Sheet

```bash
# Navigate to frontend
cd f:\project\web\frontend

# Install dependencies (first time)
npm install

# Start development server
npm run dev
# → http://localhost:5173

# Build for production
npm run build
# → Creates files in ../static/

# Preview production build
npm run preview

# Update dependencies
npm update

# Check for outdated packages
npm outdated

# Clean install (if stuck)
npm clean-install
```

---

## 📁 Quick File Locations

```
frontend/                    # React app root
├── src/
│   ├── components/         # React components (7 files)
│   ├── App.jsx            # Main component (state & logic)
│   ├── api.js             # API service
│   ├── hooks.js           # Custom hooks
│   └── utils.js           # Utility functions
├── index.html             # HTML template
├── vite.config.js         # Build config (modifiable)
├── tailwind.config.js     # CSS config (modifiable)
└── package.json           # Dependencies
```

---

## 🧩 Component Import Reference

```javascript
// In App.jsx or other components
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import MessageArea from './components/MessageArea'
import InputArea from './components/InputArea'
```

---

## 🎨 Common Tailwind Classes

```jsx
// Layout
<div className="flex gap-4 p-6">              {/* Flexbox + spacing */}
<div className="grid grid-cols-2 gap-4">      {/* Grid */}
<div className="flex-1">                      {/* Take remaining space */}

// Colors
className="text-text-primary"        {/* Text color */}
className="bg-surface-dark"          {/* Background */}
className="border border-border-dark" {/* Border */}
className="hover:bg-primary"         {/* On hover */}

// Sizing
className="w-8 h-8"                  {/* Width & height */}
className="px-4 py-2"                {/* Padding */}
className="rounded-md"               {/* Border radius */}

// Positioning
className="absolute top-0 left-0"    {/* Position */}
className="sticky"                   {/* Sticky */}
className="fixed"                    {/* Fixed position */}

// Effects
className="shadow-md"                {/* Shadow */}
className="opacity-50"               {/* Opacity */}
className="transition-all duration-200" {/* Transitions */}
```

---

## 🔌 API Usage Examples

```javascript
// In components
import { chatAPI } from '../api'

// Get all chats
const chats = await chatAPI.getChats()

// Create new chat
const newChat = await chatAPI.createChat('My Chat Title')

// Send message
const response = await chatAPI.sendMessage(chatId, 'Hello!')

// Delete chat
await chatAPI.deleteChat(chatId)

// Rename chat
await chatAPI.renameChat(chatId, 'New Title')

// Upload file
await chatAPI.uploadFile(chatId, fileObject)
```

---

## 🪝 Custom Hooks Usage

```javascript
import { useLocalStorage, useDebounce, useFetch, useToggle } from '../hooks'

// Local storage
const [savedData, setSavedData] = useLocalStorage('key', defaultValue)

// Debounced value
const debouncedSearchTerm = useDebounce(searchTerm, 500)

// Fetch data
const { data, loading, error } = useFetch('/api/chats')

// Toggle state
const [isOpen, toggleOpen] = useToggle(false)
```

---

## 📝 Component Template

```jsx
import React, { useState } from 'react'

export default function MyComponent({ props }) {
  const [state, setState] = useState(initialValue)

  const handleClick = () => {
    setState(newValue)
  }

  return (
    <div className="flex gap-4 p-4">
      <button 
        onClick={handleClick}
        className="px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-hover"
      >
        Click Me
      </button>
    </div>
  )
}
```

---

## 🎯 State Management Pattern

```jsx
// App.jsx - Hold state here
const [messages, setMessages] = useState([])
const [isLoading, setIsLoading] = useState(false)

// Pass to children
<MessageArea messages={messages} isLoading={isLoading} />

// Update state
const addMessage = (msg) => {
  setMessages([...messages, msg])
}
```

---

## 🚨 Common Fixes

| Problem | Solution |
|---------|----------|
| Build fails | `npm install`, check syntax |
| Port in use | `npm run dev -- --port 3000` |
| Styles not updating | Clear cache: Ctrl+Shift+R |
| Hot reload broken | Reload page manually (F5) |
| API 404 errors | Check endpoint spelling, Flask running |

---

## 🎨 Color Reference

```javascript
// Primary actions (buttons, highlights)
--color-primary: '#10a37f'

// Backgrounds
--bg-dark: '#0f172a'        // Main background
--surface-dark: '#1a1f3a'   // Card/component background

// Text
--text-primary: '#ececec'     // Main text
--text-secondary: '#a0a0a0'   // Dimmed text
--text-tertiary: '#808080'    // Very dim text

// Borders & separators
--border-dark: '#2a3f5f'
```

---

## 📱 Responsive Breakpoints

```javascript
// Mobile first approach
sm:  640px   {/* Small devices */}
md:  768px   {/* Tablets */}
lg: 1024px   {/* Desktops */}
```

Example:
```jsx
<div className="hidden md:flex">   {/* Hide on mobile, show on tablet+ */}
```

---

## 🔗 External Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| React Docs | https://react.dev | Official React documentation |
| Tailwind CSS | https://tailwindcss.com | CSS framework docs |
| Vite | https://vitejs.dev | Build tool docs |
| Marked.js | https://marked.js.org | Markdown parser |
| Highlight.js | https://highlightjs.org | Syntax highlighting |

---

## ✅ Pre-Commit Checklist

Before committing code:
- [ ] Components are properly exported
- [ ] No console.log() left in production code
- [ ] Tailwind classes are valid
- [ ] No unused imports
- [ ] API calls have error handling
- [ ] Responsive design tested
- [ ] Build succeeds: `npm run build`

---

## 📦 Useful npm Packages (Optional)

If you want to extend the app:

```bash
# State management
npm install zustand

# Routing (multi-page)
npm install react-router-dom

# Form handling
npm install react-hook-form

# HTTP client
npm install axios

# Date utilities
npm install date-fns

# Notifications/Toast
npm install react-hot-toast
```

---

## 🚀 Deploy Quick Steps

1. **Build**:
   ```bash
   npm run build
   ```

2. **Check output**:
   ```bash
   ls ../static/
   ```

3. **Serve with Flask** (in `app.py`):
   ```python
   from flask import send_from_directory
   
   @app.route('/')
   def index():
       return send_from_directory('static', 'index.html')
   ```

4. **Run Flask**:
   ```bash
   python app.py
   ```

5. **Visit**: http://localhost:5000

---

## 💡 Pro Tips

1. **Use React DevTools** - Install browser extension for better debugging
2. **Check Network Tab** - DevTools → Network to debug API calls
3. **Console Logs** - Use `console.log()` for quick debugging
4. **Hot Reload** - File saves auto-reload, no manual refresh needed
5. **Class Names** - Keep them short, use Tailwind utilities
6. **Comments** - Document complex logic for future you
7. **Components** - Keep them small, focused, single responsibility

---

**Keep this handy while developing! 📋**

*For detailed info, see INSTALLATION_GUIDE.md*
