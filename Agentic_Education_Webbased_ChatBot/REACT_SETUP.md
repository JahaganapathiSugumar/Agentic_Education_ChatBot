# Sahayak AI - Frontend Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Development Mode
```bash
npm run dev
```

This starts a Vite dev server at `http://localhost:5173` with hot reload.

### 3. Build for Production
```bash
npm run build
```

This creates optimized files in `../static/` directory.

### 4. Connect to Flask Backend

The frontend is configured to proxy API requests to Flask:
- Dev: `http://localhost:5000` (configured in vite.config.js)
- Production: Flask serves the built static files

## Project Structure

```
frontend/
├── src/
│   ├── components/         # React components
│   ├── App.jsx            # Main app component
│   ├── main.jsx           # React entry point
│   └── index.css          # Tailwind + global styles
├── index.html             # HTML template
├── vite.config.js         # Vite build config
├── tailwind.config.js     # Tailwind customization
└── package.json           # Dependencies
```

## Component Overview

- **Sidebar.jsx** - Chat history, new chat, user profile
- **TopBar.jsx** - Header with model selector
- **MessageArea.jsx** - Chat display area
- **MessageBubble.jsx** - Individual message with markdown
- **TypingIndicator.jsx** - AI thinking animation
- **InputArea.jsx** - Message input, file/voice buttons
- **WelcomeScreen.jsx** - New chat greeting with quick buttons

## Key Features

✅ Fully responsive (mobile, tablet, desktop)
✅ Dark theme optimized
✅ Markdown message rendering
✅ Code syntax highlighting
✅ Smooth animations
✅ Modern Tailwind CSS design
✅ Reusable components
✅ API integration ready

## Styling Guide

All styling uses Tailwind CSS utilities:

```jsx
<button className="px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-hover">
  Click me
</button>
```

Custom colors available in `tailwind.config.js`:
- `primary` - #10a37f
- `surface-dark` - #1a1f3a
- `border-dark` - #2a3f5f
- etc.

## API Endpoints

The frontend expects these Flask endpoints:

```
GET  /api/chats
POST /api/chats
GET  /api/chats/{id}
POST /api/chats/{id}/messages
DELETE /api/chats/{id}
PATCH /api/chats/{id}
```

## Environment Variables

None required - the proxy is configured in `vite.config.js`

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Android)

## Troubleshooting

**Port already in use:**
```bash
npm run dev -- --host 0.0.0.0 --port 3000
```

**Module not found errors:**
```bash
rm node_modules package-lock.json
npm install
```

**CSS not loading:**
Make sure Tailwind is processing the content files. Check `tailwind.config.js` content paths.

## Production Deployment

1. Build the frontend:
```bash
npm run build
```

2. Files go to `../static/`

3. Flask serves them automatically if configured

4. Update Flask to serve the index.html for SPA routing if needed

## Next Steps

- Add more components as needed
- Integrate with backend API endpoints
- Add state management (Zustand/Redux) if app grows
- Configure hot reload for development
- Add PWA support for mobile
- Setup CI/CD for automatic builds

For more info, see the main README.md
