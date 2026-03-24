# 🎨 Premium UI Visual Guide

## Color Palette

### Primary Colors
```
🟢 Primary:       #10a37f (Emerald Green)
🟢 Hover:         #1db584 (Bright Emerald)
🟢 Light:         #34d399 (Light Emerald)
🟢 Dark:          #059669 (Dark Emerald)
```

### Background Colors
```
⬛ Dark:          #040404 (True Black)
⬛ Darker:        #000000 (Pure Black)
⬛ Light:         #1a1a1a (Dark Gray)
```

### Surface Colors
```
🟦 Surface Dark:  #212121 (Elevated Dark)
🟦 Surface Darker:#0d0d0d (Deep Dark)
🟦 Surface Light: #2d2d2d (Lighter Dark)
```

### Text Colors
```
📝 Primary:       #ececec (Main Text)
📝 Secondary:     #b4b4b4 (Muted Text)
📝 Tertiary:      #8b8b8b (Disabled Text)
📝 Quarternary:   #5a5a5a (Hint Text)
```

### Border Colors
```
|  Dark:          #3a3a3a (Standard Border)
|  Darker:        #262626 (Subtle Border)
|  Light:         #4a4a4a (Bright Border)
```

### Semantic Colors
```
✅ Success:       #10b981 (Green)
⚠️  Warning:       #f59e0b (Amber)
❌ Error:         #ef4444 (Red)
ℹ️  Info:          #3b82f6 (Blue)
```

---

## Component Styling

### 1️⃣ Message Bubbles

#### User Message
```
┌──────────────────────┐
│ Your message here     │  ← Primary Green (#10a37f)
│ White text           │  ← Maximum 2xl width
└──────────────────────┘  ← 32px rounded corners
```
**Features:**
- Green background (gradient)
- White text
- 32px border radius
- 16px padding (vertical + horizontal)
- Max width 2xl
- Shadow on hover
- Fade-in animation

#### Bot Message
```
┌──────────────────────┐
│ AI response          │  ← Dark surface (#212121)
│ Gray text            │  ← Border for definition
│ With code support    │  ← Border: #3a3a3a
│ [Copy]              │  ← Action button
└──────────────────────┘  ← 32px rounded corners
```
**Features:**
- Dark surface background
- Border for depth
- Copy button appears on hover
- Markdown rendering
- Syntax highlighted code blocks
- Smooth animations

---

### 2️⃣ Input Area

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 📎  [    Type message... Shift+Enter     ] 🎤 ▶️ ┃  ← Auto-resize
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Powered by Sahayak AI — Your Intelligent Teaching Assistant
```

**Features:**
- Rounded input box (32px corners)
- File attach button (📎)
- Voice record button (🎤)
- Send button (▶️)
- Auto-resizing textarea
- Gradient background on parent
- Focus ring highlight
- Hover shadow effect
- Safe area padding

---

### 3️⃣ Sidebar

```
╔═══════════════════════╗
║ ✨ Sahayak            ║  ← Header with icon
║    AI Assistant       ║
╠═══════════════════════╣
║ ➕ New Chat           ║  ← Gradient button
╠═══════════════════════╣
║ 📝 Chat 1             ║  ← Current selected
║ 📝 Chat 2    ⏮         ║  ← Delete on hover
║ 📝 Chat 3             ║
╠═══════════════════════╣
║ 🔖 Bookmarks          ║  ← Footer buttons
║ 🚪 Log out            ║
╚═══════════════════════╝
```

**Features:**
- Emoji icons for visual interest
- Gradient "New Chat" button
- History section with label
- Smooth chat selection
- Delete button appears on hover
- Log out button with error styling
- Rounded corners (12-16px)
- Mobile overlay with backdrop blur

---

### 4️⃣ Top Bar

```
┌──────────────────────────────────────────────────────────┐
│ ☰  Chat Title                    ⚡ Sahayak Pro  👤 User  │
└──────────────────────────────────────────────────────────┘
```

**Features:**
- Mobile menu toggle (☰)
- Active chat title
- Model selector badge
- User profile chip with gradient
- Gradient background
- Backdrop blur
- Responsive layout

---

### 5️⃣ Welcome Screen

```
                      ✨
                   (Glowing)

      Welcome to Sahayak AI
   
   Your intelligent teaching assistant
            is ready to help


┌─────────────┬─────────────┬─────────────┐
│  ❓ Ask    │ 📚 Create  │ 🎨 Create   │
│  Question  │  Worksheet │  PPT        │
└─────────────┴─────────────┴─────────────┘
┌─────────────┬─────────────┬─────────────┐
│ 📤 Upload  │ 📁 View    │ 🎙️ Podcast │
│ Materials  │ Files      │ from Image  │
└─────────────┴─────────────┴─────────────┘
┌─────────────────────────────────────────┐
│         📸 Summary from Image            │
└─────────────────────────────────────────┘

💡 Tip: Use Shift+Enter for multi-line messages
```

**Features:**
- Large animated icon with glow
- Gradient text heading
- 7 action buttons with unique colors
- Icons + titles for each button
- Shine effect on hover
- Scale animation on click
- Helpful tip at bottom

---

### 6️⃣ Typing Indicator

```
🤖  ● ● ●  Thinking...
     ↑ ↑ ↑ (bouncing with delay)
```

**Features:**
- AI avatar emoji (🤖)
- 3 bouncing gradient dots
- Smooth animation
- Fade-in effect

---

## Animation Timings

| Animation | Duration | Easing |
|-----------|----------|--------|
| fade-in | 400ms | cubic-bezier(0.4, 0, 0.2, 1) |
| slide-in | 400ms | cubic-bezier(0.4, 0, 0.2, 1) |
| bounce-dots | 1400ms | cubic-bezier(0.4, 0, 0.6, 1) |
| pulse | 2000ms | cubic-bezier(0.4, 0, 0.6, 1) |
| fast transition | 150ms | linear |
| base transition | 200ms | linear |
| slow transition | 300ms | linear |

---

## Interactive States

### Buttons

#### Primary Button (Send, New Chat)
```
Default:   🟢 Primary Green (#10a37f)
Hover:     🟢 Bright Green (#1db584) + Shadow
Active:    🟢 Scale 95% + Shadow
Disabled:  ⚫ 50% opacity
```

#### Secondary Button
```
Default:   Dark surface + border
Hover:     Lighter surface
Active:    Color highlight
```

#### Ghost Button (Icons)
```
Default:   Transparent + muted text
Hover:     Light surface + bright text
Active:    Primary text
```

---

## Responsive Behavior

### Mobile (< 480px)
- Sidebar slides in from left
- Full-width input area
- Smaller padding
- Stacked layout
- Simplified header

### Tablet (768px)
- Sidebar visible on desktop
- Two-column chat grid (where applicable)
- Optimized spacing

### Desktop (1024px+)
- Full sidebar visible
- Centered content max-width (4xl)
- Full feature set

---

## Accessibility Features

✅ **Color Contrast**
- WCAG AA compliant
- Text renders clearly on all backgrounds
- Focus rings are visible

✅ **Keyboard Navigation**
- Tab through all interactive elements
- Enter to activate buttons
- Escape to close modals

✅ **Reduced Motion**
- `prefers-reduced-motion` support
- Animations disabled for users who prefer
- All functionality preserved

✅ **Semantic HTML**
- Proper button semantics
- ARIA labels where needed
- Keyboard shortcuts documented

---

## Shadow System

```
xs:   0 1px 2px rgba(0, 0, 0, 0.05)
sm:   0 1px 3px rgba(0, 0, 0, 0.1)
md:   0 4px 6px rgba(0, 0, 0, 0.1)
lg:   0 10px 15px rgba(0, 0, 0, 0.1)
xl:   0 20px 25px rgba(0, 0, 0, 0.1)
2xl:  0 25px 50px rgba(0, 0, 0, 0.25)
inner: inset 0 2px 4px rgba(0, 0, 0, 0.06)
```

---

## Border Radius System

```
xs:   4px   (small buttons, subtle curves)
sm:   6px   (badges, chips)
md:   8px   (default buttons)
lg:   12px  (cards, larger containers)
xl:   16px  (major components)
2xl:  32px  (message bubbles, premium buttons)
3xl:  48px  (large icons, hero elements)
full: 50%   (perfect circles, avatars)
```

---

## Spacing System (8px Grid)

```
xs:   4px   (2pt)
sm:   8px   (1rem)
md:   12px  (1.5rem)
lg:   16px  (2rem)
xl:   24px  (3rem)
2xl:  32px  (4rem)
```

---

## Font System

| Size | Line Height | Usage |
|------|-------------|-------|
| 12px | 16px | Captions, metadata |
| 13px | 18px | Small labels |
| 15px | 24px | Body text |
| 17px | 26px | Larger text |
| 20px | 28px | Subheadings |
| 24px | 32px | Section titles |
| 32px | 40px | Main headings |

---

## Gradient System

### Primary Gradient
```
from-primary (#10a37f) → to-primary-light (#34d399)
(Left to Right or Top to Bottom)
```

### Action Buttons (WelcomeScreen)
```
Blue:     from-blue-500 → to-blue-600
Purple:   from-purple-500 → to-purple-600
Pink:     from-pink-500 → to-pink-600
Green:    from-green-500 → to-green-600
Yellow:   from-yellow-500 → to-yellow-600
Orange:   from-orange-500 → to-orange-600
Indigo:   from-indigo-500 → to-indigo-600
```

---

## Performance Optimizations

⚡ **CSS Optimizations:**
- Minimal repaints via transform use
- Hardware acceleration for animations
- Efficient selectors
- No layout thrashing
- Optimized shadows

⚡ **JavaScript Optimizations:**
- Event delegation
- Debounced callbacks
- Memoized components
- Lazy loading images

⚡ **Accessibility Speed:**
- Keyboard-only navigation support
- Reduced motion respected
- No JavaScript required for basics

---

## Usage Examples

### Apply Premium Look to New Component
```jsx
<button className="px-lg py-md bg-primary hover:bg-primary-hover text-white rounded-lg transition-all duration-base shadow-md hover:shadow-lg active:scale-95">
  Click Me
</button>
```

### Create Premium Card
```jsx
<div className="rounded-xl bg-surface-dark border border-border-dark shadow-sm hover:shadow-md p-lg transition-all duration-base">
  Your content
</div>
```

### Add Premium Input
```jsx
<input 
  className="w-full px-lg py-md rounded-lg bg-surface-dark border border-border-dark text-text-primary placeholder-text-tertiary focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary focus:ring-opacity-20 transition-all duration-fast"
  placeholder="Type here..."
/>
```

---

**Your UI is now premium-grade, matching ChatGPT and Claude in visual appeal!** ✨
