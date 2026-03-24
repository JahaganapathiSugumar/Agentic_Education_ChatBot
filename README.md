# 🤖 CONVERSATIONAL MULTI AGENT AI SYSTEM FOR REAL TIME CONTENT GENERATION AND TEACHING SUPPORT

A comprehensive AI-powered educational chatbot ecosystem offering dual interfaces—web and WhatsApp—for seamless educational content delivery and interactive learning experiences.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Testing & Evaluation](#testing--evaluation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository contains two complementary educational chatbot applications powered by Google's Generative AI. Both systems leverage advanced NLP, vector-based retrieval (RAG), and multi-modal content generation to create an engaging educational platform accessible via web and WhatsApp.

**Key Capabilities:**
- 🧠 AI-powered conversational learning
- 📚 Retrieval Augmented Generation (RAG) for document-based Q&A
- 🎥 Multi-media content generation (Videos, PDFs, Presentations, Worksheets)
- 🎤 Voice input/output via OpenAI Whisper & Google Text-to-Speech
- 📱 Cross-platform accessibility (Web + WhatsApp)
- 🔐 Secure authentication & authorization
- 📊 Accuracy evaluation & metrics tracking
- ☁️ Cloud-native architecture (Firebase, Google Cloud)

---

## 📦 Projects

### 1. **Agentic Education Web-Based ChatBot** 📱
**Location:** `Agentic_Education_Webbased_ChatBot/`

A modern web application with React + Tailwind CSS frontend and Flask backend, offering a ChatGPT-like interface for interactive learning.

**Target Users:**
- Students needing AI-powered tutoring
- Teachers creating educational content
- Institutions managing curriculum delivery

**Key Components:**
- React 18 + Vite frontend with real-time chat
- Flask RESTful backend with OAuth 2.0 authentication
- Google Classroom & Forms integration
- Firebase Realtime Database for user management
- RAG system with FAISS vector indexing
- Multi-format content generation

---

### 2. **Agentic Education WhatsApp-Based ChatBot** 💬
**Location:** `Agentic_Education_Whatsappbased_ChatBot/`

A WhatsApp Business API integrated chatbot for reaching students on their preferred communication platform, with advanced media generation capabilities.

**Target Users:**
- Students preferring WhatsApp communication
- Low-bandwidth educational delivery
- Mobile-first educational institutions

**Key Components:**
- WhatsApp Business API integration
- Flask webhook endpoint handler
- Speech-to-text via OpenAI Whisper
- Advanced media generation pipeline
- Google Classroom integration
- Accuracy evaluation framework

---

## ✨ Key Features

### 🌐 Web-Based ChatBot

#### Conversational AI
- Real-time chat with Google Generative AI
- Chat history management (50+ messages per session)
- Context-aware responses
- Multi-turn conversation support

#### Authentication & Authorization
- Firebase-based user authentication
- Role-based access control (Student/Teacher)
- Session management with OAuth 2.0
- Secure token handling

#### Content Generation
| Format | Capability | Use Case |
|--------|-----------|----------|
| **PDF** | Dynamic document generation with custom formatting | Study materials, assignments |
| **PowerPoint** | Structured presentations with images | Lecture materials, summaries |
| **Word Document** | Rich text documents with formatting | Worksheets, notes |
| **Images** | AI-generated visual content | Illustrations, diagrams |
| **Audio** | Text-to-speech generation | Learning audio materials |
| **Video** | Frame-by-frame video composition | Animated tutorials |

#### Google Cloud Integration
- 📚 **Google Classroom**: Create announcements, coursework, assignments
- 📋 **Google Forms**: Auto-generated MCQ assessment forms
- 📅 **Google Calendar**: Event scheduling
- 📁 **Google Drive**: File storage and sharing

#### Document Processing
- PDF extraction and analysis
- Multi-format document ingestion
- RAG-based Q&A on uploaded documents
- Vector similarity search for content retrieval

#### UI/UX Features
- Dark mode optimized interface
- ChatGPT-style message bubbles
- Markdown & syntax-highlighted code rendering
- Real-time typing indicators
- Mobile-responsive design
- File upload/download functionality
- Message copy-to-clipboard

---

### 💬 WhatsApp-Based ChatBot

#### WhatsApp Integration
- White-listed Business API access
- Template-based message delivery
- Interactive menu selection
- Two-way messaging
- Media sharing (Documents, Audio, Video, Images)

#### Speech Recognition
- OpenAI Whisper ASR (Automatic Speech Recognition)
- Audio transcription from WhatsApp voice messages
- Accurate multi-language support
- Real-time processing

#### Media Generation Pipeline
| Media Type | Technology | Capability |
|-----------|-----------|-----------|
| **Audio** | gTTS + pydub | Multi-language voice output |
| **Images** | PIL/Pillow + AI generation | Custom illustrations |
| **Video** | MoviePy + ModelScope | Animated educational content |
| **PDF** | FPDF2 | Formatted worksheets & assignments |
| **PowerPoint** | python-pptx | Structured presentations |
| **Excel** | openpyxl (preparation) | Data-driven content |

#### Educational Content Creation
- **MCQ Question Generation**: AI-powered multiple-choice quiz creation
- **Worksheet Generation**: Structured practice exercises
- **Video Scripts**: Curriculum-aligned content outlines
- **Voiceover Generation**: Synthesized narration for videos
- **Calendar Events**: Assignment deadlines, class schedules

#### Google Cloud Integration
- Google Classroom announcements
- Google Forms quiz distribution
- Google Calendar event management
- Google Drive file storage

#### Accuracy Evaluation
- Whisper transcription accuracy metrics
- RAG system response evaluation
- Factual correctness scoring
- Completeness assessment
- Word Error Rate (WER) calculation
- JSON result storage with timestamps

---

## 🏗️ Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                               │
│  ┌──────────────────────┐              ┌──────────────────────┐ │
│  │  Web Frontend        │              │  WhatsApp Business   │ │
│  │  (React + Tailwind)  │              │  API Integration     │ │
│  └──────────────────────┘              └──────────────────────┘ │
└──────────────┬──────────────────────────────────────┬───────────┘
               │                                      │
┌──────────────┴──────────────────────────────────────┴───────────┐
│                    Application Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Flask Backend (Python)                         │  │
│  │  • Request routing & middleware                          │  │
│  │  • Authentication (OAuth 2.0, Firebase)                  │  │
│  │  • Business logic & content generation                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────────────┬───────────┘
               │                                      │
┌──────────────┴──────────────────────────────────────┴───────────┐
│                    Processing Layer                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AI & Machine Learning                                   │  │
│  │  ├─ Google Generative AI (Gemini API)                   │  │
│  │  ├─ OpenAI Whisper (Speech-to-Text)                     │  │
│  │  ├─ HuggingFace Embeddings (Semantic Search)            │  │
│  │  └─ LangChain (LLM orchestration)                       │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  Media Generation                                        │  │
│  │  ├─ PDF Generation (FPDF2)                              │  │
│  │  ├─ Presentation Creation (python-pptx)                 │  │
│  │  ├─ Audio Synthesis (gTTS, pydub)                       │  │
│  │  ├─ Image Generation (PIL, AI models)                   │  │
│  │  └─ Video Composition (MoviePy)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────────────┬───────────┘
               │                                      │
┌──────────────┴──────────────────────────────────────┴───────────┐
│                    Data & Storage Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Vector Store                      │  Cloud Services     │  │
│  │  ├─ FAISS (Semantic Search)        │  ├─ Firebase        │  │
│  │  └─ Dynamic Indexing               │  ├─ Google APIs     │  │
│  │                                    │  └─ Cloud Storage    │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Web Chatbot:**
1. User sends message via React UI
2. Flask backend processes request
3. Message is augmented with context from FAISS vector index
4. Google Generative AI generates response
5. Response is rendered in React UI with markdown support

**WhatsApp Chatbot:**
1. User sends message/audio/media via WhatsApp
2. Webhook receives callback from WhatsApp API
3. Audio transcribed via Whisper (if applicable)
4. Message processed through AI pipeline
5. Content generated (media, documents, etc.)
6. Response sent back via WhatsApp API

---

## 🛠️ Technology Stack

### Backend
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | Flask 2.x | Web framework |
| **Authentication** | Firebase Admin SDK, OAuth 2.0 | Auth & identity |
| **LLM** | Google Generative AI (Gemini) | Core AI engine |
| **Embeddings** | Sentence-Transformers (HuggingFace) | Semantic search |
| **Vector Store** | FAISS | Document similarity search |
| **LLM Framework** | LangChain | LLM orchestration |

### Data & Cloud
| Service | Provider | Use Case |
|---------|----------|----------|
| **Database** | Firebase Realtime | User data, chat history |
| **Authentication** | Firebase Auth | User management |
| **Storage** | Google Drive | File storage |
| **APIs** | Google Cloud | Classroom, Forms, Calendar |

### Media Generation
| Format | Library | Technology |
|--------|---------|-----------|
| **PDF** | FPDF2 | PDF document creation |
| **PowerPoint** | python-pptx | Presentation generation |
| **Word** | python-docx | Document creation |
| **Audio** | gTTS, pydub | Text-to-speech & audio editing |
| **Images** | Pillow (PIL) | Image processing |
| **Video** | MoviePy | Video composition |
| **Speech-to-Text** | OpenAI Whisper | Audio transcription |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **React 18** | UI framework |
| **Vite 5** | Build tool |
| **Tailwind CSS 3** | Styling |
| **Marked.js** | Markdown rendering |
| **Highlight.js** | Code syntax highlighting |
| **Lucide React** | Icon library |

### External Integrations
- **WhatsApp Business API** - Messaging platform
- **Google Classroom API** - Education platform
- **Google Forms API** - Assessment creation
- **Google Calendar API** - Event management
- **Google Drive API** - File storage

---

## 📋 Prerequisites

### System Requirements
- **Python:** 3.9 or higher
- **Node.js:** 16 or higher (for web frontend)
- **npm:** 7 or higher
- **Git:** 2.0 or higher

### Required API Keys & Credentials
1. **Google Cloud Project** with:
   - Generative AI API enabled
   - Google Classroom API enabled
   - Google Forms API enabled
   - Google Calendar API enabled
   - Google Drive API enabled

2. **Firebase Project** with:
   - Realtime Database enabled
   - Authentication enabled

3. **WhatsApp Business API Access** (for WhatsApp chatbot only)
   - Business Account
   - Phone Number ID
   - Access Token
   - Verify Token

4. **OpenAI API Key** (for embeddings/Whisper models)

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/JahaganapathiSugumar/Agentic_Education_ChatBot.git
cd Agentic_Education_ChatBot
```

### 2. Web-Based ChatBot Setup

#### Backend
```bash
cd Agentic_Education_Webbased_ChatBot

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 3. WhatsApp-Based ChatBot Setup

```bash
cd Agentic_Education_Whatsappbased_ChatBot

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in project root:

#### Web Chatbot (`.env`)
```bash
# Firebase
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_WEB_API_KEY=your_web_api_key

# Google Cloud
GENAI_API_KEY=your_gemini_api_key
GOOGLE_FORMS_TARGET_FOLDER_ID=your_folder_id
CLASSROOM_COURSE_ID=your_course_id

# Flask
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

#### WhatsApp Chatbot (`.env`)
```bash
# WhatsApp Business API
VERIFY_TOKEN=your_verify_token
ACCESS_TOKEN=your_access_token
PHONE_NUMBER_ID=your_phone_id
TEMPLATE_NAME=your_template_name

# Google & Firebase
GENAI_API_KEY=your_gemini_api_key
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_PROJECT_ID=your_project_id

# Google Cloud
GOOGLE_FORMS_SERVICE_ACCOUNT_KEY_PATH=path/to/service_account.json
GOOGLE_FORMS_TARGET_FOLDER_ID=your_folder_id
CLASSROOM_COURSE_ID=your_course_id

# Flask
FLASK_ENV=production
```

### Service Accounts

Download service account JSON files and store securely:
- `credentials.json` - Google Classroom/Forms access
- `firebase_config.json` - Firebase configuration
- `sahayak-465916-8bf5ddce5515.json` - Alternative service account

---

## 🚀 Usage

### Web Chatbot

#### Start Backend Server
```bash
cd Agentic_Education_Webbased_ChatBot
python app.py
```

Backend runs on `http://localhost:5000`

#### Start Frontend Dev Server
```bash
cd Agentic_Education_Webbased_ChatBot/frontend
npm run dev
```

Frontend runs on `http://localhost:5173`

#### Access Application
Open browser to `http://localhost:5173`

#### Main Features
- **Chat Interface**: Ask questions, receive AI responses
- **File Upload**: Upload PDFs for RAG-based Q&A
- **Content Generation**: Generate worksheets, presentations, assessments
- **Assignment Management**: Create and manage assignments via Google Classroom
- **Form Creation**: Auto-generate MCQ assessments

---

### WhatsApp Chatbot

#### Start Backend Server
```bash
cd Agentic_Education_Whatsappbased_ChatBot
python agent_os.py
```

Server runs on `http://localhost:5000`

#### Webhook Configuration
Configure WhatsApp Business API webhook settings:
- **URL**: Your server URL + `/webhook`
- **Verify Token**: Must match `VERIFY_TOKEN` in `.env`

#### Usage Examples

**Text Message:**
- User sends: "Explain photosynthesis"
- Bot responds with AI-generated explanation

**Audio Message:**
- User sends voice message
- Bot transcribes via Whisper
- Responds with relevant content

**Content Generation:**
- User requests: "Create 10 MCQ questions on biology"
- Bot generates Google Form with questions
- User receives link via WhatsApp

---

## 📁 Project Structure

### Web-Based ChatBot
```
Agentic_Education_Webbased_ChatBot/
├── app.py                              # Main Flask application
├── responder.py                        # Response handler classes
├── requirements.txt                    # Python dependencies
├── .env                               # Environment configuration
│
├── frontend/                           # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx            # Chat history sidebar
│   │   │   ├── TopBar.jsx             # Header component
│   │   │   ├── MessageArea.jsx        # Message display
│   │   │   ├── MessageBubble.jsx      # Individual messages
│   │   │   ├── TypingIndicator.jsx    # Loading animation
│   │   │   ├── InputArea.jsx          # Message input
│   │   │   └── WelcomeScreen.jsx      # Initial welcome
│   │   ├── App.jsx                    # Main app component
│   │   ├── api.js                     # API service layer
│   │   ├── hooks.js                   # Custom React hooks
│   │   └── utils.js                   # Utility functions
│   ├── index.html                     # HTML template
│   ├── vite.config.js                # Build configuration
│   ├── tailwind.config.js            # Styling configuration
│   └── package.json                  # Frontend dependencies
│
├── templates/                         # HTML templates (backup)
│   └── index.html                    # Old HTML interface
│
├── data/                              # Application data
├── vector_index/                      # FAISS vector stores
│   └── dynamic_uploads/              # User document indexes
│
└── Documentation files:
    ├── README_REACT_FRONTEND.md      # Frontend guide
    ├── INSTALLATION_GUIDE.md         # Setup instructions
    ├── REACT_SETUP.md                # React configuration
    └── QUICK_REFERENCE.md            # Command reference
```

### WhatsApp-Based ChatBot
```
Agentic_Education_Whatsappbased_ChatBot/
├── agent_os.py                        # Main Flask application
├── requirements.txt                   # Python dependencies
├── .env                              # Environment configuration
│
├── Audio_creation.py                 # Audio generation utilities
├── image_creation.py                 # Image generation utilities
├── docu_creation.py                  # Document generation utilities
├── sample.py / sample2.py            # Testing samples
│
├── accuracy_evaluation/              # Accuracy testing framework
│   ├── evaluator.py                  # Accuracy evaluator
│   ├── metrics_calculator.py         # Metrics computation
│   ├── run_accuracy_test.py          # Test runner
│   ├── test_cases.py                 # Test case definitions
│   │
│   ├── test_data/                    # Test data
│   │   ├── audio_samples/            # Audio test files
│   │   ├── test_documents/           # Document test files
│   │   └── test_images/              # Image test files
│   │
│   └── results/                      # Evaluation results
│       ├── simulated_evaluation_*.json
│       └── summary_report_*.json
│
├── data/                             # Application data
└── vector_index/                     # FAISS vector stores
    ├── dynamic_uploads/              # Dynamic document index
    ├── os_docs/                      # Operating system docs
    └── subject_docs/                 # Subject material index
```

---

## 🎨 Advanced Features

### Retrieval Augmented Generation (RAG)

Both chatbots use FAISS-based semantic search for document retrieval:

1. **Indexing**: Upload documents → Extract text → Generate embeddings → Store in FAISS
2. **Retrieval**: Query → Generate query embeddings → Find similar documents
3. **Generation**: Combine context + query → Send to Generative AI → Get response

**Benefits:**
- Accurate answers based on uploaded documents
- Reduced hallucinations
- Knowledge-grounded responses

### Multi-Modal Content Generation

**Web Chatbot can generate:**
- PDF worksheets with custom formatting
- PowerPoint presentations with text + images
- Word documents with structured content
- Text-to-speech audio narration
- Video compositions from image sequences

**WhatsApp Chatbot additionally generates:**
- Voiceover-enabled content for audio delivery
- Video scripts with timing
- Google Forms assessments
- Calendar event invitations

### Authentication & Authorization

**Web Chatbot:**
- Firebase Authentication (Email/Password, OAuth)
- Role-based access control (Student/Teacher)
- Session management
- OAuth 2.0 token refresh for Google APIs

**WhatsApp Chatbot:**
- Phone number verification via WhatsApp
- Allowlist for trusted numbers
- Webhook signature verification

---

## 🧪 Testing & Evaluation

### Accuracy Evaluation Framework (WhatsApp Chatbot)

Located in `Agentic_Education_Whatsappbased_ChatBot/accuracy_evaluation/`

#### Running Tests
```bash
cd accuracy_evaluation
python run_accuracy_test.py
```

#### Metrics Calculated
- **Whisper Accuracy**: Word Error Rate (WER), Transcription accuracy
- **RAG Accuracy**: Factual correctness, Completeness score
- **Response Quality**: Relevance, Clarity, Informativeness

#### Results Storage
Results saved to `accuracy_evaluation/results/`:
- `simulated_evaluation_TIMESTAMP.json` - Detailed results
- `summary_report_TIMESTAMP.json` - High-level metrics

---
