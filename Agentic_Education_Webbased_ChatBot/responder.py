# responder.py
from abc import ABC, abstractmethod
import base64

class Responder(ABC):
    """Abstract interface for sending messages (WhatsApp or Web)."""
    @abstractmethod
    def text(self, to, message):
        pass

    @abstractmethod
    def menu(self, to, text, options):
        pass

    @abstractmethod
    def document(self, to, file_bytes, filename):
        pass

    @abstractmethod
    def audio(self, to, audio_bytes, filename):
        pass

    @abstractmethod
    def video(self, to, video_bytes, filename):
        pass

    @abstractmethod
    def interactive(self, to, text, buttons):
        pass


class WebResponder(Responder):
    """Collects actions to return as JSON to the web frontend."""
    def __init__(self):
        self.actions = []

    def text(self, to, message):
        self.actions.append({"type": "text", "content": message})

    def menu(self, to, text, options):
        self.actions.append({"type": "menu", "text": text, "options": options})

    def document(self, to, file_bytes, filename):
        self.actions.append({
            "type": "document",
            "filename": filename,
            "data": base64.b64encode(file_bytes).decode('utf-8')
        })

    def audio(self, to, audio_bytes, filename):
        self.actions.append({
            "type": "audio",
            "filename": filename,
            "data": base64.b64encode(audio_bytes).decode('utf-8')
        })

    def video(self, to, video_bytes, filename):
        self.actions.append({
            "type": "video",
            "filename": filename,
            "data": base64.b64encode(video_bytes).decode('utf-8')
        })

    def interactive(self, to, text, buttons):
        self.actions.append({"type": "interactive", "text": text, "buttons": buttons})