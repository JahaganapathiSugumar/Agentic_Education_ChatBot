# create_test_audio.py
from gtts import gTTS
import os

def create_test_audio_files():
    """Generate test audio files using TTS"""
    audio_samples = [
        {
            'filename': 'clear_lecture_30s.wav',
            'text': 'Today we will discuss machine learning algorithms. Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.',
            'description': 'Clean educational audio about ML'
        },
        {
            'filename': 'student_question.wav', 
            'text': 'Can you explain the difference between supervised and unsupervised learning? In supervised learning, we have labeled data, while in unsupervised learning, we work with unlabeled data to find patterns.',
            'description': 'Student question about learning types'
        },
        {
            'filename': 'physics_lecture.wav',
            'text': 'Newton\'s first law states that an object at rest stays at rest, and an object in motion stays in motion with the same velocity, unless acted upon by an external force. This is also known as the law of inertia.',
            'description': 'Physics educational content'
        }
    ]
    
    os.makedirs('accuracy_evaluation/test_data/audio_samples', exist_ok=True)
    
    for sample in audio_samples:
        tts = gTTS(text=sample['text'], lang='en', slow=False)
        filepath = f"accuracy_evaluation/test_data/audio_samples/{sample['filename']}"
        tts.save(filepath)
        print(f"Created: {filepath}")

if __name__ == "__main__":
    create_test_audio_files()