# Standardized test cases for manual evaluation
WHISPER_TEST_CASES = [
    {
        'id': 'audio_001',
        'audio_file': 'test_data/audio_samples/clear_lecture_30s.wav',
        'expected_text': 'Today we will discuss machine learning algorithms including supervised and unsupervised learning approaches.',
        'description': 'Clear educational audio with technical terms',
        'difficulty': 'medium'
    },
    {
        'id': 'audio_002', 
        'audio_file': 'test_data/audio_samples/student_question.wav',
        'expected_text': 'Can you explain the difference between classification and regression in machine learning?',
        'description': 'Student question with educational context',
        'difficulty': 'easy'
    }
]

RAG_TEST_CASES = [
    {
        'id': 'rag_001',
        'document': 'test_data/test_documents/physics_chapter.pdf',
        'questions': [
            {
                'question': 'What is Newton\'s first law of motion?',
                'expected_answer': 'An object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.',
                'expected_sources': ['physics_chapter.pdf']
            }
        ]
    }
]

CONTENT_GENERATION_TEST_CASES = [
    {
        'id': 'worksheet_001',
        'topic': 'Photosynthesis',
        'expected_elements': ['chlorophyll', 'light energy', 'carbon dioxide', 'oxygen', 'glucose'],
        'num_questions': 5,
        'question_types': ['MCQ', 'short answer']
    }
]

def get_all_test_cases():
    """Return all test cases organized by module"""
    return {
        'whisper_stt': WHISPER_TEST_CASES,
        'rag_system': RAG_TEST_CASES,
        'content_generation': CONTENT_GENERATION_TEST_CASES
    }