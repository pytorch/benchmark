from enum import Enum

# Enum class for each Domain for the model and the respective tasks
# that is available in the domain.
class COMPUTER_VISION(Enum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    GENERATION = "generation"
    PATTERN_RECOGNITION = "pattern recognition"
    VIDEO_INTERPOLATION = "video interpolation"
    OTHER_COMPUTER_VISION = "other computer vision"

class NLP(Enum):
    TRANSLATION = "translation"
    LANGUAGE_MODELING = "language modeling"
    GENERATION = "generation"
    OTHER_NLP = "other nlp"

class SPEECH(Enum):
    SYNTHESIS = "synthesis"
    RECOGNITION = "recognition"

class RECOMMENDATION(Enum):
    RECOMMENDATION = "recommendation"

class REINFORCEMENT_LEARNING(Enum):
    OTHER_RL = "other rl"

class OTHER(Enum):
    OTHER_TASKS = "other tasks"

class GNN(Enum):
    CLASSIFICATION = "classification"
