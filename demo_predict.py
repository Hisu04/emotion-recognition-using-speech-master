from core.emotion_recognition import EmotionRecognizer
import os
import random
from sklearn.neural_network import MLPClassifier

def run_demo():
    # 1. Get a random audio file from emodb
    wav_dir = os.path.join("data", "emodb", "wav")
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    
    if not wav_files:
        print("No audio files found in data/emodb/wav")
        return

    sample_file = random.choice(wav_files)
    sample_path = os.path.join(wav_dir, sample_file)
    
    print(f"Testing with sample file: {sample_path}")

    # 2. Use a fresh MLPClassifier to avoid pickle incompatibility
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    print(f"Using model: MLPClassifier (freshly initialized)")

    # 3. Initialize EmotionRecognizer
    # We use some common emotions
    emotions = ["sad", "neutral", "happy", "angry"]
    features = ["mfcc", "chroma", "mel"]
    
    try:
        # Initialize the recognizer with the fresh model
        detector = EmotionRecognizer(model=model, emotions=emotions, features=features, verbose=0)
        
        print("Training model (this will extract features and train from scratch)...")
        detector.train()
        
        print(f"Test accuracy score: {detector.test_score()*100:.3f}%")
        
        print(f"Predicting emotion for {sample_path}...")
        result = detector.predict(sample_path)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
