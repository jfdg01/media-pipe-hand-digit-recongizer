"""
Gesture recognition module for the Film Rating System.
Uses MediaPipe for hand gesture recognition.
"""

import os
import time
import warnings
from typing import Optional

# Suppress verbose logging before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GestureRecognizer:
    """Handles hand gesture recognition for scoring 1-5."""
    
    # Mapping from gesture names to numeric scores
    GESTURE_TO_SCORE = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5
    }
    
    def __init__(self, model_path: str = "digit_model_v1/gesture_recognizer.task", 
                 camera_url: str = None):
        self.model_path = model_path
        self.camera_url = camera_url  # For IP Webcam (phone camera)
        self.recognizer = None
        
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(base_options=base_options)
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
        else:
            print(f"⚠️  Gesture model not found at {model_path}")
    
    def recognize(self) -> Optional[int]:
        """
        Open camera and wait for a valid gesture (1-5).
        Press 'S' to skip and enter score manually.
        
        Returns:
            Score (1-5) or None if failed.
        """
        if not self.recognizer:
            print("❌ Gesture recognizer not initialized.")
            return self._fallback_input()
        
        # Use phone camera URL if provided, otherwise use local webcam
        if self.camera_url:
            print(f"📱 Connecting to phone camera: {self.camera_url}")
            cap = cv2.VideoCapture(self.camera_url)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            if self.camera_url:
                print(f"❌ Could not connect to phone camera at {self.camera_url}")
                print("   Make sure IP Webcam is running and the URL is correct.")
            else:
                print("❌ Could not open camera.")
            return self._fallback_input()
        
        print("\n✋ Show your score (1-5 fingers) to the camera...")
        print("   Hold your gesture steady for recognition.")
        print("   Press 'S' to skip and enter manually.\n")
        
        stable_gesture = None
        stable_count = 0
        required_stable_frames = 10  # Need 10 consistent frames
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip vertically if using phone camera (IP Webcam orientation fix)
                if self.camera_url:
                    frame = cv2.flip(frame, 0)  # 0 = vertical flip
                
                # Convert frame for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Recognize gesture
                result = self.recognizer.recognize(mp_image)
                
                current_gesture = None
                confidence = 0.0
                
                if result.gestures:
                    category = result.gestures[0][0]
                    gesture_name = category.category_name
                    confidence = category.score
                    
                    if gesture_name in self.GESTURE_TO_SCORE and confidence > 0.7:
                        current_gesture = gesture_name
                
                # Check for stable gesture
                if current_gesture == stable_gesture and current_gesture is not None:
                    stable_count += 1
                else:
                    stable_gesture = current_gesture
                    stable_count = 1 if current_gesture else 0
                
                # Display info on frame
                display_text = f"Gesture: {stable_gesture or 'None'} ({stable_count}/{required_stable_frames})"
                cv2.putText(frame, display_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'S' to skip", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.namedWindow("Show Your Score (1-5)", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Show Your Score (1-5)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Show Your Score (1-5)", frame)
                
                # If we have a stable gesture
                if stable_count >= required_stable_frames:
                    score = self.GESTURE_TO_SCORE[stable_gesture]
                    print(f"✅ Gesture recognized: {stable_gesture} (Score: {score})")
                    cap.release()
                    cv2.destroyAllWindows()
                    return score
                
                # Check for 'S' to skip to manual input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    print("⏭️  Skipping to manual input...")
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return self._fallback_input()
    
    def _fallback_input(self) -> Optional[int]:
        """Fallback to keyboard input if camera/model fails."""
        print("\n⌨️  Falling back to keyboard input.")
        try:
            score = int(input("Enter your score (1-5): "))
            if 1 <= score <= 5:
                return score
            print("❌ Score must be between 1 and 5.")
        except ValueError:
            print("❌ Invalid input.")
        return None
