#!/usr/bin/env python3
"""
Film Rating System - Main Entry Point

A multimodal application that combines speech recognition,
film API lookup, and hand gesture recognition for rating movies.
"""

import os
import sys
import warnings

# Load environment variables from .env file automatically
from dotenv import load_dotenv
load_dotenv()

# Suppress verbose TensorFlow, CUDA, and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress MediaPipe protobuf warnings
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Add src to path for direct execution
# Add src to path for direct execution
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent))

try:
    from .controller import FilmRatingController
except ImportError:
    # Fallback for direct execution without -m
    sys.path.insert(0, str(Path(__file__).parent))
    from controller import FilmRatingController



def main():
    """Main entry point with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal Film Rating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main                    # Full experience
    python -m src.main --keyboard         # Type film name
    python -m src.main --language en-US   # English speech
        """
    )
    
    parser.add_argument(
        "--language", "-l",
        default="es-ES",
        help="Language code for speech recognition (default: es-ES)"
    )
    parser.add_argument(
        "--keyboard", "-k",
        action="store_true",
        help="Use keyboard input instead of speech"
    )
    parser.add_argument(
        "--camera-url", "-c",
        default=os.environ.get("CAMERA_URL"),
        help="Phone camera URL (or set CAMERA_URL env var)"
    )
    
    args = parser.parse_args()
    
    controller = FilmRatingController(
        language=args.language,
        camera_url=args.camera_url
    )
    
    # Override speech with keyboard input if requested
    if args.keyboard:
        def keyboard_listen(prompt=""):
            print(prompt)
            return input("Enter film title: ").strip()
        controller.speech.listen = keyboard_listen
    
    controller.run()


if __name__ == "__main__":
    main()
