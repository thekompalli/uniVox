"""
Setup Speaker Gallery Script
One-time script to enroll speakers into the gallery
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.speaker_service import SpeakerIdentificationService
from src.utils.audio_utils import AudioUtils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_speaker_gallery():
    """Setup speaker gallery with 3 speakers"""
    
    speaker_service = SpeakerIdentificationService()
    audio_utils = AudioUtils()
    
    # Define your 3 speaker audio files (using absolute paths)
    speakers = [
        {
            "id": "harsha",
            "name": "Harsha",
            "audio_path": str(project_root / "data" / "audio" / "speaker_audio" / "audio-harsha.mp3"),
            "language": "hindi,english"
        },
        {
            "id": "keerthi", 
            "name": "Keerthi",
            "audio_path": str(project_root / "data" / "audio" / "speaker_audio" / "audio-keerthi.mp3"),
            "language": "hindi,english"
        },
        {
            "id": "thatwik",
            "name": "Thatwik", 
            "audio_path": str(project_root / "data" / "audio" / "speaker_audio" / "audio-thatwik.mp3"),
            "language": "hindi,english"
        }
    ]
    
    logger.info("Starting speaker gallery setup...")
    
    for speaker_info in speakers:
        try:
            audio_path = Path(speaker_info["audio_path"])
            
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                logger.error(f"Please place the audio file at: {audio_path.absolute()}")
                continue
            
            logger.info(f"\nProcessing {speaker_info['name']} ({speaker_info['id']})...")
            logger.info(f"Loading audio from: {audio_path}")
            
            # Load audio
            audio_data, sample_rate = audio_utils.load_audio(str(audio_path))
            
            # Check duration
            duration = len(audio_data) / sample_rate
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            if duration < 2.0:
                logger.warning(f"⚠️  Audio is short ({duration:.1f}s). Recommended: at least 3 seconds")
            
            # Add to gallery
            metadata = {
                "name": speaker_info["name"],
                "language": speaker_info["language"],
                "duration": duration,
                "original_file": str(audio_path.name)
            }
            
            success = await speaker_service.add_speaker_to_gallery(
                speaker_id=speaker_info["id"],
                audio_samples=[audio_data],
                sample_rate=sample_rate,
                metadata=metadata
            )
            
            if success:
                logger.info(f"✅ Successfully enrolled {speaker_info['name']}")
            else:
                logger.error(f"❌ Failed to enroll {speaker_info['name']}")
        
        except Exception as e:
            logger.exception(f"❌ Error processing {speaker_info['name']}: {e}")
    
    # Display final gallery status
    logger.info("\n" + "="*50)
    logger.info("SPEAKER GALLERY SETUP COMPLETE")
    logger.info("="*50)
    
    gallery_info = await speaker_service.get_gallery_info()
    logger.info(f"\nTotal speakers enrolled: {gallery_info['total_speakers']}")
    logger.info(f"Speaker IDs: {', '.join(gallery_info['speakers'])}")
    
    logger.info("\n✅ Gallery is ready for speaker identification!")
    logger.info("You can now process meeting audio through the frontend.")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║           Speaker Gallery Setup Script                     ║
╚════════════════════════════════════════════════════════════╝

This script will enroll 3 speakers into the gallery.

BEFORE RUNNING:
1. Place your 3 speaker audio files (MP3) in: data/speaker_audio/
   - audio-harsha.mp3
   - audio-keerthi.mp3  
   - audio-thatwik.mp3

2. Or update the file paths in this script (lines 23-39)

Press Ctrl+C to cancel, or wait 3 seconds to continue...
    """)
    
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        sys.exit(0)
    
    asyncio.run(setup_speaker_gallery())