 
import os
import librosa
import warnings
import tempfile
import subprocess
import shutil

# Optional heavy dependencies for Whisper model (transformers + torch)
try:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    torch = None
    WhisperProcessor = None
    WhisperForConditionalGeneration = None
    _TRANSFORMERS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")


class AudioProcessor:
   
    
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initialize the audio processor
        
        Args:
            model_name (str): Whisper model to use
                - whisper-tiny: Fastest, less accurate
                - whisper-small: Good balance (RECOMMENDED)
                - whisper-medium: More accurate, slower
                - whisper-large: Most accurate, very slow
        """
        print(f"Initializing AudioProcessor with {model_name}...")
        
        # If transformers/torch are not available, mark transcription as unavailable and defer model loading
        if not _TRANSFORMERS_AVAILABLE:
            print("Whisper dependencies not installed. Transcription disabled. Install 'transformers' and 'torch' to enable transcription.")
            self.available = False
            self.processor = None
            self.model = None
            self.device = "cpu"
            self._model_loaded = False
            return

        # We will defer heavy model loading until transcription is actually requested.
        # This avoids long import times and large downloads during tests or when the server starts.
        self.processor = None
        self.model = None
        self.device = None
        self._model_loaded = False
        # Mark that the environment has the required packages; actual model may still fail to load on demand
        self.available = True
        self.model_name = model_name
        print("Whisper dependencies detected. Model will be loaded lazily on first transcription request.")
    
    
    def _load_model(self):
        """
        Lazily load the Whisper processor and model. This is deferred until the first transcription
        request to avoid heavy downloads and long import times at server startup or during tests.
        """
        if getattr(self, '_model_loaded', False):
            return

        try:
            # Import here to minimize import-time side effects
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging as hf_logging

            # Reduce transformer verbosity to avoid massive logs during model loading
            try:
                hf_logging.set_verbosity_error()
            except Exception:
                pass

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper model ({self.model_name}) on {self.device}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self._model_loaded = True
            print("Whisper model loaded successfully")
        except Exception as e:
            # If loading fails, mark as unavailable and raise a clear runtime error
            self.available = False
            raise RuntimeError(f"Failed to load Whisper model: {e}")


    def _convert_to_wav(self, src_path: str) -> str:
        """Attempt to convert an audio file to a 16kHz mono WAV using ffmpeg.

        Returns the path to the converted wav file. Raises RuntimeError with a helpful message
        if conversion is not possible.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to enable conversion for MP3/M4A files or upload WAV files.")

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp_wav,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to convert audio: {e}")

        return tmp_wav


    def transcribe_audio(self, audio_path):
        """
        Convert audio file to text
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Ensure the Whisper model dependencies are available
            if not getattr(self, 'available', False):
                raise RuntimeError("Whisper model not available. Install 'transformers' and 'torch' to enable transcription.")

            # Lazily load the model when needed
            if not getattr(self, '_model_loaded', False):
                self._load_model()

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            print(f"Processing: {os.path.basename(audio_path)}")
            
            # Step 1: Load audio file and resample to 16kHz (Whisper requirement)
            temp_converted = None
            try:
                audio, sampling_rate = librosa.load(audio_path, sr=16000)
            except Exception as load_err:
                print(f"Audio load failed ({load_err}); attempting ffmpeg conversion fallback")
                try:
                    temp_converted = self._convert_to_wav(audio_path)
                    audio, sampling_rate = librosa.load(temp_converted, sr=16000)
                except Exception as conv_err:
                    # Clean up any temporary file
                    if temp_converted and os.path.exists(temp_converted):
                        try:
                            os.remove(temp_converted)
                        except Exception:
                            pass
                    raise RuntimeError(f"Failed to load or convert audio: {conv_err} (original: {load_err})")
            finally:
                # If we created a temp converted file, remove it after loading
                if 'temp_converted' in locals() and temp_converted and os.path.exists(temp_converted):
                    try:
                        os.remove(temp_converted)
                    except Exception:
                        pass

            print(f"Audio duration: {len(audio)/sampling_rate:.2f} seconds")
            
            # Step 2: Process audio into model input format
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            # Move to GPU/CPU
            input_features = input_features.to(self.device)
            
            # Step 3: Generate transcription
            with __import__('torch').no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Step 4: Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            print(f"Transcription complete: {len(transcription)} characters")
            
            return transcription.strip()
            
        except Exception as e:
            msg = f"Audio processing failed: {str(e)}"
            print(f"Transcription error: {msg}")
            # Raise a runtime error with a clear message for upstream handling
            raise RuntimeError(msg)
    
    
    def transcribe_multiple_files(self, folder_path, output_file="transcripts.txt"):
        """
        Transcribe all audio files in a folder
        
        Args:
            folder_path (str): Path to folder containing audio files
            output_file (str): Output file to save transcriptions
            
        Returns:
            dict: Dictionary of filename -> transcription
        """
        transcriptions = {}
        
        # Supported audio formats
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a')
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(audio_extensions)]
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, filename in enumerate(audio_files, 1):
                print(f"\n[{idx}/{len(audio_files)}] Processing: {filename}")
                
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # Transcribe
                    transcription = self.transcribe_audio(file_path)
                    
                    # Save to dictionary
                    transcriptions[filename] = transcription
                    
                    # Write to file
                    f.write(f"{filename}: {transcription}\n\n")
                    
                except Exception as e:
                    print(f"   Failed to process {filename}: {str(e)}")
                    transcriptions[filename] = f"ERROR: {str(e)}"
        
        print(f"\nAll transcriptions saved to {output_file}")
        return transcriptions
    
    
    def get_audio_info(self, audio_path):
        """
        Get information about an audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Audio information
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            return {
                "duration_seconds": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1 if len(audio.shape) == 1 else audio.shape[0],
                "file_size_mb": os.path.getsize(audio_path) / (1024 * 1024)
            }
            
        except Exception as e:
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Audio Processor")
    print("=" * 60)
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Test with sample file (if exists)
    test_audio = "./audio/test.mp3"
    if os.path.exists(test_audio):
        result = processor.transcribe_audio(test_audio)
        print(f"\nTranscription: {result}")
    else:
        print("\nNo test audio file found. Place a file at ./audio/test.mp3 to test.")
    
    print("\n" + "=" * 60)