# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import os


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load both models from local files
        self.models = {
            "transformer": Zonos.from_pretrained("models/transformer", device=self.device),
            "hybrid": Zonos.from_pretrained("models/hybrid", device=self.device)
        }
        # Default to transformer model
        self.model = self.models["transformer"]
        
    def predict(
        self,
        text: str = Input(description="Text to convert to speech"),
        reference_audio: Path = Input(description="Reference audio file for voice cloning", default=None),
        language: str = Input(description="Language code (e.g., 'en-us', 'ja', 'zh', 'fr', 'de')", default="en-us"),
        speaking_rate: float = Input(description="Speaking rate multiplier", ge=0.5, le=2.0, default=1.0),
        model_type: str = Input(description="Model type to use ('transformer' or 'hybrid')", default="transformer"),
    ) -> Path:
        """Run a single prediction on the model"""
        # Select the model
        if model_type not in self.models:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'transformer' or 'hybrid'")
        self.model = self.models[model_type]
        
        # Load and process reference audio if provided
        if reference_audio is not None:
            wav, sampling_rate = torchaudio.load(str(reference_audio))
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
        else:
            speaker = None  # Model should use default speaker if None

        # Prepare conditioning
        cond_dict = make_cond_dict(
            text=text,
            speaker=speaker,
            language=language,
            speaking_rate=speaking_rate
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        # Generate audio
        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()

        # Save output
        output_path = Path("/tmp/output.wav")
        torchaudio.save(
            str(output_path),
            wavs[0],
            self.model.autoencoder.sampling_rate
        )

        return output_path
