"""
Diffusion utilities for DIRE and SeDID methods.

Implements DDIM inversion and reconstruction using Stable Diffusion.
"""
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from diffusers.models import AutoencoderKL, UNet2DConditionModel
except ImportError:
    raise ImportError("Please install diffusers: pip install diffusers transformers accelerate")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class DiffusionReconstructor:
    """
    Diffusion-based image reconstruction for DIRE and SeDID.

    Uses Stable Diffusion with DDIM scheduler for deterministic
    inversion and reconstruction.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize diffusion reconstructor.

        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run on
            dtype: Data type for model (float16 recommended for GPU)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        logger.info(f"Loading diffusion model: {model_id}")

        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)

        # Setup DDIM scheduler for deterministic operations
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        # Get null text embeddings (for unconditional generation)
        self.null_text_embeddings = self._get_null_embeddings()

        logger.info("Diffusion model loaded successfully")

    def _get_null_embeddings(self) -> torch.Tensor:
        """Get text embeddings for empty prompt (unconditional)."""
        text_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            null_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return null_embeddings

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using VAE.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1]

        Returns:
            latent: Latent tensor [B, 4, H/8, W/8]
        """
        with torch.no_grad():
            latent = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image using VAE.

        Args:
            latent: Latent tensor [B, 4, H/8, W/8]

        Returns:
            image: Image tensor [B, 3, H, W] in range [-1, 1]
        """
        with torch.no_grad():
            latent = latent / self.vae.config.scaling_factor
            image = self.vae.decode(latent.to(self.dtype)).sample
        return image

    def ddim_inversion(
        self,
        image: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        DDIM inversion: encode image to noise space.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1]
            num_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            latent_t: Noisy latent at timestep T
        """
        # Encode to latent space
        latent = self.encode_image(image)

        # Setup scheduler
        self.scheduler.set_timesteps(num_steps)
        timesteps = reversed(self.scheduler.timesteps)

        # DDIM inversion loop
        with torch.no_grad():
            for t in timesteps:
                t_tensor = torch.tensor([t], device=self.device)

                # Predict noise
                noise_pred = self.unet(
                    latent.to(self.dtype),
                    t_tensor,
                    encoder_hidden_states=self.null_text_embeddings.expand(latent.shape[0], -1, -1)
                ).sample

                # Compute previous latent (inversion direction)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t - self.scheduler.config.num_train_timesteps // num_steps] if t > 0 else torch.tensor(1.0)

                # DDIM inversion step
                pred_x0 = (latent - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
                latent = alpha_prod_t_prev.sqrt() * pred_x0 + (1 - alpha_prod_t_prev).sqrt() * noise_pred

        return latent

    def ddim_reconstruction(
        self,
        latent_t: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        DDIM reconstruction: denoise from noise to image.

        Args:
            latent_t: Noisy latent tensor
            num_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            image: Reconstructed image tensor [B, 3, H, W]
        """
        # Setup scheduler
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps

        latent = latent_t

        # DDIM denoising loop
        with torch.no_grad():
            for t in timesteps:
                t_tensor = torch.tensor([t], device=self.device)

                # Predict noise
                noise_pred = self.unet(
                    latent.to(self.dtype),
                    t_tensor,
                    encoder_hidden_states=self.null_text_embeddings.expand(latent.shape[0], -1, -1)
                ).sample

                # DDIM step
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        # Decode to image
        image = self.decode_latent(latent)
        return image.float()

    def compute_dire(
        self,
        image: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute DIRE (Diffusion Reconstruction Error).

        DIRE = |original_image - reconstructed_image|

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1]
            num_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            dire: DIRE error map [B, 3, H, W]
        """
        # DDIM inversion
        latent_t = self.ddim_inversion(image, num_steps, guidance_scale)

        # DDIM reconstruction
        reconstructed = self.ddim_reconstruction(latent_t, num_steps, guidance_scale)

        # Compute absolute difference
        dire = torch.abs(image - reconstructed)

        return dire

    def compute_sedid_at_timestep(
        self,
        image: torch.Tensor,
        timestep: int,
        num_steps: int = 20
    ) -> torch.Tensor:
        """
        Compute SeDID stepwise error at a specific timestep.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1]
            timestep: Timestep to analyze (0-1000)
            num_steps: Number of DDIM steps for overall process

        Returns:
            error: Stepwise error at the specified timestep
        """
        # Encode to latent
        latent = self.encode_image(image)

        # Add noise to the specified timestep
        self.scheduler.set_timesteps(num_steps)
        noise = torch.randn_like(latent)
        t_tensor = torch.tensor([timestep], device=self.device)

        noisy_latent = self.scheduler.add_noise(latent, noise, t_tensor)

        # Predict noise at this timestep
        with torch.no_grad():
            predicted_noise = self.unet(
                noisy_latent.to(self.dtype),
                t_tensor,
                encoder_hidden_states=self.null_text_embeddings.expand(latent.shape[0], -1, -1)
            ).sample

        # Compute error between actual and predicted noise
        error = torch.abs(noise - predicted_noise.float())

        # Decode error to image space for visualization
        error_image = self.decode_latent(error * self.vae.config.scaling_factor)

        return torch.abs(error_image)

    def compute_sedid(
        self,
        image: torch.Tensor,
        timesteps: List[int] = [250, 500, 750],
        num_steps: int = 20
    ) -> torch.Tensor:
        """
        Compute SeDID features for multiple timesteps.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1]
            timesteps: List of timesteps to analyze
            num_steps: Number of DDIM steps

        Returns:
            sedid_features: Concatenated error features [B, 3*len(timesteps), H, W]
        """
        errors = []
        for t in timesteps:
            error = self.compute_sedid_at_timestep(image, t, num_steps)
            errors.append(error)

        # Concatenate along channel dimension
        sedid_features = torch.cat(errors, dim=1)
        return sedid_features


def preprocess_image_for_diffusion(
    image: Image.Image,
    size: int = 512
) -> torch.Tensor:
    """
    Preprocess PIL image for diffusion model.

    Args:
        image: PIL Image
        size: Target size

    Returns:
        tensor: Image tensor [1, 3, H, W] in range [-1, 1]
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor


def postprocess_image_from_diffusion(tensor: torch.Tensor) -> Image.Image:
    """
    Postprocess tensor from diffusion model to PIL image.

    Args:
        tensor: Image tensor in range [-1, 1]

    Returns:
        image: PIL Image
    """
    # Clamp and scale to [0, 1]
    tensor = (tensor.clamp(-1, 1) + 1) / 2

    # Convert to PIL
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * 255).astype("uint8")
    image = Image.fromarray(tensor)

    return image


if __name__ == "__main__":
    # Test diffusion utilities
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Note: This test requires significant GPU memory (~8GB)
    if device == "cuda":
        print("\n--- Testing DiffusionReconstructor ---")

        reconstructor = DiffusionReconstructor(
            model_id="runwayml/stable-diffusion-v1-5",
            device=device
        )

        # Create dummy image
        dummy_image = torch.randn(1, 3, 512, 512).to(device)
        dummy_image = dummy_image.clamp(-1, 1)

        print(f"Input shape: {dummy_image.shape}")

        # Test DIRE computation
        print("\nComputing DIRE...")
        dire = reconstructor.compute_dire(dummy_image, num_steps=10)
        print(f"DIRE shape: {dire.shape}")
        print(f"DIRE range: [{dire.min():.4f}, {dire.max():.4f}]")

        # Test SeDID computation
        print("\nComputing SeDID...")
        sedid = reconstructor.compute_sedid(dummy_image, timesteps=[250, 500], num_steps=10)
        print(f"SeDID shape: {sedid.shape}")
        print(f"SeDID range: [{sedid.min():.4f}, {sedid.max():.4f}]")
    else:
        print("CUDA not available. Skipping diffusion test (requires GPU).")
