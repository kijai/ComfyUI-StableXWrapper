# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# --------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers.image_processor import PipelineImageInput
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
	ControlNetModel,
)

from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
...     "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> normals = pipe(image)

>>> vis = pipe.image_processor.visualize_normals(normals.prediction)
>>> vis[0].save("einstein_normals.png")
```
"""


@dataclass
class YosoDelightOutput(BaseOutput):
    """
    Output class for Marigold monocular normals prediction pipeline.

    Args:
        prediction (`np.ndarray`, `torch.Tensor`):
            Predicted normals with values in the range [-1, 1]. The shape is always $numimages \times 3 \times height
            \times width$, regardless of whether the images were passed as a 4D array or a list.
        uncertainty (`None`, `np.ndarray`, `torch.Tensor`):
            Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is $numimages
            \times 1 \times height \times width$.
        latent (`None`, `torch.Tensor`):
            Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline.
            The shape is $numimages * numensemble \times 4 \times latentheight \times latentwidth$.
    """

    prediction: Union[np.ndarray, torch.Tensor]
    latent: Union[None, torch.Tensor]
    gaus_noise: Union[None, torch.Tensor]


class YosoDelightPipeline():
    """ Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.github.io.
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        device: torch.device,
        dtype: torch.dtype,
        empty_text_embedding=None,
        t_start: Optional[int] = 401,
        pred_type: str = "delight",
    ):

        self.vae = vae
        self.unet = unet
        self.controlnet = controlnet
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)       
        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.empty_text_embedding = empty_text_embedding
        self.t_start= t_start # target_out latents
        self.device = device
        self.dtype = dtype
        self.pred_type = pred_type

    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(**self._progress_bar_config)
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        output_type: str = "pt",
        output_uncertainty: bool = False,
        skip_preprocess: bool = False,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`),
                `List[torch.Tensor]`: An input image or images used as an input for the normals estimation task. For
                arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible
                by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or
                three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the
                same width and height.
            num_inference_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Recommended values are 5 and higher for better precision, or 1 for
                faster inference.
            processing_resolution (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"closest"`): Defines the ensembling function applied in
                  every pixel location, can be either `"closest"` or `"mean"`.
            latents (`torch.Tensor`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_type (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.marigold.MarigoldDepthOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.marigold.MarigoldNormalsOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.marigold.MarigoldNormalsOutput`] is returned, otherwise a
                `tuple` is returned where the first element is the prediction, the second element is the uncertainty
                (or `None`), and the third is the latent (or `None`).
        """

        # 0. Resolving variables.
        device = self.device
        dtype = self.dtype
    
        # 4. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
        # optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
        # `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
        # divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
        # of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
        # operation and leads to the most reasonable results. Using the native image resolution or any other processing
        # resolution can lead to loss of either fine details or global context in the output predictions.
        if not skip_preprocess:
            image, padding, original_resolution = self.image_processor.preprocess(
                image, processing_resolution, resample_method_input, device, dtype
            )  # [N,3,PPH,PPW]
        else:
            padding = (0, 0)
            original_resolution = image.shape[2:]
        # 5. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
        # ensemble members. Each ensemble member is an independent diffused prediction, just initialized independently.
        # Latents of each such predictions across all input images and all ensemble members are represented in the
        # `pred_latent` variable. The variable `image_latent` is of the same shape: it contains each input image encoded
        # into latent space and replicated `E` times. The latents can be either generated (see `generator` to ensure
        # reproducibility), or passed explicitly via the `latents` argument. The latter can be set outside the pipeline
        # code. For example, in the Marigold-LCM video processing demo, the latents initialization of a frame is taken
        # as a convex combination of the latents output of the pipeline for the previous frame and a newly-sampled
        # noise. This behavior can be achieved by setting the `output_latent` argument to `True`. The latent space
        # dimensions are `(h, w)`. Encoding into latent space happens in batches of size `batch_size`.
        # Model invocation: self.vae.encoder.
        image_latent, pred_latent = self.prepare_latents(
            image, latents, generator, ensemble_size, batch_size
        )  # [N*E,4,h,w], [N*E,4,h,w]

        gaus_noise = pred_latent.detach().clone()
        del image

        # 6. obtain control_output

        cond_scale =controlnet_conditioning_scale
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            image_latent.detach(),
            self.t_start,
            encoder_hidden_states=self.empty_text_embedding,
            conditioning_scale=cond_scale,
            guess_mode=False,
            return_dict=False,
        )

        # 7. YOSO sampling
        latent_x_t = self.unet(
            pred_latent,
            self.t_start,
            encoder_hidden_states=self.empty_text_embedding,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]


        del (
            pred_latent,
            image_latent,
        )

        # decoder
        prediction = self.decode_prediction(latent_x_t)
        prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,3,PH,PW]

        prediction = self.image_processor.resize_antialias(
            prediction, original_resolution, resample_method_output, is_aa=False
        )  # [N,3,H,W]
        if self.pred_type == "normal":
            prediction = self.normalize_normals(prediction)
        
        if output_type == "np":
            prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,3]

        return YosoDelightOutput(
            prediction=prediction,
            latent=latent_x_t,
            gaus_noise=gaus_noise,
        )

    # Copied from diffusers.pipelines.marigold.pipeline_marigold_depth.MarigoldDepthPipeline.prepare_latents
    def prepare_latents(
        self,
        image: torch.Tensor,
        latents: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def retrieve_latents(encoder_output):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        image_latent = torch.cat(
            [
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # [N,4,h,w]
        image_latent = image_latent * self.vae.config.scaling_factor
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]

        if self.pred_type == "normal":
            pred_latent = latents
        else:
            pred_latent = torch.zeros_like(image_latent)
            
        if pred_latent is None:
            pred_latent = randn_tensor(
                image_latent.shape,
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # [N*E,4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]
        if self.pred_type == "normal":
            prediction = self.normalize_normals(prediction)

        return prediction  # [B,3,H,W]

    @staticmethod
    def normalize_normals(normals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")

        norm = torch.norm(normals, dim=1, keepdim=True)
        normals /= norm.clamp(min=eps)

        return normals

    @staticmethod
    def ensemble_normals(
        normals: torch.Tensor, output_uncertainty: bool, reduction: str = "closest"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Ensembles the normals maps represented by the `normals` tensor with expected shape `(B, 3, H, W)`, where B is
        the number of ensemble members for a given prediction of size `(H x W)`.

        Args:
            normals (`torch.Tensor`):
                Input ensemble normals maps.
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                Whether to output uncertainty map.
            reduction (`str`, *optional*, defaults to `"closest"`):
                Reduction method used to ensemble aligned predictions. The accepted values are: `"closest"` and
                `"mean"`.

        Returns:
            A tensor of aligned and ensembled normals maps with shape `(1, 3, H, W)` and optionally a tensor of
            uncertainties of shape `(1, 1, H, W)`.
        """
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")
        if reduction not in ("closest", "mean"):
            raise ValueError(f"Unrecognized reduction method: {reduction}.")

        mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
        mean_normals = MarigoldNormalsPipeline.normalize_normals(mean_normals)  # [1,3,H,W]

        sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]
        sim_cos = sim_cos.clamp(-1, 1)  # required to avoid NaN in uncertainty with fp16

        uncertainty = None
        if output_uncertainty:
            uncertainty = sim_cos.arccos()  # [E,1,H,W]
            uncertainty = uncertainty.mean(dim=0, keepdim=True) / np.pi  # [1,1,H,W]

        if reduction == "mean":
            return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

        closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
        closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
        closest_normals = torch.gather(normals, 0, closest_indices)  # [1,3,H,W]

        return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

