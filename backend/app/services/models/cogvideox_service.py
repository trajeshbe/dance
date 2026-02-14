"""
CogVideoX Image-to-Video Service with LangGraph Agent Orchestration

This service uses CogVideoX-5b-I2V from HuggingFace to animate static images.
Includes LangGraph agents for intelligent task orchestration.

Model: THUDM/CogVideoX-5b-I2V
Paper: https://arxiv.org/abs/2408.06072
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ============================================================================
# LANGGRAPH STATE DEFINITION
# ============================================================================

class VideoGenerationState(TypedDict):
    """State for the video generation agent workflow"""
    image_path: str
    prompt: str
    num_frames: int
    fps: int
    guidance_scale: float
    num_inference_steps: int
    status: str
    error: Optional[str]
    output_path: Optional[str]
    device: str


# ============================================================================
# COGVIDEOX SERVICE
# ============================================================================

class CogVideoXService:
    """
    CogVideoX Image-to-Video Generation Service

    Features:
    - Float16 precision for memory efficiency
    - Sequential CPU offloading for low-VRAM GPUs
    - LangGraph agent orchestration
    - Optimized for Colab T4 (16GB VRAM)
    """

    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-5b-I2V",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_cpu_offload: bool = True,
    ):
        """
        Initialize CogVideoX pipeline

        Args:
            model_id: HuggingFace model ID
            device: Device to use (cuda/cpu)
            enable_cpu_offload: Enable sequential CPU offloading for low VRAM
        """
        self.model_id = model_id
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.pipeline = None
        self.agent_graph = None

        logger.info(f"Initializing CogVideoX service on {device}")

    def load_model(self) -> None:
        """Load CogVideoX model with optimizations"""
        if self.pipeline is not None:
            logger.info("Model already loaded")
            return

        try:
            logger.info(f"Loading CogVideoX from {self.model_id}...")

            # Load pipeline with float16 for memory efficiency
            self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
            )

            # Enable memory optimizations
            if self.enable_cpu_offload and self.device == "cuda":
                logger.info("Enabling sequential CPU offloading...")
                self.pipeline.enable_sequential_cpu_offload()
            else:
                self.pipeline.to(self.device)

            # Enable attention slicing for lower memory
            self.pipeline.enable_attention_slicing()

            logger.info("✅ CogVideoX model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load CogVideoX model: {e}")
            raise

    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory cleared")

    # ========================================================================
    # LANGGRAPH AGENT NODES
    # ========================================================================

    def prepare_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Agent node: Prepare inputs"""
        logger.info("🤖 Agent: Preparing inputs...")

        try:
            # Validate image exists
            if not os.path.exists(state["image_path"]):
                raise FileNotFoundError(f"Image not found: {state['image_path']}")

            # Ensure model is loaded
            self.load_model()

            state["status"] = "prepared"
            logger.info("✅ Inputs prepared successfully")

        except Exception as e:
            state["status"] = "error"
            state["error"] = str(e)
            logger.error(f"❌ Preparation failed: {e}")

        return state

    def generate_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Agent node: Generate video"""
        logger.info("🤖 Agent: Generating video...")

        try:
            # Load and preprocess image
            image = Image.open(state["image_path"]).convert("RGB")

            # Generate video frames
            logger.info(f"Generating {state['num_frames']} frames...")
            video_frames = self.pipeline(
                image=image,
                prompt=state["prompt"],
                num_frames=state["num_frames"],
                guidance_scale=state["guidance_scale"],
                num_inference_steps=state["num_inference_steps"],
                generator=torch.Generator(device=state["device"]).manual_seed(42),
            ).frames[0]

            # Export to video file
            output_path = state.get("output_path") or "/tmp/cogvideox_output.mp4"
            export_to_video(
                video_frames,
                output_path,
                fps=state["fps"]
            )

            state["output_path"] = output_path
            state["status"] = "completed"
            logger.info(f"✅ Video generated: {output_path}")

        except Exception as e:
            state["status"] = "error"
            state["error"] = str(e)
            logger.error(f"❌ Generation failed: {e}")

        return state

    def cleanup_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """Agent node: Cleanup resources"""
        logger.info("🤖 Agent: Cleaning up...")

        # Optionally unload model to free memory
        # self.unload_model()

        logger.info("✅ Cleanup complete")
        return state

    # ========================================================================
    # LANGGRAPH WORKFLOW SETUP
    # ========================================================================

    def build_agent_graph(self) -> StateGraph:
        """Build LangGraph agent workflow"""
        workflow = StateGraph(VideoGenerationState)

        # Add nodes
        workflow.add_node("prepare", self.prepare_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("cleanup", self.cleanup_node)

        # Define edges
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "generate")
        workflow.add_edge("generate", "cleanup")
        workflow.add_edge("cleanup", END)

        return workflow.compile()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    async def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_path: Optional[str] = None,
        num_frames: int = 49,
        fps: int = 8,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Generate video from image using CogVideoX with LangGraph agents

        Args:
            image_path: Path to input image
            prompt: Text description for animation
            output_path: Output video path (optional)
            num_frames: Number of frames to generate (default: 49)
            fps: Frames per second (default: 8)
            guidance_scale: Classifier-free guidance scale (default: 6.0)
            num_inference_steps: Number of denoising steps (default: 50)

        Returns:
            Dict with output_path, status, and metadata
        """
        logger.info("🎬 Starting CogVideoX video generation...")

        # Build agent graph if not exists
        if self.agent_graph is None:
            self.agent_graph = self.build_agent_graph()

        # Initial state
        initial_state: VideoGenerationState = {
            "image_path": image_path,
            "prompt": prompt,
            "num_frames": num_frames,
            "fps": fps,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "output_path": output_path,
            "device": self.device,
            "status": "initialized",
            "error": None,
        }

        # Run agent workflow
        logger.info("🤖 Executing LangGraph agent workflow...")
        final_state = self.agent_graph.invoke(initial_state)

        # Return results
        if final_state["status"] == "completed":
            return {
                "success": True,
                "output_path": final_state["output_path"],
                "status": final_state["status"],
                "metadata": {
                    "model": self.model_id,
                    "num_frames": num_frames,
                    "fps": fps,
                    "prompt": prompt,
                }
            }
        else:
            return {
                "success": False,
                "error": final_state.get("error", "Unknown error"),
                "status": final_state["status"],
            }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_cogvideox_service: Optional[CogVideoXService] = None


def get_cogvideox_service() -> CogVideoXService:
    """Get singleton CogVideoX service instance"""
    global _cogvideox_service
    if _cogvideox_service is None:
        _cogvideox_service = CogVideoXService()
    return _cogvideox_service
