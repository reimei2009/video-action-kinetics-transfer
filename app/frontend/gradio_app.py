"""
Gradio Frontend - Web UI cho Video Action Recognition

Tính năng:
- Upload video (drag-and-drop)
- Chọn top-K predictions
- Hiển thị kết quả (predictions + confidence scores)
- Video preview
- Model info
"""

import gradio as gr
import requests
from pathlib import Path
from typing import Tuple, List
import json

from app.core.config import settings


# ==========================================
# API Client
# ==========================================

class APIClient:
    """
    Client để gọi FastAPI endpoints
    """
    
    def __init__(self, base_url: str = None):
        if base_url is None:
            # Use 127.0.0.1 instead of 0.0.0.0 for client connections
            host = "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST
            base_url = f"http://{host}:{settings.PORT}/api/v1"
        self.base_url = base_url
    
    def predict(self, video_path: str, top_k: int = 5) -> dict:
        """
        Gọi /predict endpoint
        
        Args:
            video_path: Path to video file
            top_k: Number of predictions
        
        Returns:
            API response dict
        """
        try:
            url = f"{self.base_url}/predict"
            
            # Open file
            with open(video_path, 'rb') as f:
                files = {'file': (Path(video_path).name, f, 'video/mp4')}
                params = {'top_k': top_k}
                
                # POST request
                response = requests.post(url, files=files, params=params, timeout=60)
            
            # Check response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "message": f"Error: {response.status_code}",
                    "details": response.text
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection error: {str(e)}"
            }
    
    def get_health(self) -> dict:
        """Gọi /health endpoint"""
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            return response.json()
        except:
            return {"status": "unreachable"}
    
    def get_models(self) -> dict:
        """Gọi /models endpoint"""
        try:
            url = f"{self.base_url}/models"
            response = requests.get(url, timeout=5)
            return response.json()
        except:
            return {"current_model": "unknown"}


# ==========================================
# Gradio Interface Functions
# ==========================================

# Global API client
api_client = APIClient()


def predict_video(
    video_file, 
    top_k: int
) -> Tuple[str, str, str]:
    """
    Hàm chính để predict video
    
    Args:
        video_file: Gradio video input
        top_k: Number of predictions
    
    Returns:
        (predictions_html, metadata_html, video_path)
    """
    if video_file is None:
        return "⚠️ Please upload a video file", "", None
    
    try:
        # Call API
        result = api_client.predict(video_file, top_k=int(top_k))
        
        if not result.get("success", False):
            error_msg = result.get("message", "Unknown error")
            return f"❌ Error: {error_msg}", "", video_file
        
        # Format predictions
        predictions = result["predictions"]
        predictions_html = format_predictions(predictions)
        
        # Format metadata
        metadata = result.get("video_metadata", {})
        processing_time = result.get("processing_time", 0)
        model_name = result.get("model_name", "unknown")
        
        metadata_html = format_metadata(metadata, processing_time, model_name)
        
        return predictions_html, metadata_html, video_file
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", video_file


def format_predictions(predictions: List[dict]) -> str:
    """
    Format predictions thành HTML với progress bars
    
    Args:
        predictions: List of {label, confidence, rank}
    
    Returns:
        HTML string
    """
    html = "<div style='font-family: Arial, sans-serif;'>"
    html += "<h3>🎯 Top Predictions</h3>"
    
    for pred in predictions:
        rank = pred["rank"]
        label = pred["label"]
        confidence = pred["confidence"]
        percentage = confidence * 100
        
        # Color based on confidence
        if confidence > 0.15:
            color = "#10b981"  # Green
        elif confidence > 0.10:
            color = "#3b82f6"  # Blue
        else:
            color = "#6b7280"  # Gray
        
        html += f"""
        <div style='margin-bottom: 15px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <span style='font-weight: bold;'>#{rank}. {label}</span>
                <span style='color: {color}; font-weight: bold;'>{percentage:.2f}%</span>
            </div>
            <div style='background: #e5e7eb; border-radius: 10px; height: 20px; overflow: hidden;'>
                <div style='background: {color}; height: 100%; width: {percentage}%; 
                            transition: width 0.3s ease;'></div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def format_metadata(
    metadata: dict, 
    processing_time: float, 
    model_name: str
) -> str:
    """
    Format metadata thành HTML
    
    Args:
        metadata: Video metadata dict
        processing_time: Processing time (seconds)
        model_name: Model name
    
    Returns:
        HTML string
    """
    html = "<div style='font-family: Arial, sans-serif; background: #f9fafb; padding: 15px; border-radius: 10px;'>"
    html += "<h4>📊 Processing Info</h4>"
    
    html += f"<p><strong>Model:</strong> {model_name.upper()}</p>"
    html += f"<p><strong>Processing Time:</strong> {processing_time:.3f}s</p>"
    
    if metadata:
        html += f"<p><strong>Filename:</strong> {metadata.get('filename', 'N/A')}</p>"
        html += f"<p><strong>Duration:</strong> {metadata.get('duration', 'N/A')}s</p>"
        
        if metadata.get('fps'):
            html += f"<p><strong>FPS:</strong> {metadata.get('fps')}</p>"
        
        html += f"<p><strong>Size:</strong> {metadata.get('size_mb', 'N/A')} MB</p>"
    
    html += "</div>"
    return html


def get_server_status() -> str:
    """
    Lấy server status
    
    Returns:
        HTML string với server info
    """
    try:
        health = api_client.get_health()
        model_info = api_client.get_models()
        
        status = health.get("status", "unknown")
        model_name = model_info.get("current_model", "unknown")
        
        if status == "healthy":
            status_color = "#10b981"
            status_icon = "✅"
        else:
            status_color = "#ef4444"
            status_icon = "❌"
        
        html = f"""
        <div style='font-family: Arial, sans-serif; padding: 10px; 
                    background: #f9fafb; border-radius: 10px;'>
            <h4>{status_icon} Server Status</h4>
            <p><strong>Status:</strong> <span style='color: {status_color};'>{status.upper()}</span></p>
            <p><strong>Model:</strong> {model_name.upper()}</p>
        </div>
        """
        return html
    
    except:
        return """
        <div style='font-family: Arial, sans-serif; padding: 10px; 
                    background: #fee; border-radius: 10px;'>
            <h4>⚠️ Server Unreachable</h4>
            <p>Cannot connect to API server. Make sure it's running.</p>
        </div>
        """


# ==========================================
# Gradio Interface
# ==========================================

def create_gradio_interface():
    """
    Tạo Gradio interface
    
    Returns:
        gr.Blocks interface
    """
    with gr.Blocks(
        title="Video Action Recognition",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🎥 Video Action Recognition
            
            Upload a video and get AI predictions for the action being performed!
            
            **Powered by:** X3D Model (Facebook Research) | **Framework:** FastAPI + Gradio
            """
        )
        
        # Server status
        with gr.Row():
            server_status = gr.HTML(get_server_status())
        
        gr.Markdown("---")
        
        # Main interface
        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Video")
                
                video_input = gr.Video(
                    label="Video File",
                    sources=["upload"],
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Predictions",
                    info="How many top predictions to show"
                )
                
                predict_btn = gr.Button(
                    "🚀 Predict Action",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    **Supported formats:** MP4, AVI, MOV, MKV, WEBM, FLV
                    
                    **Max size:** 100 MB
                    """
                )
            
            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Predictions")
                
                predictions_output = gr.HTML(
                    label="Predictions",
                    value="<p style='color: #6b7280; text-align: center; padding: 50px;'>Upload a video and click Predict to see results</p>"
                )
                
                metadata_output = gr.HTML(
                    label="Metadata"
                )
        
        # Examples
        gr.Markdown("---")
        gr.Markdown("### 📚 Example Videos")
        
        # Check if example video exists
        example_video = str(settings.PROJECT_ROOT / "4088191-hd_1920_1080_25fps.mp4")
        if Path(example_video).exists():
            gr.Examples(
                examples=[
                    [example_video, 5]
                ],
                inputs=[video_input, top_k],
                label="Try this example"
            )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown(
            """
            **How it works:**
            1. Upload a video file
            2. The model extracts a 2-second clip from the middle
            3. X3D neural network analyzes the action
            4. Returns top-K predictions with confidence scores
            
            **Model Details:**
            - Architecture: X3D-XS (efficient 3D CNN)
            - Input: 16 frames × 224×224 pixels
            - Trained on: UCF101 / Kinetics-400 datasets
            """
        )
        
        # Event handlers
        predict_btn.click(
            fn=predict_video,
            inputs=[video_input, top_k],
            outputs=[predictions_output, metadata_output, video_input]
        )
    
    return demo


# ==========================================
# FastAPI Integration
# ==========================================

def mount_gradio_app(fastapi_app):
    """
    Mount Gradio vào FastAPI app
    
    Args:
        fastapi_app: FastAPI app instance
    
    Usage:
        from app.main import app
        from app.frontend.gradio_app import mount_gradio_app
        mount_gradio_app(app)
    """
    demo = create_gradio_interface()
    
    # Mount Gradio app vào /demo route
    fastapi_app = gr.mount_gradio_app(
        fastapi_app, 
        demo, 
        path="/demo"
    )
    
    print(f"✅ Gradio UI mounted at: http://{settings.HOST}:{settings.PORT}/demo")
    
    return fastapi_app


# ==========================================
# Standalone Mode
# ==========================================

if __name__ == "__main__":
    """
    Run Gradio standalone (without FastAPI)
    
    Usage: python -m app.frontend.gradio_app
    """
    print("=" * 60)
    print("🎨 Gradio UI - Standalone Mode")
    print("=" * 60)
    print("⚠️ Make sure FastAPI server is running on port 8000")
    print("=" * 60 + "\n")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
