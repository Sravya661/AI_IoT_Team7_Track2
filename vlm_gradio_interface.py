import requests
import base64
from moviepy.editor import VideoFileClip
from PIL import Image
from io import BytesIO
import gradio as gr

API_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"  
API_KEY = "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"  

def extract_frames_from_video(video_file, num_frames=16):
    video_clip = VideoFileClip(video_file)
    total_duration = video_clip.duration
    frames = [
        video_clip.get_frame(i * total_duration / num_frames) for i in range(num_frames)
    ]
    return [Image.fromarray(frame) for frame in frames]

def encode_image_to_base64(image_frame):
    buffer = BytesIO()
    image_frame.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def query_action_detection_model(image_b64, action):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Is someone performing "{action}" in this image? <img src="data:image/png;base64,{image_b64}" />',
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": False,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if 'choices' in result:
            return "yes" in result['choices'][0]['message']['content'].lower()
    else:
        print(f"API request failed: {response.status_code} - {response.text}")
    return False

def compute_accuracy_for_video(video_frames, action):
    successful_detections = 0
    for frame in video_frames:
        encoded_image = encode_image_to_base64(frame)
        if query_action_detection_model(encoded_image, action):
            successful_detections += 1
    return (successful_detections / len(video_frames)) * 100  

def analyze_video_action_accuracy(video1_file, video2_file, action):
    try:
        
        frames_from_video1 = extract_frames_from_video(video1_file)
        frames_from_video2 = extract_frames_from_video(video2_file)

        accuracy_video1 = compute_accuracy_for_video(frames_from_video1, action)
        accuracy_video2 = compute_accuracy_for_video(frames_from_video2, action)

        return (
            f"Video 1 '{action}' Detection Accuracy: {accuracy_video1:.2f}%",
            f"Video 2 '{action}' Detection Accuracy: {accuracy_video2:.2f}%"
        )
    except Exception as e:
        return f"Error occurred: {str(e)}", None

with gr.Blocks() as app:
    gr.Markdown("# Action Detection with NVIDIA NEVA")
    gr.Markdown("Upload two videos and specify an action. This tool will calculate the action detection accuracy for each video.")
    
    with gr.Row():
        real_video = gr.Video(label="Upload Real Video")
        synthetic_video = gr.Video(label="Upload Synthetic Video")
    
    action = gr.Textbox(label="Specify Action (e.g., sitting, walking)")
    
    with gr.Row():
        real_analysis = gr.Textbox(label="Real Video Analysis")
        synthetic_analysis = gr.Textbox(label="Synthetic Video Analysis")
    
    analyze_button = gr.Button("Analyze")
    analyze_button.click(
        analyze_video_action_accuracy, 
        inputs=[real_video, synthetic_video, action], 
        outputs=[real_analysis, synthetic_analysis]
    )

if __name__ == "__main__":
    app.launch()
