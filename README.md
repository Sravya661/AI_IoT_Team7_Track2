# AI_IoT_Team7_Track2
Human Action Recognition using NVIDIA VLM Workflow

---

# Action Detection Using NVIDIA VLM

## Introduction
This project demonstrates action recognition using NVIDIA’s Neva-22B Vision-Language Model (VLM). By analyzing both real and synthetic videos, the system evaluates the accuracy of detecting specific human actions using a user-friendly Gradio interface.

This tool:
- Allows users to upload two videos (real and synthetic).
- Detects a specified human action (e.g., "sitting" or "walking").
- Computes detection accuracy for each video.
- Provides a real-time comparison of the results.

---

## Datasets

### 1. Real Video Dataset
We used the **Charades dataset**, a collection of real-world videos containing annotated human actions. It’s a well-known dataset used in human action recognition research.

### 2. Synthetic Video Dataset
Synthetic videos were sourced from the **Syn-Charades GenAI Synthetic Samples** dataset. These videos simulate human actions using computer-generated environments, offering a diverse set of examples for testing.

---

## **Setup and Installation**

### **1. Create a Virtual Environment**
Run the following commands to create and activate a virtual environment:

```bash
python -m venv vlm_env
vlm_env\Scripts\activate
```

### **2. Install Required Libraries**
Install Gradio and other dependencies:

```bash
pip install gradio moviepy requests pillow numpy nvidia-api
```
### **3. To Run The Code:**
Save the code in vlm_gradio_interface.py

```bash
python vlm_gradio_interface.py
```

---

## **NVIDIA Neva-22B API**

We integrated NVIDIA’s **Neva-22B Vision-Language Model** via the API hosted at **build.nvidia.com**. The API detects specified human actions in video frames.

### **API Details**
- **Endpoint:** `https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b`
- **Purpose:** Evaluate whether the given action is being performed in an input image (video frame).
- **Integration:** Frames from videos are converted to base64 images and sent to the API for analysis.
- **API_KEY:** `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`

---

## **Code Explanation**

1. **Frame Extraction:**
   - Videos are processed using the `moviepy` library to extract 16 evenly spaced frames.
   - Each frame represents a specific time interval of the video.

2. **Action Detection:**
   - Frames are encoded in base64 format and sent to the Neva-22B API for detection.
   - The API determines if the specified action is being performed in each frame.

3. **Accuracy Calculation:**
   - The accuracy is calculated as the percentage of frames where the action is correctly detected.

4. **Gradio Interface:**
   - The interface allows users to upload videos, specify actions, trim video lengths, and view results.

---

## **Interface Features**

1. **Video Upload:**
   - Users can upload a real video and a synthetic video for analysis.

2. **Action Input:**
   - Users can specify the action to detect (e.g., "sitting" or "walking").

3. **Trimming:**
   - Interactive sliders let users trim both videos to equal durations for a fair comparison.

4. **Results Display:**
   - Detection accuracy is displayed for both videos in real-time.

---

## **Usage Instructions**

1. Clone the repository and navigate to the project directory.
2. Activate the virtual environment:
   ```bash
   vlm_env\Scripts\activate
   ```
3. Run the Gradio application:
   ```bash
   python vlm_gradio_interface.py
   ```
4. Open the application in your browser and follow these steps:
   - Upload one real video and one synthetic video.
   - Specify the action to detect.
   - Trim the videos if necessary.
   - Click "Analyze Videos" to view the detection results.

---

## **Outputs**

### **Example Results**
- **Action:** (Sitting, walking, standing etc)
- **Real Video Accuracy:** 0-100%
- **Synthetic Video Accuracy:** 0-100%

These results show the system’s ability to detect actions accurately in both real and synthetic datasets.
---

## **Resources**

1. **Datasets:**
   - **Charades Dataset** for real videos.
   - **Syn-Charades GenAI Synthetic Samples** for synthetic videos.

2. **Dependencies:**
   - Python 3.7+
   - Gradio

3. **API:**
   - NVIDIA Neva-22B, obtained from **https://build.nvidia.com/explore/discover**.

---

## **Demo**

### **YouTube Video**
A step-by-step demonstration is available on YouTube (unlisted):  
(https://youtu.be/uYQYFXMieGk)

---

