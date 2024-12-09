# Inference for Monocular Depth Estimation and YOLO Integration

## Overview
This script combines Monocular Depth Estimation (MDE) with YOLO-based object detection to provide a robust pipeline for visual data analysis. The system supports inference from images, video streams, or webcam inputs and is designed for real-time or batch processing scenarios.

### Key Features:
- **Model Initialization**: Supports YOLO and Monocular Depth Estimation networks.
- **Flexible Input Options**: Accepts images, video files, or live streams.
- **Real-time Performance**: Optimized for GPU inference and half-precision computation.
- **Depth Estimation**: Outputs depth predictions alongside object detection.
- **Export to ONNX**: Converts the PyTorch model to ONNX format for deployment.

---

## Dependencies
Ensure the following dependencies are installed:
- Python 3.7+
- PyTorch
- NumPy
- OpenCV
- ONNX (optional, for export)
- ConfigParser

Install additional dependencies as required by `requirements.txt`.

---

## Configuration
The configuration file `cfg/mde.cfg` provides adjustable parameters for:
- YOLO properties (anchors, classes)
- Model freezing options (ResNet, MiDaS, YOLO, PlaneRCNN)

**Example Configuration Parameters:**
```ini
[yolo]
anchors = 10,13, 16,30, 33,23
classes = 80

[freeze]
resnet = True
midas = False
yolo = False
planercnn = True
```

---

## Script Breakdown

### 1. **Importing Modules**
Key imports include:
- `argparse` for command-line argument parsing.
- `torch` for model operations.
- `OpenCV` for image and video processing.

### 2. **Configuration Parsing**
The script reads and processes configuration parameters using `ConfigParser`.

### 3. **Model Initialization**
- Initializes the YOLO and Monocular Depth Estimation models.
- Supports loading weights from `.pt` files.
- Configures device compatibility for GPU or CPU.

### 4. **Detection Pipeline**
The main function `detect()` performs:
1. **Input Handling**: Processes image or video inputs.
2. **Model Inference**: Runs object detection and depth estimation.
3. **Post-Processing**:
   - Non-Maximum Suppression (NMS) for object detection.
   - Depth predictions saved as images.
4. **Output Generation**: Annotated images/videos saved to the output directory.

### 5. **Export to ONNX**
Enables exporting the model to ONNX format for deployment.

---

## Usage
Run the script with the following command-line arguments:

### Basic Example:
```bash
python main.py --source data/images --weights weights/last.pt --cfg cfg/yolov3-custom.cfg
```

### Command-Line Arguments:
- `--cfg`: Path to the YOLO configuration file.
- `--names`: Path to the class names file.
- `--weights`: Path to the model weights.
- `--source`: Input source (image, video, or webcam).
- `--output`: Directory to save output.
- `--img-size`: Image size for inference.
- `--conf-thres`: Confidence threshold for detection.
- `--iou-thres`: IOU threshold for NMS.
- `--device`: Specify `cpu` or `cuda`.

### Example with Webcam:
```bash
python main.py --source 0 --view-img
```

---

## Results
- **Output Directory**: Processed images and videos are saved in the specified output directory.
- **Annotations**: Object bounding boxes and depth maps are saved.
- **Logs**: Key performance metrics are printed to the console.

---

## Performance Optimization
- Use CUDA-enabled GPUs for faster inference.
- Enable half-precision inference (`--half`) for supported devices.
- Adjust image size (`--img-size`) based on the input resolution and available resources.

---

## Troubleshooting
### Common Issues:
- **ONNX Export Fails**: Ensure `onnx` is installed and compatible with PyTorch.
- **Missing Configurations**: Verify `mde.cfg` is properly formatted and accessible.
- **CUDA Errors**: Check GPU availability and PyTorch compatibility.

### Debugging:
Enable verbose logging by modifying print statements or using a debugger to trace issues.

---

## Future Work
- Add support for additional model architectures.
- Enhance real-time performance for high-resolution inputs.
- Integrate advanced post-processing techniques.

---
