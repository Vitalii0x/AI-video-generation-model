# AI Video Generation from Text

This project implements a state-of-the-art text-to-video generation model using transformer architecture and diffusion models.

## Features

- **Text-to-Video Generation**: Generate videos from natural language descriptions
- **Transformer Architecture**: Modern attention-based model for temporal consistency
- **Diffusion Process**: Denoising diffusion probabilistic models for high-quality generation
- **Modular Design**: Clean, extensible codebase with separate modules for different components
- **Training Pipeline**: Complete training loop with logging and checkpointing
- **Inference Script**: Easy-to-use inference for generating videos from text

## Architecture Overview

The model consists of several key components:

1. **Text Encoder**: CLIP-based text encoding for understanding text descriptions
2. **Video Encoder**: 3D CNN for processing video frames
3. **Temporal Transformer**: Attention mechanism for temporal consistency
4. **Diffusion Model**: Denoising process for video generation
5. **Decoder**: Convolutional layers for final video output

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/training_config.yaml
```

### Inference

```bash
python inference.py --text "A cat playing with a ball" --output_path output_video.mp4
```

## Project Structure

```
├── models/           # Model architectures
├── data/            # Data loading and preprocessing
├── training/        # Training utilities
├── configs/         # Configuration files
├── utils/           # Utility functions
├── train.py         # Main training script
├── inference.py     # Inference script
└── requirements.txt # Dependencies
```

## Model Architecture

The model uses a hybrid approach combining:
- **CLIP Text Encoder**: For text understanding
- **3D Convolutional Networks**: For spatial-temporal feature extraction
- **Transformer Blocks**: For temporal attention and consistency
- **Diffusion Process**: For high-quality video generation

## Training Data

The model can be trained on various video datasets:
- Kinetics-400/600/700
- UCF-101
- Custom video datasets

## Performance

- **Resolution**: 256x256, 512x512, or 1024x1024
- **Frame Rate**: 8-30 FPS
- **Duration**: 1-10 seconds
- **Quality**: High-fidelity video generation

## License

MIT License 