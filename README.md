# Neural Storyteller

An advanced multimodal deep learning system that generates natural language descriptions for images using Sequence-to-Sequence (Seq2Seq) architecture with LSTM decoder.

##  Overview

Neural Storyteller is an image captioning system that bridges computer vision and natural language processing. It analyzes images and generates human-like descriptions, making it valuable for accessibility tools, content generation, and visual understanding applications.

### Key Highlights
- **Efficient Feature Extraction**: Pre-computed ResNet50 features for faster training
- **Advanced Decoder**: Multi-layer LSTM with attention mechanisms
- **Dual Inference Modes**: Greedy search and Beam search (width=5)
- **Production Ready**: Deployed with Gradio UI for real-time inference
- **Comprehensive Metrics**: Evaluated using BLEU-4, Precision, Recall, and F1-score

##  Architecture

The system follows a classic encoder-decoder architecture optimized for image captioning:

```
Image → ResNet50 → Feature Vector (2048-dim) → LSTM Encoder → Context
                                                      ↓
                                              LSTM Decoder → Caption
```

### Pipeline Flow
1. **Feature Extraction** (Offline): ResNet50 extracts 2048-dimensional feature vectors
2. **Encoding**: Linear projection maps features to hidden space (512-dim)
3. **Decoding**: Multi-layer LSTM generates captions word-by-word
4. **Inference**: Beam search selects optimal word sequences

##  Features

- ✅ **Cached Feature Extraction**: Pre-computed image features for efficient training
- ✅ **Smart Vocabulary Building**: Frequency-based filtering (min_freq=5)
- ✅ **Robust Preprocessing**: Text normalization and tokenization
- ✅ **Teacher Forcing**: Scheduled ratio decay (0.9 → 0.5)
- ✅ **Label Smoothing**: Reduces overfitting (ε=0.1)
- ✅ **Learning Rate Scheduling**: Adaptive learning with patience=2
- ✅ **Beam Search**: Top-5 sequence generation for better quality
- ✅ **Interactive UI**: Gradio-based deployment for easy testing

##  Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-storyteller.git
cd neural-storyteller

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
Pillow>=9.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
gradio>=4.0.0
nltk>=3.8.0
```

##  Dataset

**Flickr30k Dataset**
- **Images**: 31,783 photos
- **Captions**: 5 descriptions per image (158,915 total)
- **Source**: [Kaggle - Flickr30k](https://www.kaggle.com/datasets/adityajn105/flickr30k)

### Data Structure
```
flickr30k/
├── images/          # 31,783 JPG images
└── captions.txt     # CSV with image_id and captions
```

### Preprocessing Steps
1. **Caption Loading**: Parse CSV and organize by image
2. **Normalization**: Lowercase, remove special characters
3. **Tokenization**: Word-level splitting
4. **Vocabulary Building**: Filter words with freq < 5
5. **Encoding**: Convert text to token indices
6. **Special Tokens**: `<START>`, `<END>`, `<PAD>`, `<UNK>`

##  Model Architecture

### 1. Feature Extractor (ResNet50)
```python
Input: RGB Image (224×224)
↓
ResNet50 (Pre-trained on ImageNet)
↓
Output: Feature Vector (2048-dim)
```

### 2. LSTM Encoder
```python
class LSTMEncoder(nn.Module):
    - feature_projection: Linear(2048 → 512) + LayerNorm + ReLU
    - h_init: Linear(512 → 512) for hidden state
    - c_init: Linear(512 → 512) for cell state
    - num_layers: 2
    - dropout: 0.5
```

### 3. LSTM Decoder
```python
class LSTMDecoder(nn.Module):
    - embedding: Embedding(vocab_size, 512)
    - lstm: LSTM(512, 512, num_layers=2)
    - attention: Additive attention mechanism
    - output_projection: Linear(512 → vocab_size)
    - dropout: 0.5
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Image Feature Dim | 2048 |
| Hidden Dim | 512 |
| Embedding Dim | 512 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Dropout | 0.5 |
| Batch Size | 64 |
| Learning Rate | 0.0005 |
| Epochs | 30 |
| Label Smoothing | 0.1 |
| Beam Width | 5 |

##  Training

### Training Strategy
```python
# Loss Function
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx,
    label_smoothing=0.1
)

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

# LR Scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)
```

### Teacher Forcing Schedule
- **Start**: 90% (early epochs)
- **End**: 50% (late epochs)
- **Decay**: Linear interpolation

### Data Split
- **Training**: 80% (25,426 images)
- **Validation**: 10% (3,178 images)
- **Test**: 10% (3,179 images)

### Training on Kaggle
```bash
# Set accelerator to GPU T4 x2
# Upload pre-computed features: flickr30k_features.pkl
# Upload processed captions: flickr30k_captions_processed.pkl
# Run training cells
```

## Inference

### Greedy Search
```python
def greedy_search(encoder, decoder, image_features, max_length=50):
    # Generate caption word-by-word
    # Select highest probability token at each step
    return generated_caption
```

### Beam Search
```python
def beam_search(encoder, decoder, image_features, beam_width=5):
    # Maintain top-k sequences at each step
    # Score based on log probabilities
    # Return highest scoring complete sequence
    return best_caption
```

##  Results

### Quantitative Metrics
| Metric | Score |
|--------|-------|
| **BLEU-4** | 0.XX |
| **Precision** | 0.XX |
| **Recall** | 0.XX |
| **F1-Score** | 0.XX |

### Sample Predictions

**Example 1**
- **Image**: Beach scene with people
- **Ground Truth**: "A group of people playing volleyball on the beach"
- **Generated**: "A group of people on a beach"

**Example 2**
- **Image**: Dog in park
- **Ground Truth**: "A brown dog running through the grass"
- **Generated**: "A dog is running in the grass"

### Training Curves
- Loss decreases consistently over 30 epochs
- Validation loss stabilizes around epoch 20
- No significant overfitting observed

##  Deployment

### Gradio Interface
```python
import gradio as gr

def caption_image(image):
    # Extract features
    # Generate caption using beam search
    return caption

demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Neural Storyteller - Image Captioning",
    description="Upload an image to generate a caption"
)

demo.launch()
```

### Running the App
```bash
python app.py
# Access at http://localhost:7860
```

##  Technologies Used

- **Deep Learning**: PyTorch 2.0+
- **Computer Vision**: torchvision, ResNet50
- **NLP**: NLTK, custom tokenization
- **Deployment**: Gradio
- **Visualization**: Matplotlib
- **Data Processing**: NumPy, Pandas, Pickle
- **Platform**: Kaggle (GPU T4 x2)

##  Project Structure

```
neural-storyteller/
├── notebooks/
│   └── neural_storyteller.ipynb       # Main notebook
├── models/
│   └── best_model.pth                 # Trained weights
├── data/
│   ├── flickr30k_features.pkl         # Cached features
│   └── flickr30k_captions_processed.pkl  # Processed captions
├── app.py                             # Gradio deployment
├── requirements.txt                   # Dependencies
├── README.md                          # Documentation
└── LICENSE                            # MIT License
```

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


- [Medium Article](https://medium.com/@yourusername/article)
- [Deployment Gradio](https://74a8359d23825e519a.gradio.live/)


