# ESP32 Bird Audio Classifier

An ESP32-based TinyML system designed to identify local bird species from their vocalizations using real-time audio capture, DSP preprocessing, and on-device machine learning inference.

> [!WARNING]
> **Work in Progress — Active Development**
>
> This project is currently in the early planning and development phase. The DSP pipeline, model architecture, dataset, hardware configuration, and firmware implementation are still being evaluated and may change significantly as development progresses.

---

## Overview

The **ESP32 Bird Audio Classifier** is an embedded machine learning project exploring whether a low-cost microcontroller can perform real-time bird audio classification directly on-device.

The system will capture audio using an I2S microphone, transform the raw audio into features suitable for machine learning, and run a lightweight classification model on the ESP32 using a TinyML framework such as **TensorFlow Lite for Microcontrollers (TFLM)**.

The long-term goal is to create a portable device capable of recognizing local bird species from their vocalizations without requiring a continuous connection to a computer or cloud service.

This project combines:

* Embedded systems
* Digital signal processing
* Audio feature extraction
* Machine learning
* TinyML
* Microcontroller optimization
* Real-world environmental sensing

Because the project is still in its early stages, the exact implementation is intentionally left open while the feasibility of different approaches is evaluated.

---

## System Architecture

The planned system will follow a basic audio-to-inference pipeline:

```text
┌──────────────────────┐
│     Audio Input      │
│                      │
│   INMP441 I2S Mic    │
└──────────┬───────────┘
           │
           │ Raw PCM Audio
           ▼
┌──────────────────────┐
│    DSP Pipeline      │
│                      │
│ Filtering / Framing  │
│ Windowing / FFT      │
└──────────┬───────────┘
           │
           │ Spectral Features
           ▼
┌──────────────────────┐
│   MFCC Extraction    │
│                      │
│ Mel Filterbank       │
│ Cepstral Coefficients│
└──────────┬───────────┘
           │
           │ Feature Vector
           ▼
┌──────────────────────┐
│     TinyML Model     │
│                      │
│  CNN / MLP Classifier│
│  TensorFlow Lite     │
│     Micro            │
└──────────┬───────────┘
           │
           │ Prediction
           ▼
┌──────────────────────┐
│       Output         │
│                      │
│ Bird Species +       │
│ Confidence Score     │
└──────────────────────┘
```

The architecture above represents the **planned direction**, not a finalized implementation.

---

## Hardware

### Expected Hardware

| Component                  | Purpose                                        | Status   |
| -------------------------- | ---------------------------------------------- | -------- |
| **ESP32**                  | Main microcontroller and ML inference platform | Planned  |
| **INMP441 I2S Microphone** | Digital audio capture                          | Planned  |
| **OLED Display**           | Display predicted species and confidence       | Optional |
| **SD Card Module**         | Store audio samples or predictions             | Optional |
| **Battery**                | Portable power source                          | Optional |

Hardware selections may change during development depending on audio quality, memory requirements, power consumption, and physical deployment considerations.

The initial prototype will prioritize **audio acquisition and on-device inference** before adding optional peripherals.

---

## DSP Pipeline (Planned)

The raw microphone signal will need to be transformed into a compact representation suitable for the machine learning model.

The initial DSP pipeline is expected to investigate:

```text
Raw Audio
    │
    ▼
Preprocessing
    │
    ▼
Framing / Windowing
    │
    ▼
FFT
    │
    ▼
Mel Filterbank
    │
    ▼
MFCC Extraction
    │
    ▼
Feature Vector
    │
    ▼
TinyML Model
```

### Planned Processing Steps

#### 1. Audio Sampling

The microphone will capture digital PCM audio through the ESP32's I2S interface.

A sampling rate appropriate for bird vocalizations will be selected experimentally. The exact sampling rate has **not yet been finalized**.

#### 2. Windowing

The continuous audio stream will be divided into short overlapping frames.

Window size and overlap will be evaluated based on:

* Bird vocalization characteristics
* Temporal resolution
* Computational cost
* ESP32 memory constraints

#### 3. FFT

A Fast Fourier Transform (FFT) will be used to convert each time-domain audio frame into its frequency-domain representation.

The FFT configuration is still under investigation and may be adjusted based on the target bird species and available computational resources.

#### 4. MFCC Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are currently the primary candidate for representing the audio signal.

MFCC extraction may include:

* Power spectrum calculation
* Mel-scale filterbank
* Logarithmic compression
* Discrete cosine transform

The final number of coefficients, filterbank configuration, frame size, and normalization strategy have **not yet been finalized**.

> **Note:** MFCCs are currently the planned feature representation, but alternative audio features may be evaluated if they provide better classification performance or lower computational cost.

---

## Model Architecture (Planned)

The project will use a lightweight machine learning model suitable for deployment on a resource-constrained microcontroller.

The initial candidates are:

* Small Convolutional Neural Network (CNN)
* Multi-Layer Perceptron (MLP)
* Other compact architectures if experimentation suggests a better approach

A potential high-level CNN architecture could look like:

```text
MFCC Features
      │
      ▼
┌──────────────┐
│ Conv Layer   │
└──────┬───────┘
       ▼
┌──────────────┐
│ Pooling      │
└──────┬───────┘
       ▼
┌──────────────┐
│ Conv Layer   │
└──────┬───────┘
       ▼
┌──────────────┐
│ Dense Layer  │
└──────┬───────┘
       ▼
┌──────────────┐
│ Softmax      │
└──────┬───────┘
       ▼
 Bird Species
```

This architecture is **illustrative only**.

The final model will be determined through experimentation with:

* Classification accuracy
* Model size
* RAM usage
* Flash usage
* Inference latency
* Power consumption
* Robustness to environmental noise

Quantization and other model compression techniques will also be investigated to make deployment practical on the ESP32.

---

## Dataset

Dataset development is one of the major components of this project.

The planned dataset will consist primarily of **local bird recordings**, with additional data potentially obtained from publicly available bird audio datasets.

### Potential Data Sources

* Locally recorded bird whistles
* Public bird audio datasets
* Field recordings
* Augmented audio samples

### Data Augmentation

To improve model robustness, training data may be augmented using techniques such as:

* Background noise injection
* Volume variation
* Time shifting
* Time stretching
* Frequency shifting
* Other audio transformations

The goal is to help the model distinguish bird vocalizations under realistic outdoor conditions rather than only clean recordings.

> **Current Status:** The initial five-species Xeno-canto dataset is complete
> (75 recordings per species). The collection workflow below can curate future
> species without changing completed datasets.

### Xeno-canto collection

The repository includes a reproducible collector for original Xeno-canto audio.
It selects A/B-quality, non-playback recordings of at least five seconds, with a
fixed default seed and recordist-first diversity selection. Existing completed
datasets are validated and left unchanged.

```powershell
& .\.venv\Scripts\python.exe .\collect_species.py "Blue Jay"
```

Install the Python dependencies with `python -m pip install -r requirements.txt`.
The Xeno-canto key remains in the ignored `.env` file as
`XENO_CANTO_API_KEY`; the collector loads it without printing or passing the
key on the command line. An already-exported environment variable takes
precedence.

`data/raw/<normalized_species>/metadata_only.csv` is the canonical Xeno-canto
candidate metadata file. The collector preserves the original source fields,
including `id`, `url`, `file`, `file-name`, `en`, quality, playback, recordist,
location, type, and duration. It writes the selected rows to
`data/metadata/<normalized_species>_selected.csv` and stores untouched original
audio at `data/raw/<normalized_species>/<gen>_<sp>/<file-name>`.

Optional arguments are `--recordings`, `--workers`, `--seed`, and
`--max-replacement-rounds`. The collector rejects a species before audio
download if fewer than the requested number of eligible recordings exist. It
reports diversity as a warning rather than a hard threshold, replaces failed
downloads up to a finite limit without retrying a previously failed ID as a
replacement, and validates exact `file-name` values at the end. It never
infers a recording ID from a filename or alters raw audio.

---

## Firmware Overview

The ESP32 firmware will eventually handle the complete embedded inference pipeline.

The planned firmware flow is:

```text
I2S Microphone
      │
      ▼
Audio Buffer
      │
      ▼
DSP Processing
      │
      ▼
MFCC Extraction
      │
      ▼
Feature Buffer
      │
      ▼
TinyML Inference
      │
      ▼
Classification Result
      │
      ▼
Display / Serial / Storage
```

The firmware will be responsible for:

1. Configuring the I2S microphone interface
2. Continuously capturing audio samples
3. Managing audio buffers
4. Performing DSP preprocessing
5. Extracting model features
6. Running the TinyML model
7. Processing classification results
8. Outputting predictions

Memory management and computational efficiency will be important considerations because the ESP32 has significantly fewer resources than a desktop or cloud computing environment.

Firmware implementation is currently in the **early planning stage**.

---

## Repository Structure

The repository structure is expected to evolve as development progresses.

A potential organization is:

```text
esp32-bird-audio-classifier/
├── firmware/
│   ├── src/
│   ├── include/
│   └── platformio.ini
│
├── dsp/
│   ├── preprocessing/
│   └── feature_extraction/
│
├── model/
│   ├── training/
│   ├── evaluation/
│   └── exported/
│
├── dataset/
│   └── README.md
│
├── notebooks/
│
├── tests/
│
└── README.md
```

The final structure will depend on the development workflow and tools selected.

---

## Setup

> **Note:** Setup instructions are preliminary and will be updated as the project implementation stabilizes.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/esp32-bird-audio-classifier.git
cd esp32-bird-audio-classifier
```

### 2. Set Up the Development Environment

The project is expected to use a combination of:

* ESP32 development tools
* Python for dataset processing and model training
* TensorFlow / TensorFlow Lite
* TensorFlow Lite for Microcontrollers
* ESP-IDF or another ESP32 development framework

Exact dependencies will be documented once the implementation is finalized.

### 3. Prepare the Dataset

Place collected and processed audio data into the appropriate dataset directory.

The training pipeline will eventually handle:

```text
Audio Recordings
      │
      ▼
Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
```

### 4. Train the Model

The training process will eventually generate a lightweight model suitable for conversion to a microcontroller-compatible format.

Example workflow:

```bash
python train.py
python evaluate.py
python export_model.py
```

> These commands are placeholders and will be replaced with the actual training pipeline once implemented.

### 5. Flash the ESP32

Once the firmware is implemented and the model has been integrated:

```bash
idf.py build
idf.py flash
idf.py monitor
```

The exact flashing workflow may change depending on whether the project uses ESP-IDF, Arduino, PlatformIO, or another development environment.

---

## Demo

### Coming Soon

A demonstration will be added once the first functional prototype is operational.

Planned demonstrations include:

* Real-time bird audio detection
* Predicted bird species
* Classification confidence
* ESP32 inference latency
* Performance under outdoor background noise
* Optional OLED output

---

## Roadmap

### Phase 1 — Planning & Hardware

* [ ] Select ESP32 development board
* [ ] Validate INMP441 microphone interface
* [ ] Define initial target bird species
* [ ] Determine initial recording requirements
* [ ] Establish development environment

### Phase 2 — Dataset

* [ ] Collect local bird audio
* [ ] Investigate public bird audio datasets
* [ ] Clean and label recordings
* [ ] Implement data augmentation
* [ ] Create training / validation / test splits

### Phase 3 — DSP

* [ ] Implement audio preprocessing
* [ ] Implement windowing
* [ ] Implement FFT
* [ ] Implement MFCC extraction
* [ ] Evaluate feature representations
* [ ] Benchmark DSP performance on ESP32

### Phase 4 — Machine Learning

* [ ] Train baseline classifier
* [ ] Evaluate CNN architecture
* [ ] Evaluate MLP architecture
* [ ] Optimize model size
* [ ] Investigate quantization
* [ ] Convert model for TinyML deployment

### Phase 5 — Firmware

* [ ] Implement I2S audio capture
* [ ] Implement audio buffering
* [ ] Integrate DSP pipeline
* [ ] Integrate MFCC extraction
* [ ] Integrate TinyML inference
* [ ] Implement classification output

### Phase 6 — Optimization

* [ ] Optimize RAM usage
* [ ] Optimize flash usage
* [ ] Reduce inference latency
* [ ] Measure power consumption
* [ ] Improve robustness to environmental noise

### Phase 7 — Field Testing

* [ ] Conduct outdoor field tests
* [ ] Evaluate classification accuracy
* [ ] Test different environmental conditions
* [ ] Analyze false positives and false negatives
* [ ] Refine dataset and model

### Phase 8 — Release

* [ ] Finalize hardware configuration
* [ ] Finalize firmware
* [ ] Finalize model
* [ ] Document reproducible setup
* [ ] Release **v1.0**

---

## Current Development Status

The project is currently **not a finished bird identification device**.

The primary objectives at this stage are to determine:

* Which audio features work best for the target birds
* Whether MFCCs provide sufficient classification performance
* What model architecture can run efficiently on the ESP32
* How much memory and computation the complete pipeline requires
* How well the system performs with real-world environmental noise

As development progresses, this README will be updated to reflect measured hardware, DSP, model, and firmware implementations rather than planned specifications.

---

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for the full license text.

---

## Acknowledgements

This project builds upon the work of the open-source embedded ML and audio processing communities.

Potential tools, frameworks, and resources include:

* **TensorFlow Lite for Microcontrollers** — Embedded machine learning inference
* **TensorFlow** — Model development and training
* **ESP-IDF** — ESP32 development framework
* **Espressif** — ESP32 hardware and software ecosystem
* **Bird audio datasets and recording communities** — Training and evaluation data
* Open-source DSP and audio processing libraries

Additional datasets, libraries, and research resources will be credited as they are incorporated into the project.

---

## Project Goals

The long-term goal is to develop a compact, low-power, standalone embedded system capable of recognizing bird species from their vocalizations using on-device machine learning.

The project will serve as an exploration of the intersection between:

**Embedded Systems × DSP × Machine Learning × TinyML × Environmental Sensing**

More importantly, the project will investigate the practical constraints involved in taking an ML pipeline from a dataset and Python environment to a resource-constrained microcontroller running in the real world.
