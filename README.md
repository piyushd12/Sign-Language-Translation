# Sign Language Translator

This project enables real-time translation of sign language gestures into text using deep learning and hand landmark detection. It allows users to record custom gesture data, train a gesture recognition model using LSTM networks, and run real-time inference via webcam. Built using MediaPipe and TensorFlow, this system bridges communication gaps for the hearing-impaired community.

**Supports any sign language** — The user records their own gestures, so the system is fully generalizable to any custom or regional sign language.

---

## Key Features

- Real-time gesture recognition using webcam input
- Create your own dataset with custom signs
- Generalized to any sign language
- LSTM-based gesture recognition model
- Modular structure with separate scripts for training and inference
- Export trained models to TensorFlow Lite for lightweight deployment

---

## Project Structure

```
SLR_Project/
│
├── data/                         # Stored keypoints for each gesture
├── models/                       # Trained .keras and .tflite models
├── utils/
│   └── mediapipe_utils.py        # Functions to process MediaPipe output
│
├── Scripts/
│   ├── train_model.py            # Trains the LSTM model
│   ├── run_inference.py          # Runs real-time gesture inference
│   └── convert_to_tflite.py      # Converts trained model to .tflite
│
├── data_collection.py            # Collects gesture data with webcam
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

- Python ≥ 3.6
- TensorFlow ≥ 2.10
- OpenCV-Python
- MediaPipe
- NumPy
- Keyboard

---

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/SLR_Project.git
   cd SLR_Project
   ```

2. **Create and Activate a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Use

### 1. **Add New Signs**
Open `data_collection.py` and edit the `actions` list to add new sign words you want to recognize.

```python
actions = np.array(['hello', 'thank_you', 'yes', 'no'])
```

These can be from **any sign language** – the model learns what *you* record.

---

### 2. **Collect Gesture Data**
Run the script to record sign gesture sequences for each action:

```bash
python data_collection.py
```

Each gesture will be recorded as a sequence of hand landmarks using your webcam. Move your hand slightly between each recording to ensure data diversity.

---

### 3. **Train the Model**
After collecting data, train your LSTM model using:

```bash
python Scripts/train_model.py
```

This script will save the trained model to `models/SLR.keras`.

---

### 4. **Run Real-Time Inference**
Use your webcam to detect and translate signs in real-time:

```bash
python Scripts/run_inference.py
```

This will continuously show the translated text on the screen based on your gestures.

---

### 5. **Convert to TensorFlow Lite (Optional)**
To deploy the model on edge devices:

```bash
python Scripts/convert_to_tflite.py
```

This will generate a `.tflite` version of the trained model in the `models/` directory.

---

## 📄 .gitignore

```
__pycache__/
venv/
data/
models/
```

---

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) by Google for real-time hand tracking.
- [TensorFlow](https://www.tensorflow.org/) for building and training the neural network
- Inspiration from various sign language recognition research papers and open-source projects.
---

**License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

