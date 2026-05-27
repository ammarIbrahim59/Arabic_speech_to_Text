# NN Project 1 — Arabic Speech-to-Text with Emotion Detection (Bonus)

This project is a notebook-based pipeline for Arabic ASR using OpenAI Whisper, with evaluation (WER/CER), model-size experiments, a noise-robustness test, and an optional emotion-detection module. It also includes a Gradio demo for interactive transcription, keyword spotting, and summarization.

## Contents

- `PROJECTNN1_with_emotion_BONUS.ipynb` — full end-to-end notebook

## Features

- Arabic speech recognition with Whisper (tiny/small/medium/large-v3)
- Dataset preparation and train/val/test split
- Batch transcription with WER/CER evaluation
- Model-size comparison experiment
- Robustness test with additive Gaussian noise
- Emotion detection via acoustic features + MLP classifier
- Gradio demo UI (transcription, keywords, optional summary, emotion)

## Dataset

The notebook is configured for the **Mozilla Common Voice Arabic** dataset hosted on Kaggle:

- Example Kaggle path: `/kaggle/input/datasets/mayarjao/arabic-tts`
- Expected structure:
  - `arabic_tts/metadata.csv`
  - `arabic_tts/wavs/*.wav`

If you run locally, update `DATA_DIR` in the notebook to your dataset location.

## Requirements

- Python 3.9+
- `ffmpeg` installed on the system
- Python packages (installed in Cell 1 / ED-1):
  - `openai-whisper`
  - `jiwer`
  - `gradio`
  - `librosa`
  - `soundfile`
  - `kaggle`
  - `scikit-learn`
  - `imbalanced-learn`
  - `joblib`

## How to Run

1. Open `PROJECTNN1_with_emotion_BONUS.ipynb` in Colab or Kaggle.
2. (Optional) Adjust `DATA_DIR` to your dataset path.
3. Run cells in order:
   - Install dependencies
   - Load dataset + build splits
   - Load Whisper and run batch inference
   - Evaluate WER/CER and experiments
   - Train emotion model (bonus) and run the Gradio demo

## Outputs

- Train/val/test splits saved to `/content/train.csv`, `/content/val.csv`, `/content/test.csv`
- Predictions saved to `/content/predictions.csv`
- Emotion model artifacts saved to `/kaggle/working/`:
  - `emotion_clf.joblib`
  - `emotion_scaler.joblib`
  - `emotion_le.joblib`

## Notes

- For best accuracy, use `MODEL_SIZE = "medium"` (or `"large-v3"` if GPU memory allows).
- The Gradio demo launches a local UI and can optionally generate a public share link.
