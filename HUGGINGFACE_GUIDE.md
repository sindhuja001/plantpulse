# 🚀 Deploying to Hugging Face Spaces

This guide will walk you through hosting your **AI + Quantum Plant Disease Detection** app on Hugging Face (HF) Spaces using Streamlit.

## 1. Prepare Your Project Files
Ensure you have the following files ready in your directory:
- `app.py` (The main web application)
- `requirements.txt` (Dependencies)
- `plant_model.pkl` (The trained model)
- `images.npy` & `labels.npy` (Required if your app loads them at startup)

## 2. Create a Hugging Face Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Click **"Create new Space"**.
3. **Space Name**: Give it a name (e.g., `quantum-plant-detect`).
4. **SDK**: Select **Streamlit**.
5. **License**: Choose any (e.g., "Apache 2.0").
6. **Visibility**: Public (usually).
7. Click **Create Space**.

## 3. Upload Your Files
You can upload files directly through the browser or via Git. For a quick start:
1. Go to the **Files and versions** tab in your new Space.
2. Click **Add file** -> **Upload files**.
3. Select the files mentioned in Step 1.
4. Click **Commit changes**.

## 4. Set Up Your IBM Quantum Token (Secrets)
Since you shouldn't hardcode your private tokens:
1. Go to the **Settings** tab of your HF Space.
2. Scroll down to **Variables and secrets**.
3. Click **New secret**.
4. **Name**: `IBM_QUANTUM_TOKEN`
5. **Value**: Paste your IBM Quantum API Token.
6. Click **Save**.

The code I already prepared in `app.py` is configured to read from this secret:
```python
IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "fallback_token")
```

## 5. Metadata Configuration (README.md)
Hugging Face reads a `README.md` file at the root to configure the Space. You should create a `README.md` with the following content:

```yaml
---
title: AI + Quantum Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# AI + Quantum Plant Disease Detection
A hybrid application combining Scikit-Learn Random Forest and Qiskit Quantum Circuits for plant health verification.
```

## 6. Run and Test
- Once you commit the files, HF will begin building your Space.
- Check the **Logs** tab to see the progress.
- If everything installs correctly, your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`.

---

### Troubleshooting Common HF Errors
- **Memory Error**: If the model is too large, you might need a "T4 Small" or "L4" hardware upgrade on HF (though 26MB is very small and should run on the free tier).
- **ModuleNotFoundError**: Double-check that all packages in `requirements.txt` are spelled correctly.
- **Quantum Connection**: If the IBM backend is down, the app will automatically switch to the AI-only fallback as per our enhanced logic!
