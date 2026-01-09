# FHE Depression Prediction

This research project aims to address privacy concerns in machine learning–based depression prediction, where sensitive user data is typically exposed to the server during model inference. To achieve this goal, the research focuses on the following aspects:

- **Evaluation**: Compare prediction accuracy and computational overhead between plaintext (non-encrypted) and encrypted model inference.
- **Implementation**: Develop a client–server system to demonstrate how privacy-preserving inference can be applied in a real-world setting.

> ⚠️ **DISCLAIMER**: This project is a **research demo** and **not** a clinical diagnostic tool. Do not use it for real clinical decisions.

## 🔎 Project Structure

The repository is organized into the following directories, each serving a specific purpose:

- `ml-model/`: Trains a depression prediction model using a source dataset.
- `fhe-evaluation/`: Evaluates the feasibility of implementing FHE by comparing model inference performance between encrypted and non-encrypted data.
- `client-side/`: Implements FHE on the client side. It collects user input, applies preprocessing, encrypts the data, sends it to the server, and decrypts the encrypted prediction result.
- `server-side/`: Implements FHE on the server side. It receives encrypted input from the client, performs model inference directly on the encrypted data, and returns the encrypted result to the client.


## ▶️ How to Run

### 1: Real World Scenario

### 2: Testing from the Repository

## 🤝 Collaboration

This project was developed as part of an undergraduate Computer Science thesis at IPB University. The full manuscript is available [here](https://repository.ipb.ac.id/).

For academic collaboration or inquiries, please contact: 
- Muhammad Ghifar Azka Nurhadi (m.ghifarazka@gmail.com)