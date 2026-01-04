# FHE Depression Prediction

This research project aims to address privacy concerns in machine learning–based depression prediction, where sensitive user data is typically exposed to the server during model inference. To achieve this goal, the research focuses on the following aspects:

- **Evaluation**: comparing prediction accuracy and computational overhead between plaintext (non-encrypted) and encrypted model inference.
- **Implementation**: developing a client–server system to demonstrate how privacy-preserving inference can be applied in a real-world setting.

This project was developed as part of an undergraduate Computer Science thesis. The full manuscript is available [here](https://repository.ipb.ac.id/).

> ⚠️ **DISCLAIMER**: This project is a **research demo** and **not** a clinical diagnostic tool. Do not use it for real clinical decisions.

## 🔎 Project Overview

The repository is organized into the following directories, each serving a specific purpose:

- `ml-model`: Contains the code used to develop the depression prediction model, including data preprocessing, model training, and the source dataset.
- `evaluation`: Contains scripts used to evaluate and compare prediction accuracy and computational overhead between plaintext (non-encrypted) and encrypted model inference.
- `client-side`: Contains the client-side implementation of the client–server system. This component collects user input, applies preprocessing, encrypts the data, sends it to the server, and decrypts the encrypted prediction result.
- `server-side`: Contains the server-side implementation of the client–server system. It receives encrypted input from the client, performs model inference directly on the encrypted data, and returns the encrypted result to the client.