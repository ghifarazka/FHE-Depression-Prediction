# `ml-model`

This component develops a depression prediction model to serve as a case study for implementing Fully Homomorphic Encryption (FHE) in machine learning-based depression prediction systems. For the purpose of this project, this model would be referred to as `dass42-svr`, based on its source dataset and training algorithm.

The model was trained using the "Answers to the Depression Anxiety Stress Scales" dataset obtained from [OpenPsychometrics](https://openpsychometrics.org/_rawdata/). The dataset contained the answer to the 42 questions of the DASS-42 questionnaire created by Lovibond and Lovibond (1995). In DASS-42, 14 questions are used to measure depression, 14 to measure anxiety, and the other 14 to measure stress. In addition to that, the dataset includes a TIPI personality test, a demographic survey, as well as several technical data.

The goal of this part of the research was to provide:
- **[OUT-FOR]** A clear **data input format** for the user of the prediction model, based on the cleaned dataset
- **[OUT-PRE]** A **data preprocessor** object to preprocess the input data on the FHE implementation
- **[OUT-DAT]** Train-test **data** for evaluating the FHE implementation
- **[OUT-MOD]** Model parameter: **weights** and **bias**

It is to be noted that optimizing the model was NOT the focus of the research.
