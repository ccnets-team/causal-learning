# Causal Learning Framework by CCNets

[![Static Badge](https://img.shields.io/badge/Release-v1.1.1-%25%2300FF00)](https://github.com/ccnets-team/causal-learning)
[![Static Badge](https://img.shields.io/badge/LICENSE-DUAL-%23512BD4)](./LICENSE/)
[![Static Badge](https://img.shields.io/badge/Python-3.9.18-%233776AB)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/PyTorch-2.3.0-%23EE4C2C)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97GPT%20model-Hugging%20Face-%23FF9D0B)](https://huggingface.co/gpt2)
[![Static Badge](https://img.shields.io/badge/CCNets-LinkedIn-%230A66C2)](https://www.linkedin.com/company/ccnets/posts/?feedView=all)
[![Static Badge](https://img.shields.io/badge/Patent-Google-%234285F4)](https://patents.google.com/patent/WO2022164299A1/en)
[![Static Badge](https://img.shields.io/badge/Patent-KR-F70505)](https://doi.org/10.8080/1020237037422)

# Table of Contents

- [🎈 **Overview**](#🎈-overview)
- [❗️ **Dependencies**](#❗️-dependencies)
- [📥 **Installation**](#📥-installation)
- [📖 **Features**](#📖-features)
- [🔎 **API Documentation**](#🔎-api-documentation)
- [🐞 **Issue Reporting Policy**](#🐞-issue-reporting-policy)
- [✉️ **LICENSE**](#✉️-LICENSE)

# 🎈 Overview

## **Introduction**

CCNet is a new ML framework designed to uncover and model causal relationships between input observations 𝑋 and labels 𝑦 in datasets. This framework employs three neural networks to form a cooperative structure that enables bidirectional inference between input 𝑋 and target 𝑦.

This framework learns an explanation vector 𝑒 that transforms the associations observed between inputs 𝑋 and outputs y into a causal relationship. Here, 𝑒 and 𝑦 are considered comprehensive factors instrumental in generating 𝑋.

## **Key Capabilities**
CCNet consists of three neural networks having role of —Explainer, Reasoner, and Producer—to execute six fundamental operations in machine learning:

- `Explain`: Extracts key features from input observations (X) to form an Explanation Vector(e), which captures the essential aspects of the data.

- `Reason`: Utilizes the Explanation Vector alongside Input Observations (X) to infer the associated label (y) of the observation.

- `Produce`: Generates new data based on specified conditions and the derived Explanation(e), enabling the creation of data instances that resemble authentic observations.

- `Infer`: Determines outputs (y') from input data (X) by integrating the insights from both the Explainer and Reasoner networks, providing a prediction or outcome based on learned patterns.

- `Generate`: Constructs new data using the Explanation Vector with randomly sampled conditions (y), allowing for the exploration of possible data scenarios that could occur under different circumstances.

- `Reconstruct`: Rebuilds input data (X) by sequentially explaining, reasoning, and then producing the output, effectively creating a reconstructed version of the input based on the network's understanding and reasoning.

<br>

# ❗️ ****Dependencies****

```python
conda create -name ccnets python=3.9.18
conda activate ccnets
pip install jupyter
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pandas==2.2.2
pip install scikit-learn==1.1.0
pip install transformers==4.40.2
pip install tensorboard==2.16.2
pip install ipywidgets==8.1.2
```

# 📥 **Installation**

- Steps to install the framework.
- Note: Ensure you have the required dependencies installed as listed in the "Dependencies" section above.

**Installation Steps:**

1. Clone the repository:
    
    ```bash
    git clone https://github.com/ccnets-team/causal-learning.git
    ```
    
2. Navigate to the directory and install the required packages:
    
    ```bash
    cd ccnets

    pip install -r requirements.txt
    
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
# 📖 **Features**

### Dual Cooperative Network Architecture in CCNets API:

CCNets harnesses a dual cooperative network structure, each designed to optimize the processing and analysis of complex datasets

<p align="center">
  <img src="https://github.com/ccnets-team/causal-learning/assets/95277008/7b66bf01-d917-419d-8979-b8693df67a5d" alt="two_nets" width="700">
</p>

- **Core Cooperative Network (GPT-based)**

    At the core of CCNets’ architecture is a Cooperative Network configured with GPT models. These models are optimized for sequence learning and label comprehension within extensive datasets. They serve as the central processing unit, adept at handling and interpreting sequence data and extracting meaningful insights from complex patterns.

- **Encoder Cooperative Network**

    The Encoder Cooperative Network is engineered to preprocess and transform raw input data into a format that significantly enhances the analytical capabilities of the Core Cooperative Network. This network specializes in adapting raw data into a structured, analyzable form. For example, in handling image data, this network translates visual information into a trajectory format.


<br>
<br>

### Example Categories :

| Label⬇️\Data➡️ Type | Tabular                                                                                                                                                                                                                                           | Time-Series                                                                                                                | Image                                                                                                                                                                                                                                                                                                                                                                                                        | Text                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Binary Classification**       | [- Creditcard Fraud Dataset](examples/ccnet_generate_credit_card_fraud_dataset.ipynb)<br>[- Airline Customer Satisfaction](examples/beyond_correlation_how_causality_ensures_perfect_airline_customer_satisfaction.ipynb) |                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                   |
| **Multi Class Classification**  |                                                                                                                                                                                                                                                    | [- EEG performance](examples/ccnet_gpt_performance_eeg.ipynb)<br>[- Mice DNA Prediction](examples/ccnet_prediction_of_mice_dna_sequences.ipynb)<br>[- Korean Stock Prediction](examples/predicting_korean_stock_prices_using_ccnet.ipynb) | [- Crop & Weed Classification](examples/Annotated_food_crops_and_weed_images.ipynb)<br>[- CelebA CCNet](examples/ccnet_encoder_to_core_model_using_celebA.ipynb)<br>[- Generate Handwritten Digits](examples/ccnet_generate_your_handwritten_digits_from_any_number.ipynb)<br>[- Old Painting to Photo](examples/ccnet_transform_old_painting_to_real_photo.ipynb)<br>[- Gender Expression Reshape](examples/let_you_smile_by_reshaping_gender_expressions.ipynb)<br>[- Animal Match](examples/discover_your_animal_match_with_ccnet.ipynb)<br>[- Recycling Classification](examples/recycling_waste_image_classification_with_ccnet.ipynb) |                                                                                   |
| **Multi Label Classification**  |                                                                                                                                                                                                                                                    |                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                   |
| **Single Regression**          |                                                                                                                                                                                                                                                    | [- Climate Prediction](examples/ccnet_daily_climate_prediction.ipynb)<br>[- Energy Prediction](examples/ccnets_renewable_energy_prediction.ipynb) |                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                   |
| **Compositional Regression**   | [- Drug Composition Prediction](examples/sparse_drug_composition_prediction_from_NIR_dataset_using_ccnet.ipynb)                                                                                                                                     |                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                   |
| **Ordinal Regression**         |                                                                                                                                                                                                                                                    |                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                               | [- Amazon Reviews NLP](examples/ccnet_using_pretrained_nlp_for_amazon_reviews.ipynb) |
| **Generation**                 | [- Synthesize Fraud Data](examples/ccnet_synthesize_credit_card_fraud_dataset.ipynb)                                                                                                                                                               |                                                                                                                            | [- Make Pokemon](examples/ccnet_pick_your_own_pokemon.ipynb)<br>[- Celeba encoding CCNet](examples/celeba_encoding_ccnet.ipynb)                                                                                                                                                                                                                                                                               |                                                                                   |

<br>
<br>

# 🔎 **API Documentation**

- We're currently in the process of building our official documentation webpage to better assist you. In the meantime, if you have any specific questions or need clarifications, feel free to reach out through our other support channels. We appreciate your patience and understanding!

<br>
<br>

# 🐞 **Issue Reporting Policy**
<details>
<summary>More Information</summary>
Thank you for taking the time to report issues and provide feedback. This helps improve our project for everyone! To ensure that your issue is handled efficiently, please follow the guidelines below:

### **1. Choose the Right Template:**

We provide three issue templates to streamline the reporting process:

1. **Bug Report**: Use this template if you've found a bug or something isn't working as expected. Please provide as much detail as possible to help us reproduce and fix the bug.
2. **Feature Request**: If you have an idea for a new feature or think something could be improved, this is the template to use. Describe the feature, its benefits, and how you envision it.
3. **Custom Issue Template**: For all other issues or general feedback, use this template. Make sure to provide sufficient context and detail.

### **2. Search First:**

Before submitting a new issue, please search the existing issues to avoid duplicates. If you find a similar issue, you can add your information or 👍 the issue to show your support.

### **3. Be Clear and Concise:**

- **Title**: Use a descriptive title that summarizes the issue.
- **Description**: Provide as much detail as necessary, but try to be concise. If reporting a bug, include steps to reproduce, expected behavior, and actual behavior.
- **Screenshots**: If applicable, add screenshots to help explain the issue.

### **4. Use Labels:**

If possible, categorize your issue using the appropriate GitHub labels. This helps us prioritize and address issues faster.

### **5. Stay Engaged:**

After submitting an issue, please check back periodically. Maintainers or other contributors may ask for further information or provide updates.

Thank you for helping improve our project! Your feedback and contributions are invaluable.

</details>

<br>
<br>


# ✉️ **LICENSE**
Causal Learning is dual-licensed under the GNU General Public License version 3(GPLv3) and a separate Commercial License.

Please consult the [LICENSE](./LICENSE/) files in the repository for more detailed information on the licensing of Causal Learning.
