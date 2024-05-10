# Causal Learning Framework by CCNets

[![Static Badge](https://img.shields.io/badge/Release-v1.1.1-%25%2300FF00)](https://github.com/ccnets-team/causal-rl)
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
- [🏃 **Quick Start**](#🏃-quick-start)
- [📖 **Features**](#📖-features)
- [🔎 **API Documentation**](#🔎-api-documentation)
- [🌟 **Contribution Guidelines**](#🌟-contribution-guidelines-)
- [🐞 **Issue Reporting Policy**](#🐞-issue-reporting-policy-)
- [✉️ **Support & Contact**](#✉️-support--contact)

# 🎈 Overview

## **Introduction**

CCNets is an innovative ML framework specifically designed for uncovering and modeling causal relationships between features and labels in complex datasets. This framework employs a unique structure comprising encoder networks and core networks to facilitate a deeper understanding of causality in data.

## **Key Capabilities**
CCNets harnesses its capability through six core functions: Explain, Reason, Produce, Infer, Reconstruct, and Generate. Each of these functions plays a crucial role:

`Explain`: Identifies key features from **Input Observations** and extracts an **Explanation Vector**. 

`Reason`: Utilizes the **Explanation** and **Input Observations** to reason about the **Label**.

`Produce`: Generates new data based on **Conditions** and **Explanation**.

`Infer`:  Infers output from **Input Data** using the *explainer* and *reasoner* networks.

`Reconstruct`: Reconstructs the **Input Data** by first explaining, then reasoning , and filnally producing the output.

`Generate`: Generate new data based on **Explanation** with **random discrete Conditions** 

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

### 1. Integrating Encoder and Core Nets

CCNets is designed with a robust architecture comprising two main components: the Encoder Net and the Core Net. This setup enables efficient handling and processing of diverse data types through specialized encoding techniques.

<p align="center">
  <img src="https://github.com/ccnets-team/causal-learning/assets/95277008/734fbf41-c312-4d30-a68e-212e626bc226" alt="two_nets" width="1000>
</p>


- `Encoder Net`: Responsible for converting input data into a format that is conducive for causal analysis. For **image data**, this component transforms it into a *trajectory* format, effectively capturing the temporal and spatial dynamics, which are crucial for subsequent processing by the Core Net.
- `Core Net`: Acts as the framework’s central network, where the encoded data from the Encoder Net is further analyzed. It is adept at handling the complexities of causal inference and prediction, seamlessly integrating various forms of data.



### 2. Data Type Flexibility

CCNets stands out for its versatility in handling almost all types of data, which is made possible through specific mechanisms tailored to optimize causal inference:

- `Tubular Data`: Typically used in traditional machine learning scenarios where structured data is prevalent.

- `Image Data`: Processed into trajectories by the Encoder Net to ensure detailed capture and analysis.

- `Time-Series Data`: Utilizes the GPT model to manage inherently sequential data, allowing for more effective analysis by maintaining temporal integrity, essential for accurate causal predictions.

- `Imbalanced Data`: Innovatively addresses challenges associated with imbalanced datasets by causally recreating data for minority label classes. This method effectively augments the dataset, providing a balanced environment for model training and overcoming issues related to data scarcity.



### 3. Various Supported Models
CCNets is compatible with a variety of state-of-the-art models, which enhance its adaptability and effectiveness across different applications:

- `GPT`: Excellent for handling sequential data like time-series, using its robust capabilities to understand and generate text based on underlying causal relationships.

- `StyleGAN`: Employed for generating high-quality, artificial images that can be further analyzed for causal relationships.


- `DeepFM`: Combines deep learning with factorization machines to effectively manage recommendation systems and causal inference in sparse data scenarios.

- `SuperNet`: Provides flexibility in architecture choices, optimized for specific causal inference tasks, accommodating various requirements and enhancing model performance.



# 🔎 **API Documentation**

- We're currently in the process of building our official documentation webpage to better assist you. In the meantime, if you have any specific questions or need clarifications, feel free to reach out through our other support channels. We appreciate your patience and understanding!



# 🐞 **Issue Reporting Policy**

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


# ✉️ **Support & Contact**

Facing issues or have questions about our framework? We're here to help!

1. **Issue Tracker**:
    - If you've encountered a bug or have a feature request, please open an issue on our **[GitHub Issues page](https://github.com/ccnets-team/causal-rl/issues)**. Be sure to check existing issues to avoid duplicates.
2. **Social Media**:
    - Stay updated with announcements and news by following us on **[LinkedIn](https://www.linkedin.com/company/ccnets)**.
3. **Emergency Contact**:
    - If there are security concerns or critical issues, contact our emergency team at support@ccnets.org.

*Please be respectful and constructive in all interactions.*


# LICENSE
Causal Learning is dual-licensed under the GNU General Public License version 3(GPLv3) and a separate Commercial License.

Please consult the [LICENSE](./LICENSE/) files in the repository for more detailed information on the licensing of Causal Learning.
