# Interpretable Urdu Fake News Detection Framework: Enhancing Classification and Explainability

**Authors:**
- Saamiya Mariam (20I-0612)
- Maliha Masud (20I-2606)

## Abstract

Fake news detection in the Urdu language presents intricate challenges owing to linguistic complexities, limited labeled datasets, and cultural nuances. This research addresses these challenges by proposing an interpretable framework, DITFEND (Interpretable Urdu Fake News Detection), designed to enhance classification accuracy while providing transparent decision-making processes.

The problem statement outlines the specific hurdles encountered in Urdu fake news detection, emphasizing the scarcity of labeled data, linguistic intricacies, and the critical need for interpretability in the detection process. To mitigate these challenges, the DITFEND model integrates a multi-step approach involving General Model Training, Transferability Assessment, and Target Domain Adaptation. Through extensive experimentation and evaluation on curated datasets, the DITFEND model demonstrates substantial advancements in accuracy and interpretability compared to existing models. The results showcase not only superior classification performance but also enhanced explainability, enabling users to comprehend the rationale behind classification decisions.

**Key contributions of this research include:**
- Identification and detailed elucidation of challenges specific to Urdu fake news detection.
- Introduction of DITFEND, an interpretable framework tailored for Urdu, addressing these challenges.
- Empirical validation of DITFEND’s effectiveness in improving classification accuracy and interpretability.

The findings underscore the significance of interpretable frameworks in addressing the complexities of detecting fake news in Urdu, paving the way for more reliable and transparent systems in combating misinformation.

## 1. Introduction

In a world where the propagation of misinformation threatens societal discourse, the identification and mitigation of fake news stand as paramount challenges, especially within the Urdu linguistic context. With its unique linguistic intricacies and limited labeled datasets, detecting and combating fake news in Urdu presents a significant challenge. This research aims to address these hurdles by proposing an interpretable framework, the "Domain- and Instance-level Transfer Framework for Fake News Detection" (DITFEND), designed to not only enhance classification accuracy but also provide transparency through explainable decision-making processes.

The complexity of detecting fake news in Urdu stems from the scarcity of labeled datasets specific to this language, compounded by intricate linguistic nuances. As misinformation continues to proliferate across digital platforms catering to Urdu-speaking communities, the necessity for a reliable and transparent detection mechanism becomes increasingly evident. The need for interpretability in the detection process is crucial, ensuring that the model’s decisions are comprehensible, thereby fostering trust and understanding among users.

This research endeavors to contribute significantly to the field of fake news detection by offering a novel approach tailored specifically for Urdu. By addressing the challenges of limited datasets and linguistic intricacies, the proposed DITFEND framework aims to enhance the accuracy and explainability of fake news classification in Urdu. Furthermore, the model’s adaptability across different news domains signifies its potential to be a pivotal tool in combatting misinformation within the Urdu linguistic landscape. Through a comprehensive investigation and validation process, this research seeks to establish a robust foundation for interpretable and accurate fake news detection in Urdu, ensuring its relevance and applicability in real-world contexts.

## 2. Literature Review

The prevalence of fake news and misinformation in the digital landscape has highlighted the importance of robust detection mechanisms, especially in languages like Urdu, known for their complexities in linguistics and cultural variations. However, the research landscape for Urdu fake news detection remains relatively underdeveloped, presenting unique challenges that demand tailored methodologies.

### 2.1 Prior Works in Interpretable Fake News Detection

Bhattarai, Granmo, and Jiao (2022) introduced an explainable framework utilizing the Tsetlin Machine for fake news detection. Their work emphasized the significance of interpretability in decision-making processes, encouraging that transparency is necessary for users to comprehend classification outcomes. While demonstrating efficiency in other languages, the adaptability and effectiveness of this framework in the context of Urdu fake news detection require further exploration.

Nan et al. (2022) addressed domain and instance-level transfer to enhance fake news detection across varying domains. Their study primarily focused on languages with ample linguistic resources, offering insights into transferability techniques. However, its applicability to Urdu, with its unique linguistic nuances and limited datasets, warrants specific exploration. The emphasis on transferability motivates the investigation of adaptation and transfer learning techniques tailored explicitly for Urdu.

### 2.2 The Significance of Interpretability in Fake News Detection

Interpretability has emerged as a crucial aspect in the development of fake news detection models, particularly concerning user trust and comprehension of the classification process. Providing comprehensible explanations for classification decisions enhances transparency, enabling users to understand the rationale behind the model’s decisions.

Several studies underscore the significance of interpretability in fake news detection models across languages. However, within the existing literature, the integration of interpretable frameworks that are specifically made for Urdu fake news detection remains largely unexplored. This research aims to address this gap by introducing the DITFEND framework, aiming to enhance classification accuracy and interpretability, specifically tailored to the complexities of Urdu language in the context of fake news detection.

## 3. Problem Statement

### 3.1 Challenges in Detecting Fake News in Urdu

Spotting fake news in Urdu is tough because the language has many different ways of saying things. There aren’t many labeled datasets to teach computers how to spot fake news accurately. Also, cultural differences and how people see news can make it hard to tell what’s real and what’s fake.

The computers we use to find fake news in Urdu don’t explain why they think something is fake. This makes it hard for us to trust them and understand why they make certain decisions.

### 3.2 Why We Need an Easy-to-Understand System

Because of these problems, we really need a system that can spot fake news in Urdu but also tells us why it thinks something is fake:

- **Understanding Why:** If we know why a computer thinks something is fake, it helps us check if the news is actually true.
- **Building Trust:** If the system explains why it thinks a piece of news is fake, it helps us trust it more and rely on it better.
- **Dealing with Language Differences:** A system that’s easy to understand can help us figure out the tricky parts of Urdu and how fake news might use these tricks.

## 4. Proposed Solution: DITFEND Framework

The DITFEND framework is designed to address the challenges specific to Urdu fake news detection, emphasizing interpretability and classification accuracy. It consists of three main components: General Model Training, Transferability Assessment, and Target Domain Adaptation.

### 4.1 General Model Training

The first step involves training a general model using existing labeled datasets in Urdu. This step aims to equip the model with a foundational understanding of fake news detection in the language. A deep neural network architecture, incorporating attention mechanisms and linguistic features unique to Urdu, is employed.

### 4.2 Transferability Assessment

To enhance the model's adaptability, a transferability assessment is conducted. This involves evaluating the performance of the trained model across different domains and linguistic variations. The goal is to identify the model's robustness and its ability to generalize fake news detection across diverse contexts.

### 4.3 Target Domain Adaptation

The final step focuses on adapting the model to the target domain, specifically addressing linguistic intricacies and challenges unique to Urdu fake news. This phase involves fine-tuning the model with limited labeled data from the target domain, ensuring that the model becomes specialized in recognizing fake news within the Urdu linguistic landscape.

## 5. Experimental Methodology

### 5.1 Dataset Collection and Preprocessing

Two distinct datasets are utilized in this study: a general dataset comprising labeled instances of fake and real news in Urdu and a target domain dataset representing the specific domain of interest. The general dataset is employed for the initial model training, while the target domain dataset facilitates domain-specific adaptation.

### 5.2 Model Architecture

The DITFEND framework employs a deep neural network architecture, integrating attention mechanisms and linguistic features unique to Urdu. The model's architecture is optimized for both accuracy and interpretability, ensuring that the decision-making processes are transparent and comprehensible.

### 5.3 Evaluation Metrics

Classification accuracy, precision, recall, and F1 score are employed as primary evaluation metrics. Additionally, the interpretability of the model is assessed through attention maps and feature importance analysis.

## 6. Results and Discussion

### 6.1 General Model Training Results

The initial training phase on the general dataset demonstrates promising results, achieving competitive accuracy in fake news detection. Attention mechanisms within the model highlight significant linguistic features contributing to classification decisions.

### 6.2 Transferability Assessment Results

The transferability assessment across diverse domains reveals the model's adaptability, showcasing robust performance in varying linguistic contexts. This step validates the generalizability of the trained model.

### 6.3 Target Domain Adaptation Results

Fine-tuning the model with limited labeled data from the target domain results in improved performance, specifically addressing linguistic intricacies unique to Urdu fake news. The adaptation process enhances the model's domain specificity.

### 6.4 Interpretability Analysis

Attention maps and feature importance analysis contribute to the interpretability of the DITFEND framework. Users can gain insights into the linguistic cues and features influencing the model's decision-making processes, fostering trust and understanding.

## 7. Conclusion

This research introduces the DITFEND framework, an interpretable Urdu fake news detection system tailored to address the challenges of limited datasets, linguistic intricacies, and the critical need for transparency. The proposed model demonstrates significant advancements in classification accuracy and interpretability compared to existing models, paving the way for more reliable and transparent systems in combatting misinformation within the Urdu linguistic landscape.

Future work includes expanding the model's adaptability to different Urdu dialects, exploring additional linguistic features, and further enhancing the interpretability of decision-making processes. The DITFEND framework holds promise in contributing to the development of robust and interpretable fake news detection systems, fostering a safer and more trustworthy information environment for Urdu speakers.
