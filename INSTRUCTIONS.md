# Course Instructions and Projects

## 📚 Initial Instructions for Students

This repository is the official starting point for all course projects. ### 📏 Note on Project Sizes & Group Dynamics
The "Suggested Size" (Small, Medium, Large) is a rough estimate of the workload, mapping approximately to group sizes of 1, 2, and 3 members. However, these are just recommendations! 
- A 3-person group selecting a "Small" project will be expected to complete all **Extra Objectives** and demonstrate a higher level of polish and depth. 
- Conversely, a solo student who ambitiously selects a "Large" project will be evaluated with adapted expectations regarding the breadth of completed tasks.

Here are the steps to get started:
1. **Choose a project**: Consult the [project list below](#project-list) to read the available tracks and check which ones are free or already assigned. Then communicate your chosen project to the professor via email.
2. **Fork**: Create a **fork** of this repository in your personal GitHub account. While it is preferable to use the Fork button in the top right (to keep the history visible for evaluation), you can also create a standalone repository and keep it private if you prefer.
3. **Clone**: Clone your fork locally.
4. **Work in this root**: Consult [CONTRIBUTING.md](CONTRIBUTING.md) for the required conventions on how to structure folders (`src/`, `data/`, `notebooks/`), how to write clean code, and how to use Git professionally in a team. Replace the placeholders in the `README.md` file with the technical information about the repository and `docs/REPORT.md` with the description of the project and the work done.
5. **AI Usage Policy**: The use of generative AI tools (ChatGPT, GitHub Copilot, Claude, etc.) is **permitted, but regulated**. The use of these tools is encouraged to speed up boilerplate code writing, for debugging, or as documentation support. However, **never delegate strategic thinking and architectural choices to AI**. Elaborate your strategy, write or generate the code, and take full responsibility for every line. The use of such tools must be explicitly declared in the final report.
6. **License**: It is good practice to release your work open source. You will find a `LICENSE` file (pre-set to MIT license). Open the file, replace `[Year]` and `[Name and Surname]` with the current year and the members of your team. Remember to choose a different one if you do not want to freely share your code.
7. **Submission**: Your GitHub fork is the **final deliverable** of the project. Ensure the code is reproducible following the instructions below and that the slides for the exam presentation are placed inside the `docs/` folder. If you opted for a private repository, evaluation can take place by making the repo visible to the professor (handle `antoninofurnari`) or by sending the repository source code via email.

---

This file contains the list of available projects, complete details for each project, and the formed groups.

## Project List

<!-- | ID | Title | Reference Module | Suggested Size | Dataset | Assigned |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | [Feature-based Knowledge Distillation](#project-1) | Knowledge Distillation | Small | UCF-101 / HMDB-51 | Free |
| 2 | [Cross-Modal Knowledge Distillation (Audio to Vision)](#project-2) | Knowledge Distillation | Medium | EPIC-Kitchens | Free |
| 3 | [Metric Learning for Egocentric Face Recognition](#project-3) | Metric Learning | Small | EGTEA Gaze+ | Free |
| 4 | [Few-shot Learning for Gesture Recognition](#project-4) | Metric Learning | Medium | miniImageNet | Free |
| 5 | [Graph-based Metric Learning for Scene Understanding](#project-5) | Metric Learning | Large | Visual Genome | Free |
| 6 | [Knowledge Distillation for Mobile Action Recognition](#project-6) | Knowledge Distillation | Small | HMDB-51 | Free |
| 7 | [Domain Adaptation for Action Recognition – Egocentric → Exocentric](#project-7) | Domain Adaptation | Medium | Source (egocentric) | Free |
| 8 | [Domain Adaptation with Image-to-Image Translation (CycleGAN)](#project-8) | Domain Adaptation | Medium | Office-31 | Free |
| 9 | [Multi-source Domain Adaptation for Action Recognition](#project-9) | Domain Adaptation | Large | Source 1 | Free |
| 10 | [Contrastive Learning for Video Representation (SimCLR Video)](#project-10) | Self-Supervised Learning | Small | Kinetics-400 | Free |
| 11 | [Masked Video Modeling (MAE-style) for Egocentric Video](#project-11) | Self-Supervised Learning | Medium | EPIC-Kitchens | Free |
| 12 | [Clustering-based Self-Supervised Learning for Action Discovery](#project-12) | Self-Supervised Learning | Medium | Unlabeled procedural videos (e.g., YouTube DIY, ~500 videos) | Free |
| 13 | [Temporal Action Localization with 1D CNN](#project-13) | Video Understanding | Small | ActivityNet-1.3 | Free |
| 14 | [Action Recognition with Vision Transformer (ViT-based)](#project-14) | Video Understanding | Medium | HMDB-51 | Free |
| 15 | [Vision-Language Alignment with CLIP for Video](#project-15) | Vision & Language | Medium | MSR-VTT | Free |
| 16 | [Multimodal Action Recognition – Video + Audio + Text](#project-16) | Vision & Language | Large | Synthetic (create your own videos with audio) or AudioSet + video | Free |
| 17 | [Egocentric Video + Gaze for Procedural Understanding](#project-17) | Video Understanding | Medium | EPIC-Kitchens + gaze | Free |
| 18 | [State-Space Models (Mamba) for Long Sequences](#project-18) | Advanced Sequential Modeling | Large | EPIC-Kitchens | Free |
| 19 | [Transformer vs RNN for Procedural Video Understanding](#project-19) | Advanced Sequential Modeling | Medium | Assembly101 | Free |
| 20 | [Diffusion Models for Trajectory/Motion Generation](#project-20) | Advanced Sequential Modeling | Medium | Human3.6M | Free |
| 21 | [Deep Q-Learning for Frame Selection in Video](#project-21) | Reinforcement Learning | Small | Video classification task (e.g., HMDB-51, 100 videos) | Free |
| 22 | [Policy Gradient for Gesture Control](#project-22) | Reinforcement Learning | Medium | Gesture dataset (e.g., MediaPipe skeleton of 5 common gestures) + Gymnasium environment (CartPole or GridWorld) | Free |
| 23 | [Multi-agent RL for Task Coordination](#project-23) | Reinforcement Learning | Large | Synthetic: MultiAgentEnv environment based on Gymnasium | Free |
| 24 | [Differentiable Task Graphs (Yao method) – Group A](#project-24) | Research Topic (Graphs/Procedural) | Large | Assembly101 | Free |
| 25 | [Task Graphs – Softmax vs Sum Feasibility – Group B](#project-25) | Research Topic (Graphs/Procedural) | Large | Same as Track 24: Assembly101 | Free |
| 26 | [Procedural Error Detection with Gaze – Group A](#project-26) | Research Topic (Egocentric/Multimodal) | Medium | EPIC-Kitchens | Free |
| 27 | [Error Detection – Progress-Aware Model – Group B](#project-27) | Research Topic (Egocentric/Multimodal) | Medium | Same as Track 26 | Free |
| 28 | [Graph Autoencoder for Geometric Representations](#project-28) | Research Topic (Graphs/Representation) | Large | Visual Genome | Free |
| 29 | [Hyperbolic Embeddings for Action Hierarchy](#project-29) | Research Topic (Advanced Representations) | Medium | ActivityNet | Free |
| 30 | [Generative Models for Data Augmentation in Egocentric Domain](#project-30) | Research Topic (Egocentric/Generative) | Medium | EPIC-Kitchens | Free |
| 31 | [Online Episodic Memory for Action Anticipation](#project-31) | Research Topic (Memory/Anticipation) | Large | EPIC-Kitchens | Free |

--- -->

## Detailed Project Descriptions


<a id='project-1'></a>
### Track 1: Metric Learning for Face Recognition
**Suggested Size**: Small  
**Reference Module**: Metric Learning  

#### Problem Description
Learn to recognize faces by training a model on a set of images of faces and testing it on a different set. Create a small demo working on images of faces collected by you.

#### Dataset
- **CASIA-WebFace** (https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) or a subset of it.
- ~500,000 images of ~10,000 subjects.

#### Minimum Objectives
1. CNN backbone (ResNet-18 fine-tuned)
2. Classification-based baseline, KNN generalization to new faces
3. Mtric Learning: Triplet loss with hard negative mining: given an anchor face, find positives (same person) and negatives (other people)
4. Retrieval evaluation: mAP @1, 5, 10 (if I show a face, does the model retrieve the same person in the top-10 results more frequently?)
5. Cluster analysis in latent space: are faces of the same person close together?

#### Extra Objectives
- Comparison with other losse functions (e.g., ArcFace, Siamese etc.).
- Ablations of parameters (mining strategies, batch size, etc.)

---

<a id='project-2'></a>
### Track 2: Few-shot Learning for Gesture Recognition
**Suggested Size**: Medium  
**Reference Module**: Metric Learning  

#### Problem Description
Recognize hand gestures from 1–5 labeled examples. This is useful when the gesture is rare or new. Train on a large dataset of gestures and test on a held-out set in a few-shot setting. Create a small demo with a few new gestures the model should generalize to.

#### Dataset
- **HAGRID** (https://github.com/hukenovs/hagrid) (or a subset)
- 1M RGB frames
- 33 classes of gestures

#### Objectives
1. 2D/3D CNN / Transformers on gesture images, classification-based baseline, KNN for generalization to new classes.
2. Metric learning: Triplet loss with hard negative mining: given an anchor, find positives (same gesture) and negatives (other gestures), linear probe for generalization to new classes.
3. Metric: Accuracy 5-way 1-shot, 5-way 5-shot
4. Report: how does performance change with more examples?

#### Extra Objectives
- 1D CNN/RNN/Transformer on skeleton coordinates (e.g., extracted with media-pipe)
- Detailed failure case analysis

---

<a id='project-3'></a>
### Track 3: Graph-based Metric Learning for Scene Understanding
**Suggested Size**: Large  
**Reference Module**: Metric Learning  

#### Problem Description
Represent scenes (e.g., kitchen, office) as graphs (objects = nodes, spatial relations = edges) and learn robust embeddings for scene-to-scene retrieval.

#### Dataset
- **GQA** (https://cs.stanford.edu/people/dorarad/gqa/download.html) or a subset
- 100K images
- Each with scene graphs and a subset with scene labels (location)

#### Minimum Objectives
1. Scene graph encoder: GCN that processes the graph and produces an embedding. Work with ground truth graphs.
2. Contrastive loss: pairs of similar scenes (same place, same activity) must have close embeddings (e.g., Triplet loss)
3. Retrieval: given a query graph, find the most similar scene graphs in the dataset and attach the corresponding scene label
4. Metrics: standard classification metrics (Accuracy etc.)

#### Extra Objectives
- Robustness to perturbations (remove nodes/edges from the test graph, check if retrieval degrades)
- Dynamic graph: extract scene graphs from videos (nodes = object tracks, edges = temporal interactions), e.g., using VLMs - evaluate the quality of the extracted graphs
- Interpretability: which edges are critical for similarity?

---

<a id='project-4'></a>
### Track 4: Feature-based Knowledge Distillation
**Suggested Size**: Small  
**Reference Module**: Knowledge Distillation  

#### Problem Description
Test different knowledge distillation strategies to train a small student model for image classification.

#### Dataset
- **CIFAR-100** or **ImageNet** (subset)

#### Minimum Objectives
1. **Teacher and Student**: Large CNN (e.g., ResNet-50) and Small CNN (e.g., ResNet-18)
2. **Algorithm**: Implement Classic KD (logits) and FitNets (feature distillation)
3. Compare student performance with and without distillation
4. Metrics: Accuracy comparison (teacher vs student no KD vs student + KD), model size (MB), inference time (ms)

#### Extra Objectives
- Combine feature distillation and logit distillation
- Try also Attention Transfer and Relational Knowledge Distillation

---

<a id='project-5'></a>
### Track 5: Cross-Modal Knowledge Distillation (Audio to Vision)
**Suggested Size**: Large  
**Reference Module**: Knowledge Distillation  

#### Problem Description
Distill knowledge from a video-only teacher into an audio-only student model. This helps when video is not available during inference but is present during training.

#### Dataset
- **EPIC-Kitchens** (https://epic-kitchens.github.io/) and **EPIC-Sounds** (https://epic-kitchens.github.io/epic-sounds/)

#### Minimum Objectives
1. **Teacher model**: Train an image encoder (e.g. ResNet-50) on EPIC-Kitchens
2. **Student model**: Train a audio encoder (e.g. Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778) on EPIC-Sounds using distillation loss from the video teacher
3. Evaluate student's performance against a vision-only baseline
4. Metrics: Accuracy comparison (teacher vs student no KD vs student + KD), model size (MB), inference time (ms)

#### Extra Objectives
- Implement contrastive distillation between audio and video embeddings
- Use a video encoder (e.g. 3D CNNs/Transformer) on EPIC-Kitchens
- Explore different audio encoders

---

<a id='project-6'></a>
### Track 6: Knowledge Distillation for Mobile Action Recognition
**Suggested Size**: Small  
**Reference Module**: Knowledge Distillation  

#### Problem Description
Compress a heavy video model (e.g., 3D ResNet-50) into a lightweight one (e.g., MobileNet) maintaining performance, for deployment on mobile devices.

#### Dataset
- **HMDB-51** or **UCF-101** (sports/daily actions)
- Video features available online

#### Minimum Objectives
1. **Teacher**: Pre-trained 3D ResNet-50 (baseline accuracy on the test set)
2. **Student**: MobileNet 3D (light version, e.g., 5–10x fewer parameters)
3. Training loop: student learns from the soft output of the teacher
4. Metrics: Accuracy comparison (teacher vs student no KD vs student + KD), model size (MB), inference time (ms)

#### Extra Objectives
- Temperature tuning: how does performance change with T = 1, 5, 10, 20?
- Attention transfer: not only logits, but also intermediate activation maps
- Visualization of what the teacher transmits to the student (t-SNE of the latent space)

---

<a id='project-7'></a>
### Track 7: Domain Adaptation for Action Recognition – Exocentric → Egocentric
**Suggested Size**: Medium  
**Reference Module**: Domain Adaptation  

#### Problem Description
A model trained on exocentric videos (fixed cameras) does not work well on egocentric videos (from smart glasses). Use Domain Adaptation (DA) to transfer knowledge. Use exocentric as source and egocentric as target.

#### Dataset
- **Assembly101** (https://assembly101.github.io/)
- Multi-view: pick one as exo and one as ego
- May use a subset

#### Minimum Objectives
1. **Baseline fine-tuning**: train on labeled target (egocentric), evaluate accuracy
2. **Adversarial DA**: gradient reversal layer
   - Shared encoder (CNN)
   - Classification head (predicts action on target)
   - Domain discriminator (predicts if egocentric=0 or exocentric=1)
   - Backprop: loss_class - λ*loss_domain (adversarial)
3. Metrics: target accuracy, standard classification metrics
4. Report: does the model manage to confuse the discriminator? Does accuracy improve with DA vs fine-tuning?

#### Extra Objectives
- Maximum Mean Discrepancy (MMD) loss and other domain adaptation methods
- Visualization of feature alignment (t-SNE source vs target)
- Per-class analysis: which actions are easy/difficult to adapt?

---

<a id='project-8'></a>
### Track 8: Domain Adaptation with Image-to-Image Translation (CycleGAN)
**Suggested Size**: Medium  
**Reference Module**: Domain Adaptation  

#### Problem Description
Translate images from domain A to domain B without aligned pairs (e.g., sketch → photo). Use the translation as pre-processing to improve the classifier on the target.

#### Dataset
- **Office-31** (source: Amazon, target: DSLR) (https://www.kaggle.com/datasets/xixuhu/office31)
- Or **VisDA** (syn → real) (https://ai.bu.edu/visda-2019/)

#### Minimum Objectives
1. **CycleGAN**: two generators (A→B, B→A) and two discriminators
2. Loss: adversarial (discriminator convinces) + cycle-consistency (G_AB(G_BA(x)) ≈ x)
3. Pipeline: 
   - Train CycleGAN to translate source → target
   - Use translated + original images to train classifier
4. Metrics:
   - Classifier accuracy on target
   - Visual quality (human)

#### Extra Objectives
- Simultaneous DANN: while CycleGAN translates, an adversarial domain discriminator
- Train on source and test on translated target - is this a better approach?
- Compare with other domain adaptation methods (feature-based)

---

<a id='project-9'></a>
### Track 9: Multi-source Domain Adaptation for Action Recognition
**Suggested Size**: Large  
**Reference Module**: Domain Adaptation  

#### Problem Description
Perform action recognition considering two labeled datasets (source) and one unlabeled one (target). Instead of a single source domain, use information from 3 different sources to improve on the target.

#### Dataset
- **Source 1**: HMDB-51
- **Source 2**: UCF-101
- **Target**: Kinetics subset

#### Minimum Objectives
1. Model with: shared encoder + 2 source classifiers + target classifier
2. Domain discriminator for each source (or global)
3. Weighted ensemble: assign weight to each source based on similarity with the target
4. Training loop: optimize all simultaneously
5. Metrics: target accuracy, per-source contribution analysis

#### Extra Objectives
- Incomplete batch simulation: what happens if a source is missing during training?
- Analogy study: how does performance vary with the number of sources?

---

<a id='project-10'></a>
### Track 10: Contrastive Learning for Video Representation (SimCLR Video)
**Suggested Size**: Small  
**Reference Module**: Self-Supervised Learning  

#### Problem Description
Pre-train a video encoder without labels using a contrastive loss on pairs of augmented frames/clips from the same video. Split the dataset into two sets: one for pre-training and one for linear probing.

#### Dataset
- **UCF-101** (https://www.crcv.ucf.edu/data/UCF101.php)
- 1k videos
- 101 action categories

#### Minimum Objectives
1. Supervised baseline on the labeled set for reference
2. 3D ResNet-18 as backbone
3. Pre-train with SimCLR on the unlabeled set and do linear probing on the labeled set
4. Metric: standard classification accuracy, compare linear probe accuracy vs supervised training from scratch

#### Extra Objectives
- Temperature in contrastive loss: T=0.1, 0.5, 1.0, effect on convergence
- Visualization: t-SNE embeddings of similar videos should cluster together
- Momentum contrast (MoCo) for larger batch size
- Other contrastive learning methods
- Linear probing vs fine-tuning

---

<a id='project-11'></a>
### Track 11: Masked Autoencoders for image representation learning
**Suggested Size**: Small  
**Reference Module**: Self-Supervised Learning  

#### Problem Description
Pre-train a transformer masked autoencoder in an unsupervised way, then use it for downstream tasks with linear probing. Split the dataset in an unlabeled set for pre-training and a labeled set for linear probing.

#### Dataset
- **ImageNet 1K** or a subset of it (https://www.image-net.org/download.php)
- 1K classes
- 1M images


#### Minimum Objectives
1. Supervised baseline on the labeled set for reference
2. Train a masked autoencoder transformer on the unlabeled set and do linear probing on the labeled set
3. **Evaluation**: standard classification accuracy, compare linear probe accuracy vs supervised training from scratch

#### Extra Objectives
- Visualization of reconstructed frames
- Ablate hyper-parameters (masking percentage etc.)
- Compare with other self-supervised learning methods

---

<a id='project-12'></a>
### Track 12: Clustering-based Self-Supervised Learning for Action Discovery
**Suggested Size**: Medium  
**Reference Module**: Self-Supervised Learning  

#### Problem Description
Discover recurring actions in **unlabeled** videos using iterative clustering. Useful when you have no annotations but the video has repetitive patterns.

#### Dataset
- Kinetics-400 (https://github.com/cvdfoundation/kinetics-dataset)
- Pick a subset of classes/videos if too large

#### Minimum Objectives
1. Train a self-supervised model on Kinetics-400 (e.g., SimCLR or VideoMAE)
2. Extract features from the trained model for each video
3. K-means clustering (start with k=10, then experiment)
4. Pseudo-labels: assign a cluster label to each clip
5. Evaluation: Cluster purity (how many videos in cluster 0 are truly similar?)

#### Extra Objectives
- Try different clustering methods
- Try different self-supervised learning methods
- Inspect clusters and try to assign meaningful labels to them

---

<a id='project-13'></a>
### Track 13: Temporal Action Segmentation from Video
**Suggested Size**: Small  
**Reference Module**: Video Understanding  

#### Problem Description
Train a model to segment actions from video assigning a label to each frame.

#### Dataset
- **EGTEA Gaze+** (https://cbs.ic.gatech.edu/fpv/)
- Consider pre-extracted features (e.g., from https://github.com/antoninofurnari/rulstm)
- Annotations: start/end frames for each action (may consider labels from https://github.com/antoninofurnari/rulstm)

#### Minimum Objectives
1. 1D CNN model: processes features along the temporal dimension
2. LSTM model
3. xLSTM model
4. Compare the three models
5. Report: systematic errors (e.g., always predicts the action too short/long)?

#### Extra Objectives
- Soft-NMS post-processing: merge overlapping detections
- Compare with Mamba-based models

---

<a id='project-14'></a>
### Track 14: Action Anticipation from Video
**Suggested Size**: Medium  
**Reference Module**: Video Understanding  

#### Problem Description
Anticipate the action in a video considering the setup of https://github.com/antoninofurnari/rulstm.

#### Dataset
- **EPIC-KITCHENS** (https://epic-kitchens.github.io)
- Pre-extracted features from https://github.com/antoninofurnari/rulstm (or newer features if available)
- Look at the action anticipation challenge (https://www.codabench.org/competitions/14471/)

#### Minimum Objectives
1. RULSTM baseline from https://github.com/antoninofurnari/rulstm
2. xLSTM-based model
3. Mamba-based model
4. Compare the three models

#### Extra Objectives
- Re-implement RULSTM with an xLSTM and compare with others
- Compare with Transformer-based models

---

<a id='project-15'></a>
### Track 15: Vision-Language Alignment with CLIP for Video
**Suggested Size**: Medium  
**Reference Module**: Vision & Language  

#### Problem Description
Align video features with text using a contrastive loss (CLIP style). Allows text queries for videos (e.g., "person chopping vegetables" → find similar videos).

#### Dataset
- **EPIC-KITCHENS** (https://epic-kitchens.github.io)
- Loot at the multi-instance retrieval challenge (https://www.codabench.org/competitions/12008/)

#### Minimum Objectives
1. **Video encoder**: pre-trained (TimeSformer, SlowFast) - extract features
2. **Text encoder**: pre-trained (BERT, DistilBERT) - extract features
3. **Contrastive loss**: Train an alignment module (e.g. FC layers) on top of the pre-trained encoders
4. Training and evaluation:
   - Text-to-video retrieval: given text, find a similar video
   - Metric: R@1, R@5, R@10 (top-K recall)
5. Report: does zero-shot search work?

#### Extra Objectives
- Fine-tuning encoder (vs frozen)
- Failure analysis: which pairs does the model confuse?

---

<a id='project-16'></a>
### Track 16: Multimodal Action Recognition – Video + Audio
**Suggested Size**: Large  
**Reference Module**: Video Understanding  

#### Problem Description
Classify actions by simultaneously exploiting video and audio. More modalities = more robustness.

#### Dataset
- **EPIC-KITCHENS** (https://epic-kitchens.github.io)
- Look at the action recognition challenge (https://www.codabench.org/competitions/13636/)

#### Minimum Objectives
1. **Video encoder**: 3D CNN
2. **Audio encoder**: 1D CNN on spectrogram (librosa)
3. **Fusion strategy**: embedding concatenation + FC
4. Loss: standard CE
5. Evaluation: 
   - Multimodal accuracy (all 2 modalities)
   - Single modality accuracy (for comparison)
   - Contribution analysis: which modality counts the most?

#### Extra Objectives
- Missing modality: robustness when video/audio/text is missing
- Cross-modal attention (perceiver-like module for audio-video fusion)
- Analysis of late vs early fusion

---

<a id='project-17'></a>
### Track 17: Egocentric Video + Gaze for Action Recognition
**Suggested Size**: Large  
**Reference Module**: Video Understanding  

#### Problem Description
Combine egocentric video (from a first-person point of view) with gaze tracking (where the person is looking) to understand what they are doing.

#### Dataset
- **EGTEA Gaze+** (https://cbs.ic.gatech.edu/fpv/)
- Features: video frames + gaze heatmap

#### Minimum Objectives
1. **Video encoder**: 2D CNN on frames (ResNet-18)
2. **Gaze encoder**: 2D CNN on gaze heatmap (Gaussian blob on gaze point)
3. **Fusion**: embedding concatenation
4. **Classification**: FC head → action
5. Evaluation: 
   - Action accuracy
   - Ablation: video only vs gaze only vs fused
6. Report: does gaze really help? In which actions?

#### Extra Objectives
- Saliency map: where does the model "look" vs where does the person look?
- Attention bottleneck: is gaze a bottleneck for some actions?
- Temporal alignment: synchronize video and gaze

---

<a id='project-18'></a>
### Track 18: State-Space Models (Mamba) for Mistake Detection
**Suggested Size**: Large  
**Reference Module**: Advanced Sequential Modeling  

#### Problem Description
Use State-Space Models (Mamba, Hippo) to find mistake in long procedural videos. Take a dataset with per-frame mistake labels and train a model to predict mistake or not for each frame.

#### Dataset
- **Assembly101** (look for pre-extracted features)
- Use one of the views
- Annotations: per-frame mistake labels

#### Minimum Objectives
1. **Baseline C2F from the original paper**: re-implement the baseline from the original paper and replicate results
2. **Mamba/SSM**: implementation (or use library: mamba-ssm, ssm-lib)
3. **xLSTM**: implementation (or use library: xlstm)
4. **Benchmark**: Mamba vs xLSTM vs Baseline on long sequences

#### Extra Objectives
- Compare with online Transformer (TeSTra etc.)
- Ablation: how does sequence length affect Mamba vs LSTM?
- Is injecting ground truth actions helpful?

---

<a id='project-19'></a>
### Track 19: Transformer vs RNN for Procedural Video Understanding
**Suggested Size**: Medium  
**Reference Module**: Advanced Sequential Modeling  

#### Problem Description
Compare Transformer and RNN on procedural step understanding tasks (e.g., Assembly, cooking): which is more effective on procedures?

#### Dataset
- **EGO4D Goal-Step** (https://github.com/facebookresearch/ego4d-goalstep)
- Consider the online step detection task
- Look for pre-extracted video features

#### Minimum Objectives
1. **LSTM baseline**: encoder on video features
2. **Transformer encoder**: multi-head attention, positional encoding
3. Training: given past history, classify the current step
4. Metrics:
   - Frame-level accuracy (predicts correct step for each frame)
   - Per-class F1 score
   - Latency: Transformer vs LSTM inference

#### Extra Objectives
- Hybrid: Transformer + recurrence layer
- Attention analysis: attention heads specialized for temporal vs contextual?


---

<a id='project-20'></a>
### Track 20: Image & Language Representation Learning
**Suggested Size**: Medium  
**Reference Module**: Vision & Language  

#### Problem Description
Learn joint representations of images and text using a contrastive loss (CLIP style). Allows text queries for images.

#### Dataset
- **MS-COCO** (https://cocodataset.org/)
- 1.5M image-text pairs
- can pick a subset

#### Minimum Objectives
1. **Image encoder**: pre-trained (ResNet-50, ViT)
2. **Text encoder**: pre-trained (BERT, DistilBERT)
3. **Contrastive loss**: Train an alignment module (e.g. FC layers) on top of the pre-trained encoders
4. Training and evaluation:
   - Image-to-text retrieval: given text, find a similar image
   - Metric: R@1, R@5, R@10 (top-K recall)
5. Report: does zero-shot search work?

#### Extra Objectives
- Fine-tuning encoder (vs frozen)
- Failure analysis: which pairs does the model confuse?

---

<a id='project-21'></a>
### Track 21: Deep Reinforcement Learning for Frame Selection in Video
**Suggested Size**: Large  
**Reference Module**: Reinforcement Learning  

#### Problem Description
An agent learns to select informative frames from a video to reduce computational cost while maintaining good action classification accuracy. This is useful for compressed video.

#### Dataset
- **UCF101** (https://www.crcv.ucf.edu/data/UCF101.php)

#### Minimum Objectives
1. **EfficientNet-B0** encoder on individual frames as an agent. It predicts a binary probability (select or not).
2. **ResNet-18** encoder on the full clip as a baseline
3. **ResNet-18** encoder on the selected frames (after binary classification)
4. **Reward Model** assigns a reward based on the action classification accuracy/loss and number of selected frames. Choose your own reinforcement learning algorithm (e.g. DQN, PPO, etc.)
5. **Evaluation**:
   - Action accuracy vs # of selected frames
   - Comparison: random selection vs learned selection

#### Extra Objectives
- Compare multiple reinforcemnt learning algorithms
- Compare with other frame selection methods (random, uniform, etc.)
- Compare different reward models

---

<a id='project-22'></a>
### Track 22: Learn to Play Super Mario Bros with Deep Reinforcement Learning
**Suggested Size**: Large  
**Reference Module**: Reinforcement Learning  

#### Problem Description
Train an agent to play Super Mario Bros using deep reinforcement learning. The agent receives a reward based on the distance traveled, coins collected, and enemies defeated.

#### Dataset
- **Super Mario Bros** (see here: https://github.com/yfeng997/MadMario)

#### Minimum Objectives
1. **Agent**: Deep Q-Network (DQN) with a 2D CNN for perception
2. **Reward Model**: Reward based on distance traveled, coins collected, and enemies defeated
3. **Comparison**: Compare with other reinforcement learning algorithms (e.g. PPO, etc.)
4. **Evaluation**:
   - Cumulative reward over time
   - Convergence speed
   - Number of worlds completed

#### Extra Objectives
- Ablate different reward models
- Experiment with 3D CNNs and temporal models (e.g., RNNs) for action selection (agent)

---

<a id='project-23'></a>
### Track 23: Align a Small LLM with GRPO for Strict Code or JSON Generation
**Suggested Size**: Large  
**Reference Module**: Reinforcement Learning

#### Problem Description
Large Language Models often struggle with strict formatting constraints when generating complex outputs. This project involves fine-tuning a small LLM (e.g., 1.5B to 3B parameters) using Group Relative Policy Optimization (GRPO) to output syntactically perfect code or strict JSON structures. Instead of training a memory-intensive neural Reward Model, the system uses a programmatic reward function (such as a JSON linter or Python compiler) to evaluate generations and assign rewards based on successful execution or parsing.

#### Dataset
- **Synthetic Data**: Generate a dataset of instructions that require strict formatting (e.g., "Generate a JSON object with the following fields: ...", "Write a Python function that ...").
- **Reward Model**: A rule-based programmatic function (e.g., `json.loads()` for JSON, or `ast.parse()` for Python) that assigns a high reward $R_i$ for syntactically correct outputs and a penalty for parsing errors.

#### Minimum Objectives
1. **Agent**: A small pre-trained LLM (e.g., Gemma-2B or Phi-3-mini) utilizing LoRA/PEFT for memory-efficient training. Choose a small enough model that does not work very well on the task.
2. **Reward Model**: A rule-based programmatic function (e.g., `json.loads()` for JSON, or `ast.parse()` for Python) that assigns a high reward $R_i$ for syntactically correct outputs and a penalty for parsing errors.
3. **Algorithm**: Implement the GRPO training loop, utilizing the advantage calculation $A_i = \frac{R_i - \mu_R}{\sigma_R}$ across small generation groups.
4. **Evaluation**:
   - Syntax error rate / Pass@1 score on a holdout test set
   - Reward convergence over time
   - Comparison of format adherence before and after RL fine-tuning

#### Extra Objectives
- Compare the GRPO fine-tuned model's performance against standard Supervised Fine-Tuning (SFT).
- Ablate different group generation sizes (e.g., $G=4$ vs. $G=8$) to evaluate the impact on training stability and GPU memory usage.
- Implement a dual-reward system that gives partial points for proper use of `<think>` tags prior to generating the final code.

---

## Groups

| Group Name | Members |
| :--- | :---: |
| LeMeCla | 3 |
| BAT 🦇 (Backpropagation Attention Team) | 3 |
| Deep Team | 3 |
| Justgood AI | 3 |
| FiCo | 3 |
| FlyNow | 3 |
| Overfittony | 3 |
| The Outliers 2.0 | 3 |
| Zero e Uno | 2 |
| DataMinds | 2 |
| TEAM CassiaBranca | 2 |
| Le larunghie | 2 |
| DataLost | 2 |
| EventHorizonTeam | 2 |
| Marte | 2 |
| G16 | 1 |
| G17 | 1 |
| G18 | 1 |
| G19 | 1 |
| G20 | 1 |
| G21 | 1 |
| G22 | 1 |
| G23 | 1 |
| G24 | 1 |
| G25 | 1 |
| G26 | 1 |
| G27 | 1 |
| G28 | 1 |
| G29 | 1 |
| G30 | 1 |
| G31 | 1 |
