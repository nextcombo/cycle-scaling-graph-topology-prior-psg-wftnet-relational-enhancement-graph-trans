# Cycle + Scaling Graph Topology Prior | PSG-WFTNet Relational Enhancement Graph Transformer for Rotating Machinery Fault Diagnosis New Framework

Preface: Rolling bearings are key supporting components in rotating machinery. Local defects such as the inner ring, outer ring, and rolling elements often cause repetitive impacts during operation. These impacts, after being transmitted through structural paths, can also trigger local resonance, amplitude modulation, and non-stationary decay responses. Therefore, a real vibration signal typically contains:
1. Periodic repetitive modes: The fault location repeatedly enters the contact area with rotational motion, forming approximately periodic impacts;
2. Local transient impacts: Each collision appears as a spike or non-stationary abrupt change within a short time range;
3. Resonance and multiscale responses: Impacts excite different frequency bands responses of the bearing, shaft, and base structure;
4. Harmonic and modulation components: Base frequency, double frequency, sidebands, and rotating frequency coupling make the spectrum structure complex;
5. Noise and operational disturbances: Rotational speed, load, sensor installation, and background noise can alter vibration statistical characteristics.
Therefore, the fundamental requirement for rolling bearing fault diagnosis is to simultaneously answer two questions: "What global cycle repeats in the impact?" and "What are the most significant local scales and frequency bands at each impact?" This is precisely the direct motivation behind PSG-WFTNet's simultaneous modeling of Period and Scale. We propose an innovative solution: PSG-WFTNet (Period-Scale Graph Wavelet-Fourier Transform Network). Inspired by WFTNet's idea of jointly modeling global—local cycles, this model further elevates Top-K extracted from FFT and multiscale local responses from CWT from "feature branches" to "structural semantic nodes." On this basis, the model constructs a Period-Scale Graph that includes harmonic relationships, periodic—scale frequency alignment relationships, and scale neighborhood relationships. Through Relation-aware Graph Transformer, it directly injects structural relationships into multi-head attention, achieving joint modeling of data-driven relevance and frequency-scale priors. Finally, through graph-level attention pooling, it adaptively recognizes key cycles and scales in different fault samples, and can further utilize node importance, Period-to-Scale Attention, and prior-learning consistency analysis to form interpretable diagnostic evidence chains. This model is particularly suitable for fault diagnosis tasks in rolling bearings and other rotating machinery with characteristics of periodic impacts and local resonances. (Effectiveness validation provided for two public datasets!)
1. Model Introduction
1.1 Overall Architecture of the Model
PSG-WFTNet maps an original vibration signal to two complementary structural spaces: (1) Fourier Period Branch: selecting Top-K main frequencies from FFT spectrum to obtain corresponding global periods, and reconstructing a two-dimensional periodic representation based on these periods; (2) Wavelet Scale Branch: constructing multiscale local representations using CWT, extracting transient features at each scale while preserving scale dimensions, forming Scale Nodes; (3) Period-Scale Graph: integrating K Period Nodes with S Scale Nodes into a unified graph to form three types of structural relationships; (4) Relation-aware Graph Transformer: explicitly injecting edge relations into multi-head attention mechanisms to control message propagation between nodes simultaneously by data features and period-scale priors; (5) Graph Attention Pooling + Classifier: adaptively computing contributions of different period and scale node pairs to the current sample, obtaining graph-level diagnostic representations and completing classification.
PSG-WFTNet can be summarized as the following end-to-end process: Raw vibration signals → Lightweight Temporal Encode → Fourier Main Period Extraction / CWT Multiscale Extraction → Period Nodes / Scale Nodes → Period-Scale Graph → Relation-aware Graph Transformer → Node Attention Pooling → Fault Classification. Its fundamental difference from a typical "dual-branch network" is that the two branches do not simply Concatenate or Weighted Sum at the end but undergo structured interactions at the node relationship level.
\t  
1.2 Introduction to Model Innovation Points
Innovation Point 2: Three Types of Multi-relational Edges for Fault Mechanisms. If only the Period Nodes and Scale Nodes are input into a standard Transformer, while learning data correlation is possible, all node relationships need to be re-estimated from the training data, lacking the structural priors that are already explicitly present in mechanical signals. PSG-WFTNet does not use a simple binary adjacency matrix, but defines multi-relational edges: (1) Harmonic relations: describe whether the different Fourier main frequencies are close to integer multiples. (2) Period-scale alignment relationship: describes whether the FFT main frequency and CWT pseudo frequency are close to the band. (3) Scale neighborhood relationship: It describes the continuous frequency structure between different wavelet scales. Multi-relational edges allow the graph structure to no longer be "graphed for the sake of using GNNs," but rather, it is generated from relationships with clear signal processing meaning. It unifies the harmonic structure in Fourier, the scale continuity in Wavelet, and the frequency alignment between two types of representations into a single graph.
3. Innovation Point: Relation-aware Graph Transformer Global-Local Collaborative Inference. Traditional Graph Transformers or standard Self-Attention typically only calculate based on node features, meaning that even if two nodes are highly correlated in frequency, the model must rely entirely on the training process to discover this relationship itself. PSG-WFTNet directly incorporates edge relationships into the attention score, thus the attention is simultaneously driven by two components: the first is data-driven node semantic similarity, and the second is edge biases generated from periodic-scale structural relationships. This mechanism avoids the passive fusion of Fourier and Wavelet paths after feature extraction, allowing them to query and correct each other at the level of semantic nodes. The model can learn "what local scale evidences are needed for a certain global cycle" and also "what kind of global repetitive pattern belongs to a certain local transient scale."
4. Innovation Point: Syntactic Node Design of Periodicity Significance and Scale Energy Enhancement
\tIf only the two-dimensional feature pooling results are used as nodes, the nodes have deep features but the meaningful information such as "how significant this period is" and "how strong the local energy at this scale is" is not explicitly preserved. The mechanism of PSG-WFTNet: Period Node fuses Fourier magnitude, Scale Node fuses CWT scale energy, therefore, a node carries:
Deep data representation;
Significant periodicity or scale energy;
\tPeriod / Scale type identity;
\tCorresponding frequency position.
This enables the Period and Scale nodes to transition from "abstract depth feature vectors" to diagnostic entities with clear structural semantics, providing a stable semantic foundation for subsequent graph relation modeling.
2. Bearing Data Loading and Preprocessing
2.1 Import data
1. Preprocessing for Classification of Faults 10
2. CWRU Bearing Data Classification Dataset:
   - Read common CSV format data:
     - Each row = 1024 sample points + 1 label
   - Suitable for already organized CWRU ten-class data, can be extended to other bearing fault datasets. train_set, val_set, test_set are divided into training set, validation set, and test set in a ratio of 7:2:1, and finally save the data.
3. Preprocessing of Fault Data
4. Fault Diagnosis Model Based on PSG-WFTNet
   - Define the network model
   - Set parameters, train the model with parameters, and train the model
   - 50 epochs, accuracy 100%, the classification effect of PSG-WFTNet is significant, which can extract important features from fault signal characteristics for model recognition, and the effect is very obvious!
   - Model evaluation
      (1) Accuracy, Precision, Recall, F1 Score
      (2) Fault classification confusion matrix
      (3) Classification label visualization
      (4) t-SNE feature visualization after model training:
   - Explanability analysis
      (1) Importance of Period–Scale nodes for different fault categories
      From the importance of Period–Scale nodes, it can be seen that there are obvious differences in the dependence of different fault categories on global periodical features and local scale features. Some fault categories are more dependent on Fourier period nodes, while others show higher CWT scale node weights, indicating that PSG-WFTNet can adaptively adjust the contribution of global periodic information and local transient information according to different fault patterns, achieving differentiated feature fusion.
(2)Period → Scale Global Attention
The attention heatmap from Period → Scale shows that all three period nodes pay strong attention to the intermediate scales S4∼S7, with S5 having the highest attention. This indicates that during the global period modeling process, the model actively acquires supplementary information from representative local wavelet scales, validating the effectiveness of the Period-Scale Graph in facilitating cross-domain information interaction between global periodic patterns and multiscale transient features.

 (3)Top-K Fourier Frequency for Different Fault Categories
From the top-K Fourier main frequency distribution for different fault categories, it is evident that there are significant differences in dominant frequency ranges across various categories, suggesting that Fourier cycle nodes can extract globally spectral information with discriminative characteristics. These high-frequency components are more suitable for interpreting as dominant spectrum components triggered by fault impacts or structural resonance responses, rather than directly equivalent to theoretical fault characteristic frequencies, further demonstrating that PSG-WFTNet can utilize spectral patterns related to categories to aid in fault identification.
Click to download: Full original data, Python code
END

2. Friends who like this article can click to follow, give "likes" to the article, share on social media or discuss groups
Previously excellent content:
Rapid publication papers | Based on 2D-SWinTransformer+1D-CNN-SENet parallel fault diagnosis model
Bearing life prediction major update! Added CNN-BiGRU-Attention model collection + rich visualization charts
Multimodal - Fault Diagnosis | Large kernel convolution opens a new era in visuals!
Strong! Direct writing in one zone! Based on SSA+Informer-SENet fault diagnosis model
Multisource Information Fusion: Data layer fusion + feature layer fusion fault diagnosis models!

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
