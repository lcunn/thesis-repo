
## **Experiment 1: Training an Embedding Model to Capture Musical Invariances (with Soft-DTW Integration)**

### **Objective:**

To develop and train a neural network embedding model that captures melodic similarities while inherently handling musical invariances such as transposition, tempo changes, and minor perturbations, potentially integrating Soft-DTW into the training process to enhance alignment learning.

### **Description:**

1. **Data Preparation:**
   - **Normalization Techniques:**
     - **Pitch Representation:**
       - Use **pitch intervals** instead of absolute pitches to achieve transposition invariance.
     - **Rhythmic Representation:**
       - Represent note durations relative to a standard unit or use a **beat-aligned grid**.
     - **Other Features:**
       - Include additional musical features such as articulation and dynamics if available.
   - **Data Augmentation:**
     - Apply **transpositions** to different keys.
     - Introduce **tempo variations** and slight **timing shifts** to simulate expressive performances.
   - **Dataset Composition:**
     - Use a diverse set of symbolic melodies (e.g., MIDI files) across various genres and styles.
     - Include melodies with known similarities and differences to train the model effectively.

2. **Model Development:**
   - **Architecture:**
     - **Sequence-to-Sequence Model:**
       - Use a **Bidirectional LSTM** or **Transformer encoder** to process melodic sequences.
     - **Embedding Layer:**
       - Project the sequence output to a fixed-size embedding vector (e.g., 128 dimensions).
   - **Soft-DTW Integration:**
     - Incorporate **Soft-DTW** as a differentiable loss function to directly model alignment between sequences.
     - **Objective:**
       - Minimize the Soft-DTW distance between embeddings of similar melodies and maximize it for dissimilar ones.

3. **Training Procedure:**
   - **Training Data:**
     - Create pairs of sequences:
       - **Positive Pairs:** Similar or perturbed versions of the same melody.
       - **Negative Pairs:** Dissimilar melodies.
   - **Loss Function:**
     - Combine **Soft-DTW loss** with a **contrastive loss** to optimize both alignment and embedding space.
     - **Total Loss:** `L_total = α * L_softdtw + β * L_contrastive`
       - **α** and **β** are weighting factors to balance the losses.
   - **Hyperparameters:**
     - **Batch Size:** 64
     - **Learning Rate:** 0.001
     - **Optimizer:** Adam
     - **Epochs:** 50 (with early stopping)
     - **Regularization:** Dropout (rate: 0.5), L2 regularization

4. **Evaluation:**
   - **Embedding Quality:**
     - Use **t-SNE** or **UMAP** to visualize embedding clusters.
     - Calculate **silhouette scores** to assess clustering of similar melodies.
   - **Similarity Assessment:**
     - Compute **Pearson** and **Spearman** correlation coefficients between model distances and known similarity measures.
   - **Baseline Comparison:**
     - Compare against models trained without Soft-DTW integration and with different loss functions.

### **What This Experiment Aims to Achieve:**

- **Capture Musical Invariances:** By using normalized representations and Soft-DTW, the model learns to focus on meaningful musical similarities while being invariant to transpositions and tempo changes.
- **Enhanced Alignment Learning:** Integrating Soft-DTW helps the model learn sequence alignments, which is crucial for capturing melodic similarities.
- **Foundation for Retrieval:** The embeddings generated will serve as a robust foundation for efficient retrieval in subsequent experiments.

## **Specification:**

### Training an Embedding Model (with Soft-DTW Integration)**

#### **Data Preparation**

- **Datasets:**
  - Use symbolic music datasets such as the **Essen Folk Song Collection**, **Nottingham Database**, and custom datasets with diverse genres.
- **Normalization Techniques:**
  - **Pitch Intervals:** Calculate the interval between successive pitches.
  - **Relative Durations:** Express durations as ratios relative to a base duration (e.g., quarter note).
- **Data Augmentation:**
  - **Transpositions:** Shift melodies across all 12 semitones.
  - **Tempo Variations:** Adjust tempos within a reasonable range (e.g., ±10 BPM).
- **Subsequence Extraction:**
  - **Window Size:** 2-4 bars
  - **Overlap:** 50%

#### **Model Development**

- **Architecture:**
  - **Input Layer:** Accepts sequences of pitch intervals and relative durations.
  - **Sequence Modeling Layer:**
    - **Bidirectional LSTM:** 2 layers with 128 units each.
  - **Embedding Layer:**
    - Fully connected layer projecting to 128-dimensional embeddings.
- **Soft-DTW Integration:**
  - Implement **Soft-DTW** as per Cuturi and Blondel's differentiable approximation.
  - **Temperature Parameter (γ):** Set to 0.1 for smoothness.

#### **Training Procedure**

- **Loss Function:**
  - **Total Loss:** `L_total = α * L_softdtw + β * L_triplet`
    - **L_softdtw:** Soft-DTW loss between sequences.
    - **L_triplet:** Triplet loss on embeddings.
    - **Weights:** α = 0.5, β = 0.5 (adjustable based on validation performance).
- **Hyperparameters:**
  - **Batch Size:** 64
  - **Learning Rate:** 0.001
  - **Optimizer:** Adam with learning rate decay
  - **Epochs:** 50 (early stopping with patience of 5 epochs)
  - **Regularization:** Dropout (0.5), L2 regularization (1e-5)

#### **Evaluation**

- **Embedding Visualization:**
  - Use **t-SNE** to reduce embeddings to 2D for visualization.
- **Similarity Metrics:**
  - Calculate **Mean Squared Error (MSE)** between predicted and actual similarity scores.
- **Baseline Models:**
  - Compare with models trained without Soft-DTW and with only contrastive loss.

---

## **Experiment 2: Efficient Retrieval Using Approximate Nearest Neighbor Search**

### **Objective:**

To implement and evaluate efficient retrieval methods using the embeddings from Experiment 1, enabling fast querying and subsequence matching in large-scale melody databases through approximate nearest neighbor (ANN) search.

### **Description:**

1. **Embedding Indexing:**
   - **Index Construction:**
     - Use embeddings from Experiment 1 to build an ANN index.
     - **Tools:** Utilize libraries like **FAISS** for efficient indexing.
   - **Indexing Methods:**
     - **Index Types:**
       - **IVF (Inverted File Index):** Suitable for large datasets.
       - **HNSW (Hierarchical Navigable Small World Graphs):** Offers a balance between speed and accuracy.
     - **Parameter Tuning:**
       - Adjust parameters like the number of clusters (`nlist`) and the number of probes (`nprobe`) to optimize performance.
   - **Dimensionality Reduction:**
     - Optionally apply **PCA** to reduce embedding dimensions and improve search speed.

2. **Query Processing:**
   - **Subsequence Extraction:**
     - From the query melody, extract overlapping subsequences using the same windowing approach as in Experiment 1.
     - **Window Size:** 2-4 bars
     - **Overlap:** 50%
   - **Embedding Computation:**
     - Generate embeddings for query subsequences using the trained model from Experiment 1.

3. **Similarity Search and Aggregation:**
   - **Nearest Neighbor Search:**
     - For each query embedding, retrieve the top **k** nearest neighbors from the index.
   - **Aggregation Strategy:**
     - **Voting Mechanism:**
       - Tally votes for each candidate melody based on the number of matching subsequences.
     - **Similarity Scoring:**
       - Compute an aggregate similarity score for each candidate melody.
   - **Thresholding and Ranking:**
     - Apply thresholds to filter out low-similarity matches.
     - Rank candidates based on aggregate scores.

4. **Evaluation:**
   - **Performance Metrics:**
     - **Accuracy:**
       - Precision@k, Recall@k, F1-score
     - **Efficiency:**
       - Query response time
       - Index build time
       - Memory usage
   - **Scalability Testing:**
     - Test with databases of varying sizes (e.g., 10K, 100K, 1M melodies).
   - **Baseline Comparison:**
     - Compare ANN search performance with exact nearest neighbor search (where feasible).

### **What This Experiment Aims to Achieve:**

- **Efficient Retrieval:** Demonstrate that ANN search enables fast and accurate retrieval of similar melodic subsequences in large databases.
- **Practical Applicability:** Show that the system meets the efficiency requirements necessary for real-world plagiarism detection.
- **Foundation for Plagiarism Detection:** Provide a retrieval mechanism that can be integrated into an end-to-end plagiarism detection system.

## **Specification:**

### **Experiment 2: Efficient Retrieval Using ANN**

#### **Embedding Indexing**

- **Index Construction:**
  - Use **FAISS** library.
  - **Index Type:** IVF-PQ (Inverted File with Product Quantization).
  - **Parameters:**
    - **nlist (number of clusters):** 1024
    - **nprobe (number of probes):** 16
    - **PQ Code Size:** 64 dimensions
- **Dimensionality Reduction:**
  - Apply **PCA** to reduce embeddings from 128 to 64 dimensions before indexing.

#### **Query Processing**

- **Subsequence Extraction:**
  - Same windowing and overlap as in Experiment 1.
- **Embedding Computation:**
  - Use the trained model to generate embeddings for query subsequences.

#### **Similarity Search and Aggregation**

- **Nearest Neighbor Search:**
  - Retrieve top 10 nearest neighbors for each query embedding.
- **Aggregation:**
  - **Voting Mechanism:** Count how many times each database melody appears in the top results.
  - **Scoring Function:** Sum of similarity scores for each candidate melody.

#### **Evaluation**

- **Performance Metrics:**
  - **Accuracy:** Precision@k, Recall@k, F1-score
  - **Efficiency:** Average query time, index build time, memory usage
- **Scalability Testing:**
  - Test with datasets of sizes: 10K, 100K, 1M melodies
- **Baseline Comparison:**
  - Compare with exact nearest neighbor search (brute-force) on smaller datasets.

---

## **Experiment 3: End-to-End Symbolic Music Plagiarism Detection System**

### **Objective:**

To develop and evaluate a complete symbolic music plagiarism detection system that leverages the embedding model and efficient retrieval methods from previous experiments, focusing on matching query subsequences with melodies in the database and verifying potential plagiarized cases.

### **Description:**

1. **System Integration:**
   - **Candidate Retrieval:**
     - Use the retrieval system from Experiment 2 to obtain candidate melodies with high similarity scores.
   - **Alignment Verification:**
     - Apply precise alignment methods to the top candidates to verify potential plagiarism.
     - **Alignment Methods:**
       - Use **Soft-DTW** for efficient and differentiable alignment scoring.
       - Optionally, apply **traditional DTW** for comparison.

2. **Plagiarism Decision-Making:**
   - **Scoring Mechanism:**
     - Combine embedding similarity scores with alignment verification scores.
     - **Composite Score:** `Score = γ * Embedding_Similarity + δ * Alignment_Score`
       - **γ** and **δ** are weighting factors.
   - **Threshold Determination:**
     - Establish thresholds for the composite score to decide whether a candidate melody is considered plagiarized.
     - **Threshold Tuning:**
       - Use validation data to optimize sensitivity (true positive rate) and specificity (true negative rate).

3. **Evaluation:**
   - **Datasets:**
     - Use a test set containing known cases of plagiarism and non-plagiarism.
     - Include real-world cases if available.
   - **Performance Metrics:**
     - **Detection Accuracy:**
       - True Positive Rate, False Positive Rate
       - Precision, Recall, F1-score
     - **ROC Curve Analysis:**
       - Plot ROC curves to visualize the trade-off between true positive and false positive rates.
     - **Confusion Matrix:**
       - Analyze the types of errors made by the system.
   - **Expert Evaluation:**
     - Collaborate with musicologists to assess the validity of detected plagiarism cases.

4. **Comparative Analysis:**
   - **Baseline Methods:**
     - Compare the system's performance with traditional plagiarism detection methods (e.g., alignment-only approaches).
   - **Ablation Study:**
     - Evaluate the impact of different components (e.g., without alignment verification) on overall performance.

### **What This Experiment Aims to Achieve:**

- **Practical Application:** Demonstrate the effectiveness of the developed system in detecting symbolic music plagiarism in a realistic setting.
- **Holistic Evaluation:** Assess the system's ability to correctly identify plagiarized melodies while minimizing false positives.
- **Contribution to the Field:** Provide insights into the benefits of combining embedding-based retrieval with alignment verification in plagiarism detection.

## **Specification:**

### **End-to-End Plagiarism Detection System**

#### **System Integration**

- **Candidate Retrieval:**
  - Use top N candidates (e.g., N=5) from Experiment 2 for each query subsequence.
- **Alignment Verification:**
  - **Soft-DTW Scoring:**
    - Compute Soft-DTW alignment score between query and candidate subsequences.
  - **Composite Scoring:**
    - `Composite_Score = γ * Embedding_Similarity + δ * (1 - Normalized_SoftDTW_Score)`
    - Normalize Soft-DTW scores to [0,1].
    - **Weights:** γ and δ adjusted based on validation set performance.

#### **Plagiarism Decision-Making**

- **Threshold Setting:**
  - Determine a **Composite Score Threshold** using ROC curve analysis on validation data.
- **Decision Rule:**
  - If `Composite_Score ≥ Threshold`, label as potential plagiarism.

#### **Evaluation**

- **Datasets:**
  - Test set with known plagiarism cases, including:
    - **Artificial Cases:** Created by introducing known modifications to melodies.
    - **Real-World Cases:** Legal cases of alleged plagiarism.
- **Performance Metrics:**
  - **Confusion Matrix:** True positives, false positives, true negatives, false negatives.
  - **Detection Metrics:** Precision, Recall, F1-score, Accuracy.
  - **ROC and AUC:** Receiver Operating Characteristic curve and Area Under the Curve.
- **Expert Evaluation:**
  - Have music experts assess a sample of detected cases for qualitative analysis.

#### **Comparative Analysis**

- **Baseline Methods:**
  - **Alignment-Only Approach:** Use traditional DTW without embeddings.
  - **Embedding-Only Approach:** Use embedding similarity without alignment verification.
- **Ablation Study:**
  - Evaluate the impact of removing Soft-DTW from the system.
  - Test different weighting schemes for composite scoring.

---

## **Data Considerations**

- **Ethical Compliance:**
  - Ensure all datasets are used in accordance with licensing agreements.
  - Anonymize any sensitive data if necessary.
- **Data Diversity:**
  - Include a variety of musical genres, cultures, and historical periods.
- **Quality Assurance:**
  - Validate the accuracy and consistency of the datasets.
- **Data Splits:**
  - Use consistent training, validation, and test splits across experiments.

---

## **Next Steps**

1. **Implementation:**
   - Begin coding the models and systems as per the specifications.
   - Use modular programming to reuse components across experiments.

2. **Testing:**
   - Start with small subsets of data to verify functionality.
   - Debug and optimize code for efficiency and accuracy.

3. **Hyperparameter Tuning:**
   - Use grid search or Bayesian optimization to find optimal hyperparameters.
   - Monitor training and validation losses to prevent overfitting.

4. **Evaluation and Analysis:**
   - Collect and analyze results according to the specified metrics.
   - Use statistical tests to assess the significance of improvements.

5. **Documentation:**
   - Keep detailed records of experiments, including configurations and outcomes.
   - Prepare visualizations (graphs, charts) to support your findings.

6. **Expert Collaboration:**
   - Engage with musicologists or legal experts to interpret results, especially for Experiment 3.

---