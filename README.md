# GeoConcept-XAI
### Spectral-Spatial Explainable Remote Sensing Using Concept-Based Counterfactual Intelligence and Geographic Reliability Assessment

<p align="center">
  <strong>Explainable AI for Multispectral Satellite Image Classification</strong>
</p>

## 1. Project Overview

**GeoConcept-XAI** is an explainable artificial intelligence framework designed to improve the transparency, interpretability and reliability of multispectral remote sensing image classification.

Traditional deep learning models can classify satellite images with high accuracy, but understanding *why* a model predicts a particular land-cover class remains challenging. Existing explainability techniques often focus on individual pixels or spectral bands without identifying meaningful, recurring spectral-spatial concepts.

GeoConcept-XAI addresses this challenge by combining spectral-spatial feature learning, concept discovery, concept-based counterfactual explanations and geographic reliability assessment.

The framework uses multispectral Sentinel-2 imagery and provides a research dashboard through which users can explore satellite images, view classification results, inspect learned concepts, generate explanations and evaluate model reliability.

## 2. Problem Statement

Conventional remote sensing classification models provide predictions without sufficiently explaining the spectral and spatial patterns responsible for their decisions.

Existing Explainable AI methods often lack:
- Identification of meaningful and recurring spectral-spatial concepts.
- Concept-level counterfactual explanations.
- Evaluation of explanation reliability under image perturbations.
- Assessment of model generalisation across different geographic regions.

Therefore, a unified framework is required to make remote sensing predictions more interpretable, explainable and geographically reliable.

## 3. Project Objectives

- Develop a multispectral spectral-spatial deep learning model.
- Identify meaningful concepts from intermediate model representations.
- Generate spectral-spatial concept fingerprints.
- Explain predictions using multiple XAI techniques.
- Generate concept-level counterfactual interventions.
- Build a Concept Evidence Graph to represent relationships between learned concepts.
- Evaluate explanation consistency, faithfulness, robustness and geographic reliability.
- Provide an interactive dashboard for exploring model predictions and explanations.

## 4. Dataset

**EuroSAT Multispectral (EuroSAT-MS)**

| Dataset property | Details |
|---|---|
| Dataset | EuroSAT-MS |
| Total image patches | 27,000 |
| Spectral bands | 13 Sentinel-2 bands |
| Land-cover classes | 10 |
| Input format | Multispectral satellite image patches |
| Main task | Land-cover classification |

### Land-cover classes

1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. Sea and Lake

The preprocessing pipeline includes band-order validation, normalization using training-set statistics and reproducible data splitting.

## 5. Proposed System Architecture

GeoConcept-XAI follows a shared spectral-spatial representation architecture.

### Processing pipeline

```text
             Sentinel-2 Multispectral Images
                         |
                         v
                Data Preprocessing
          (Band Validation and Normalization)
                         |
              +----------+----------+
              |                     |
              v                     v
       Spectral Branch        Spatial Branch
          1D-CNN                ResNet-18
              |                     |
              v                     v
       Band Attention        Spatial Attention
              |                     |
              +----------+----------+
                         |
                         v
                Gated Feature Fusion
                         |
                         v
                 Transformer Layer
                         |
                         v
                 10-Class Classifier
                         |
              +----------+----------+
              |                     |
              v                     v
        Concept Discovery      Classification
              |                     |
              v                     |
       Concept Fingerprints          |
              |                     |
              +----------+----------+
                         |
                         v
                Explainability Engine
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Grad-CAM           IG          LIME / SHAP
          |              |              |
          +--------------+--------------+
                         |
                         v
             Concept Counterfactuals
                         |
                         v
              Concept Evidence Graph
                         |
                         v
            Geographic Reliability
                         |
                         v
                Research Dashboard
```

## 6. Core Features and Modules

### 6.1 Multispectral Image Explorer
- Browse EuroSAT satellite image samples.
- Support a validated 13-band GeoTIFF upload workflow.
- Provide a separately labelled RGB demonstration mode.
- Display spectral-band information.
- Generate land-cover classification results.

### 6.2 Spectral-Spatial Classification
- Uses a 1D-CNN-based spectral feature extraction branch.
- Uses a ResNet-18-based spatial feature extraction branch.
- Applies band and spatial attention.
- Combines features using gated fusion and a Transformer layer.
- Produces predictions for 10 land-cover classes.

### 6.3 Concept Discovery
- Extracts intermediate spatially indexed latent representations.
- Discovers concepts using K-Means, Gaussian Mixture Models (GMM) and optional HDBSCAN.
- Generates representative concept prototypes.
- Identifies recurring spectral-spatial patterns.

### 6.4 Concept Fingerprinting
- Generates concept-level spectral-spatial fingerprints.
- Displays spectral profiles and band-importance information.
- Provides concept membership and confidence indicators.
- Associates discovered concepts with class information and representative patches.

### 6.5 Explainability Engine

The framework provides multiple explanation methods:

| Method | Purpose |
|---|---|
| Grad-CAM | Highlights important image regions |
| Integrated Gradients | Measures input-feature contributions |
| LIME | Explains predictions through local approximations |
| SHAP | Estimates feature contributions |
| Concept-based XAI | Explains predictions using learned concepts |

### 6.6 Concept-Based Counterfactual Intelligence
- Allows users to select a learned concept.
- Supports learned concept-gate interventions.
- Provides a latent-direction baseline for comparison.
- Measures validity, proximity, sparsity, plausibility, locality and stability.
- Compares predictions before and after concept interventions.

### 6.7 Concept Evidence Graph
- Represents discovered concepts as graph nodes.
- Displays measured relationships between concepts.
- Supports relationships such as feature similarity, co-occurrence, spatial adjacency and class association.
- Helps users explore connections between learned concepts.

### 6.8 Geographic Reliability Assessment
- Evaluates concept-level consistency.
- Measures explanation faithfulness.
- Examines spectral stability.
- Compares geographic generalisation.
- Assesses robustness under perturbations.
- Displays geographic coverage and reliability information.

### 6.9 Research Evaluation Centre
- Displays classification metrics.
- Presents stored model evaluation results.
- Tracks experiment status.
- Supports comparison of completed and pending experiments.
- Helps researchers inspect model performance and reliability.

## 7. Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| Deep Learning | 1D-CNN, ResNet-18-based architecture, Transformer |
| Explainable AI | Grad-CAM, Integrated Gradients, LIME, SHAP |
| Concept Discovery | K-Means, GMM, HDBSCAN |
| Graph Processing | NetworkX |
| Database | SQLite |
| Dataset | EuroSAT Multispectral |
| Image Processing | Multispectral GeoTIFF processing |
| API Communication | REST APIs |

## 8. Implementation Workflow

1. **Dataset Preparation:** Load EuroSAT multispectral images and validate the 13-band configuration.
2. **Preprocessing:** Apply normalization using training-set statistics and prepare reproducible data splits.
3. **Model Development:** Build spectral and spatial feature extraction branches.
4. **Feature Fusion:** Combine spectral and spatial features for classification.
5. **Concept Discovery:** Extract intermediate representations and identify concepts using clustering.
6. **Fingerprint Generation:** Generate concept prototypes and spectral-spatial fingerprints.
7. **Explainability:** Integrate Grad-CAM, Integrated Gradients, LIME, SHAP and concept-based explanations.
8. **Counterfactual Analysis:** Apply concept-level interventions and calculate evaluation metrics.
9. **Graph Construction:** Build a Concept Evidence Graph using measured concept relationships.
10. **Reliability Evaluation:** Assess explanation reliability and geographic generalisation.
11. **Dashboard Integration:** Connect the model and analysis modules to FastAPI and the React frontend.

## 9. Model Evaluation and Current Results

The project presentation and dashboard recording report the following results for the selected hybrid model:

| Evaluation metric | Reported result |
|---|---:|
| Classification accuracy | 94.82% |
| Macro F1-score | 92.04% |
| Precision | 90.72% |
| Recall | 95.00% |
| Model parameters | 0.96 million |
| Model size | 3.9 MB |
| Training time | 31.4 minutes |
| Inference time | 2.64 ms |

**Sample prediction:** The recorded Image Explorer demonstration classified sample `eurosat:39` as AnnualCrop with 96.6% confidence.

**Geographic evaluation:** The presentation reports 90.9% unseen-region performance, while the dashboard recording displays 91.4%. These values require reconciliation against the saved experiment results before being treated as the final geographic evaluation.

### Counterfactual evaluation

The recorded comparison reports:
- Learned concept gate: prediction changed in 44% of cases.
- Latent-direction baseline: prediction changed in 70% of cases at full intervention strength.

The learned gate was improved using a margin-hinge objective on log-activation to address softmax saturation. Further validation and improvement of intervention effectiveness remain important Phase-2 tasks.

### Testing status

The project presentation reports:
- 70 automated test cases passed.
- Five development-time failures were fixed.

Reported test groups:

| Test module | Test cases |
|---|---:|
| Preprocessing | 9 |
| Model | 13 |
| Concepts | 7 |
| Fingerprints | 5 |
| Counterfactuals | 8 |
| Graph | 6 |
| Reliability | 9 |
| API | 13 |
| **Total** | **70** |

*Note: These testing results are reported from the project presentation. The supplied demonstration archive contained a screen recording rather than the source-code repository and test logs; therefore, the test suite was not independently rerun for this report.*

## 10. Dashboard Screenshots

Add the actual screenshots of the implemented application in this section.

Suggested GitHub folder:

```text
assets/
├── dashboard-overview.png
├── image-explorer.png
├── explanation-studio.png
├── concept-library.png
├── concept-fingerprint.png
├── counterfactual-lab.png
├── concept-evidence-graph.png
├── geographic-reliability.png
└── evaluation-centre.png
```

Example Markdown:

```markdown
### Dashboard Overview
![Dashboard Overview](assets/dashboard-overview.png)

### Multispectral Image Explorer
![Image Explorer](assets/image-explorer.png)

### Explanation Studio
![Explanation Studio](assets/explanation-studio.png)

### Concept Library
![Concept Library](assets/concept-library.png)

### Counterfactual Analysis
![Counterfactual Lab](assets/counterfactual-lab.png)

### Concept Evidence Graph
![Concept Evidence Graph](assets/concept-evidence-graph.png)

### Geographic Reliability
![Geographic Reliability](assets/geographic-reliability.png)
```

Replace these image paths with the actual screenshot filenames committed to the repository.

## 11. Current Implementation Status

**Phase-1 completion: Approximately 70%**

| Milestone | Progress | Implementation |
|---|---:|---|
| Week 8 – Core implementation | 30% | Dataset preparation, preprocessing and primary classification model |
| Week 10 – Feature development | 55% | Concept discovery, fingerprints, explainability and counterfactual modules |
| Week 14 – Phase-1 consolidation | 70% | Concept graph, geographic reliability, API, dashboard and reported testing |

## 12. Future Scope – Phase 2

The following work is planned:

- **Baseline and ablation experiments:** Complete the five queued experiments comparing RGB/multispectral inputs, CNN/hybrid architectures and module contributions.
- **External dataset validation:** Evaluate transfer performance using SEN12MS and BigEarthNet after checking band compatibility, spatial resolution and label harmonisation.
- **Manual concept validation:** Review discovered prototypes and confirm semantic concept labels.
- **Vision-language integration:** Explore optional vision-language models for concept naming.
- **Multi-seed evaluation:** Repeat key experiments using three seeds and calculate confidence intervals.
- **Research publication:** Consolidate results, figures and tables for the final research report/manuscript.

The separate Explanation Drift Detection engine is outside the current Phase-1 implementation scope.

## 13. Expected Impact

GeoConcept-XAI aims to support more transparent and reliable remote sensing applications by connecting model predictions with interpretable spectral-spatial concepts.

Potential application areas include:
- Land-cover classification.
- Agricultural monitoring.
- Forest and vegetation analysis.
- Urban development monitoring.
- Environmental observation.
- Remote sensing research and model evaluation.

## 14. Project Team

**Institution:** REVA University  
**Department:** Center for Artificial Intelligence and Machine Learning  
**Project Guide:** Mr. Khandgonda Nitin

| Team Member | SRN |
|---|---|
| Deeksha S | R23EF066 |
| Dhanyashree H | R23EF071 |
| Manoj K | R23EF133 |
| Meghana S H | R23EF138 |

## 15. Conclusion

GeoConcept-XAI presents a unified approach to explainable multispectral remote sensing by combining spectral-spatial deep learning, concept discovery, concept-based counterfactual intelligence and geographic reliability assessment.

Rather than providing only a land-cover prediction, the framework aims to reveal the concepts influencing the prediction, examine how interventions affect model decisions and evaluate whether explanations remain reliable across different conditions and geographic regions.

The Phase-1 implementation establishes the core model, concept analysis pipeline, explanation modules, reliability assessment and interactive research dashboard. Further experiments, external validation and multi-seed evaluation are planned for Phase 2.

---

**Project:** GeoConcept-XAI  
**Institution:** REVA University  
**Domain:** Artificial Intelligence | Explainable AI | Remote Sensing | Deep Learning  
**Status:** Phase-1 – Approximately 70% Complete
