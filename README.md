# PsyIntent
\[Under review\] Official repository for "Psychological-Inspired Visual Intent Recognition via Multi-modal Large Language Models"

Visual intent recognition in social media involves analyzing user-uploaded images to infer implicit psychological intent, which supports the detection of public opinion and the underlying motivations. However, existing methods primarily emphasize visual feature extraction for intent mining, often neglecting subtle psychological cues essential for revealing users' underlying intent in social images. To address this, we propose a Psychological-Inspired Transformer (PsyIntent) that employs a pre-trained multimodal large language model (MLLM) to capture latent psychological cues. These cues guide the fusion of multimodal interactions and uncover user motivations beyond visual features. Specifically, a psychological feature extraction module models image visual features as well as text features by extracting psychological cues using MLLM. Subsequently, a visual-psychological interaction module mitigates modality gaps via global semantic alignment, enhancing multimodal feature integration to yield psychologically enriched visual representations. Finally, a query generation component produces image-specific learnable queries guided by psychological semantics to decode intent based on enriched visual representations. Collectively, these components enable PsyIntent to accurately and efficiently extract implicit user intent. Extensive experiments validate the method's efficacy.


## Overview

![Model Overview](method.png)  

The proposed PsyIntent architecture first extracts visual, psychological analysis, and emotional features from an image. These features are then fused through cross-modal interactions to produce enriched visual representations, with a global semantic alignment component reinforcing their associations. Finally, an intent decoding module,featuring a psychological-aware query generator that creates learnable intent queries from the psychological text features,feeds both the queries and enriched representations into a transformer decoder for intent prediction.
