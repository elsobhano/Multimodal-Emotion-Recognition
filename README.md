# Multimodal Emotion Recognition

## introduction

The final goal of this project is **Emotion Recognition** using visual and textual information.
Images and text are two different modalities, which have been used for the task of **Emotion Recognition** separetely. 

One of the simple and traditional way of mitimodal learning is concatanating features, which have been used from two separate feature extractor, of different types of information. Obviously, this approach is not enough when we are talking about different natures of information like text and image.

Here we have used pre-trained Transformer-Based models as a backbone and then finetune the whole model using visual and textual data. The chosen backbone is ClipBERT.


**ClipBERT**, an efficient framework for end-to-end learning for image-text and video-text tasks. It takes raw videos/images + text as inputs, and outputs task predictions. ClipBERT is designed based on 2D CNNs and transformers, and uses a sparse sampling strategy to enable efficient end-to-end video-and-language learning. For more detailed information about ClipBERT and its application check out this [link.](https://github.com/jayleicn/ClipBERT)




