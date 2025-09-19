
中文：
这段代码的功能是：批量读取指定文件夹中的病理图像（H&E 染色的胃癌组织切片），通过 InternVL2-8B 多模态大模型生成图像描述，并将生成的文本保存到与图像对应的 .txt 文件中。代码支持 大图的多块裁剪处理 与 小图的缩放处理，保证模型能同时获取局部和全局的组织学特征。

English:
This script processes a batch of histopathological images (H&E-stained gastric cancer tissue) by feeding them into the InternVL2-8B multimodal model to generate descriptive captions. The results are saved as .txt files corresponding to each image. The code supports multi-tile cropping for large images and rescaling for small images, ensuring the model captures both local details and global tissue structures.