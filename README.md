## scDAEC
scDAEC method for clustering scRNA-seq data
    We developed a novel cell type identification method for single-cell RNA sequencing (scRNA-seq) data that employs an adaptive Zero-Inflated Negative Binomial (adaptive-ZINB) model to effectively balance technical dropouts and genuine low-expressed genes, thereby enhancing the influence of these lowly expressed genes on clustering outcomes. The method simultaneously incorporates a dynamically composable multi-head attention mechanism to adaptively integrate two distinct clustering strategies, enabling more effective learning of cluster representations.

## run environment

python --- 3.8

pytorch --- 1.11.0

torchvision --- 1.12.0

torchaudio --- 0.11.0

scanpy --- 1.8.2

scipy --- 1.6.2

numpy --- 1.19.5

leidenalg --- 0.8.10


## Usage
For applying scDAEC, the convenient way is  run ["run_scDAEC.py"]

Please place the scRNA-seq dataset you want to analyze in the directory ["./Data/AnnData"], where is the default for model input.
If you want to calculate the similarity between the predicted clustering resluts and the true cell labels (based on NMI or ARI score), please transmit your true labels into the "adata.obs['celltype']" and setting the argument "-celltype" to **True**.
