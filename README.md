# So this is something

## Yolo SAM pipeline

1. [camus_yolo_dataset_builder.ipynb](./camus_yolo_dataset_builder.ipynb) - this converts the input CAMUS dataset into easy to use with yolo training format

2. Create dataset.yaml pointing to the dataset directory

3. [yolo_train_ultralytics.py](./yolo_train_ultralytics.py) - this uses ultralytics library to train yolo9 on CAMUS dataset.

4. [inference_yolo.ipynb](./inference_yolo.ipynb) - Inference on the trained yolo model. References the model from the folder that ultralytics creates in the prev script

5. [camus_sam_dataset_builder.ipynb](./camus_sam_dataset_builder.ipynb) - Builds the dataset for SAM - it has a different format. The output is put as images in folder, and also as huggingface dataset.

6. [sam_train_accelerate.py](./sam_train_accelerate.py) - train SAM on the dataset made earlier (5.). Uses accelerate to handle multi-gpu training, and mixed precision training (making training faster)

7. [inference_yolo_sam.ipynb](./inference_yolo_sam.ipynb) - The entire inference pipeline - from input image - yolo - sam - output.

8. [convert_sam_to_hf.py](./convert_sam_to_hf.py) - To use the [MedSAM Checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) from the MedSAM paper, you need to convert the checkpoint to hf format (hf format makes it easier to use)

    `python convert_sam_to_hf.py --model_name sam_vit_b_01ec64 --checkpoint_path medsam_vit_b.pth --pytorch_dump_folder_path medsam`

## Cardiac Phase Detector

1. [Cardiac_Phase_Detector_training.ipynb](./Cardiac_Phase_Detector_training.ipynb) - Reference Notebook

2. Dataset generation code

3. Training Script

4. Validation + Reporting Script