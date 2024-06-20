import numpy as np
from statistics import mean
from skimage.transform import resize

import torch
from torch.optim import Adam
from torch.utils.data import Dataset

from datasets import load_from_disk
from transformers import SamModel, SamProcessor

import monai
import monai.metrics

from accelerate.utils import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


# facebook/sam-vit-large
# facebook/sam-vit-huge
# facebook/sam-vit-base

SAM_MODEL = "facebook/sam-vit-huge"
EPOCHS = 20


dataset = load_from_disk("/home/aistudent/camus/camus_sam_ds")
ds_train = dataset["train"]
ds_val = dataset["validation"]

# Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["image"] = item["image"].convert("RGB")
        image = np.array(item["image"])
        ground_truth_mask = np.array(item["label"])
        ground_truth_mask = resize(ground_truth_mask, (256, 256))

        prompt = get_bounding_box(ground_truth_mask)
        inputs = self.processor(
            images=image, input_boxes=[[prompt]], return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs


processor = SamProcessor.from_pretrained(SAM_MODEL)

train_dataset = SAMDataset(dataset=ds_train, processor=processor)
val_dataset = SAMDataset(dataset=ds_val, processor=processor)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=4
)
valid_dataloader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, drop_last=False
)

model = SamModel.from_pretrained(SAM_MODEL)

for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for epoch in range(EPOCHS):
    epoch_losses = []
    epoch_dice = []

    num_batches = len(train_dataloader)
    bar = tqdm(total=num_batches)

    for batch in train_dataloader:
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_boxes=batch["input_boxes"],
            multimask_output=False,
        )

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float()
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        epoch_losses.append(loss.item())
        gen_dice = monai.metrics.compute_generalized_dice(
            predicted_masks, ground_truth_masks.unsqueeze(1)
        )
        epoch_dice.append(accelerator.gather(gen_dice).mean().item())

        bar.set_description(
            f"Epoch: {epoch} Loss: {loss.item():.4f} Dice: {accelerator.gather(gen_dice).mean().item():.4f}"
        )
        bar.update()

    accelerator.print(f"EPOCH: {epoch}")
    accelerator.print(f"Mean Dice: {mean(epoch_dice)}")
    accelerator.print(f"Mean loss: {mean(epoch_losses)}")

    bar.close()

torch.save(model.module.state_dict(), "./sam-huge-camus.pth")
