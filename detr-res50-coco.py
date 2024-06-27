from transformers import AutoImageProcessor
from datasets import load_dataset
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer
from accelerate import Accelerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from transformers.image_transforms import center_to_corners_format
from functools import partial

import albumentations
import numpy as np
import torch
from dataclasses import dataclass

from huggingface_hub import login

import wandb
import os
from PIL import Image

login(token = "",write_permission=True)

wandb.require("core")
wandb.login(key="")


os.environ["WANDB_PROJECT"] = "detr-res50-coco-caronly" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

#csfg_caronly = load_dataset("data/CityscapesHF/leftImg8bit_foggy_caronly")
origin = "CityscapesHFjpg/leftImg8bit_caronly"
data = {"train": f"{origin}/train/metadata.jsonl", "validation": f"{origin}/val/metadata.jsonl"}
cs_caronly = load_dataset("json", data_files = data)

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def transform_aug_ann_test(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    examples["image"] = [Image.open(f"{origin}/train/{file[:-4]}.jpg") for file in examples['file_name']]
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

def transform_aug_ann_val(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    examples["image"] = [Image.open(f"{origin}/val/{file[:-4]}.jpg") for file in examples['file_name']]
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

def convert_bbox_yolo_to_pascal(boxes, image_size):
    boxes = center_to_corners_format(boxes)

    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    for batch in targets:
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    metric1 = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
    metric1.update(post_processed_predictions, post_processed_targets)
    metric1 = metric1.compute()
    metric1 = {k: round(v.item(), 4) for k, v in metric1.items()}

    metric2 = CompleteIntersectionOverUnion(box_format="xyxy", class_metrics=False)
    metric2.update(post_processed_predictions, post_processed_targets)
    metric2 = metric2.compute()
    metric2 = {k: round(v.item(), 4) for k, v in metric2.items()}

    metrics = dict(metric1, **metric2)
    return metrics

cs_caronly["train"] = cs_caronly["train"].with_transform(transform_aug_ann_test)
cs_caronly["validation"] = cs_caronly["validation"].with_transform(transform_aug_ann_val)
categories = ["car"]
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

training_args = TrainingArguments(
    output_dir="results/detr-r50-coco-caronly",
    push_to_hub=True,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_steps=500,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    load_best_model_at_end = True,
    eval_strategy = "steps",
    hub_model_id="carbohydration/detr-coco-caronly",
    report_to="wandb"
    )

accelerator = Accelerator()

trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=cs_caronly["train"],
    eval_dataset = cs_caronly["validation"],
    tokenizer=image_processor,
    compute_metrics=eval_compute_metrics_fn,
))

trainer.train()
trainer.push_to_hub()
