from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from PIL import Image
from PIL import ImageDraw
import torch
from tqdm import tqdm
from huggingface_hub import login
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from transformers.image_transforms import center_to_corners_format

login(token = "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")

def get_input(data):
    input = data ["validation"]["objects"]
    for entry in input:
        entry.pop("id")
        bbox = [[a[0],a[1],a[0]+a[2],a[1]+a[3]] for a in entry.pop("bbox")]
        entry["boxes"] = torch.tensor(bbox)
        entry["labels"] = torch.tensor(entry.pop("category"))
        entry["area"] = torch.tensor(entry["area"])
    return input

@torch.no_grad()
def get_output(origin,data,batch_size,processor,model):
    image_batch = [Image.open(f"{origin}/val/{img_file[:-4]}.jpg") for img_file in data["validation"]["file_name"]]
    outdict = []
    for x in tqdm(range(int(len(image_batch)/batch_size))):
        inputs = processor(images=image_batch[batch_size*x:batch_size*x+batch_size], return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        out = processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=torch.tensor([[1024,2048] for x in range(batch_size)])
        )
        outdict.extend(out)

    return outdict

def compute_metrics(preds, gt):

    for x in range(len(preds)-1,-1,-1):
        preds[x]['labels']=preds[x]['labels'].cpu()
        preds[x]['boxes']=preds[x]['boxes'].cpu()
        preds[x]['scores']=preds[x]['scores'].cpu()
        if len(gt[x]['boxes'])==0:
            preds.pop(x)
            gt.pop(x)
            
    metric1 = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
    metric1.update(preds, gt)
    metric1 = metric1.compute()
    metric1 = {k: round(v.item(), 4) for k, v in metric1.items()}

    metric2 = CompleteIntersectionOverUnion(box_format="xyxy", class_metrics=False)
    metric2.update(preds, gt)
    metric2 = metric2.compute()
    metric2 = {k: round(v.item(), 4) for k, v in metric2.items()}

    metrics = dict(metric1, **metric2)
    return metrics

def visualize(origin,inbox,outbox,image):
    im = Image.open(f"{origin}/val/{image[:-4]}.jpg")
    draw = ImageDraw.Draw(im)

    for box in inbox:
        draw.rectangle(tuple(box), outline="red", width=1)

    for box in outbox:
        draw.rectangle(tuple(box), outline="green", width=1)

    im.save("test.png","PNG")

def evaluate(origin,model_checkpoint):
    files = {"train": f"{origin}/train/metadata.jsonl", "validation": f"{origin}/val/metadata.jsonl"}
    data = load_dataset("json", data_files = files)

    categories = ["car"]
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    ).to("cuda")
    model.eval()

    inn = get_input(data)
    out = get_output(origin, data,1,processor,model)

    visualize(origin,inn[1]['boxes'],out[1]['boxes'],data['validation'][1]["file_name"])

    print(f"metrics for {origin} on {model_checkpoint} model")
    print(compute_metrics(out,inn))

evaluate("CityscapesHFjpg/leftImg8bit_foggy_caronly","carbohydration/detr-coco-caronly-vft")
evaluate("CityscapesHFjpg/leftImg8bit_caronly","carbohydration/detr-coco-caronly-vft")