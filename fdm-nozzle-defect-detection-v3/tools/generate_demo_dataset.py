from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random, math

root = Path('/mnt/data/fdm-nozzle-defect-detection-v3')
random.seed(42)
W,H = 640,480
class_names = ['stringing','nozzle_blob','support_failure','detached']

def norm_bbox(box):
    x1,y1,x2,y2=box
    cx=((x1+x2)/2)/W; cy=((y1+y2)/2)/H
    bw=(x2-x1)/W; bh=(y2-y1)/H
    return cx,cy,bw,bh

def draw_base(draw):
    # build plate
    draw.rectangle([40,310,600,430], fill=(72,72,72), outline=(110,110,110), width=3)
    draw.line([60,410,580,410], fill=(130,130,130), width=2)
    # printer frame
    draw.line([80,50,80,440], fill=(95,95,95), width=4)
    draw.line([560,50,560,440], fill=(95,95,95), width=4)
    draw.line([80,70,560,70], fill=(95,95,95), width=4)
    # nozzle body
    draw.rectangle([285,70,355,120], fill=(70,80,90), outline=(170,170,170), width=2)
    draw.polygon([(305,120),(335,120),(325,160),(315,160)], fill=(165,165,165), outline=(230,230,230))
    draw.ellipse([314,154,326,166], fill=(45,45,45))
    # printed part base
    draw.rounded_rectangle([230,300,410,350], radius=6, fill=(190,120,80), outline=(210,150,100), width=2)

def add_noise(img):
    pix = img.load()
    for _ in range(600):
        x=random.randrange(W); y=random.randrange(H)
        r,g,b=pix[x,y]
        delta=random.randint(-15,15)
        pix[x,y]=(max(0,min(255,r+delta)),max(0,min(255,g+delta)),max(0,min(255,b+delta)))
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img

def draw_defect(draw, cls_id):
    # returns bbox
    if cls_id == 0: # stringing: thin line/wire web
        x1=random.randint(130,230); y1=random.randint(170,240)
        x2=random.randint(390,520); y2=random.randint(210,300)
        # multi thin lines
        for k in range(3):
            off=random.randint(-10,10)
            draw.line([x1, y1+off, x2, y2+random.randint(-12,12)], fill=(235,235,210), width=random.choice([1,1,2]))
        box=(min(x1,x2)-4,min(y1,y2)-16,max(x1,x2)+4,max(y1,y2)+16)
    elif cls_id == 1: # nozzle_blob: blob near nozzle
        cx=random.randint(300,340); cy=random.randint(155,210)
        rx=random.randint(18,34); ry=random.randint(18,38)
        draw.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], fill=(215,125,70), outline=(255,185,120), width=3)
        draw.ellipse([cx-8,cy-8,cx+12,cy+12], fill=(180,95,55))
        box=(cx-rx,cy-ry,cx+rx,cy+ry)
    elif cls_id == 2: # support_failure: broken support lines under part
        x0=random.randint(255,310); y0=random.randint(245,300)
        for k in range(4):
            x=x0+k*22
            draw.line([x,y0,x+random.randint(-20,20),y0+55], fill=(230,160,100), width=4)
        draw.line([x0-10,y0+25,x0+90,y0+random.randint(20,45)], fill=(230,160,100), width=3)
        box=(x0-20,y0-5,x0+110,y0+65)
    else: # detached: part shifted from bed
        x1=random.randint(255,330); y1=random.randint(250,285)
        x2=x1+random.randint(130,190); y2=y1+random.randint(45,75)
        draw.polygon([(x1,y1+20),(x2,y1),(x2-10,y2),(x1-15,y2+10)], fill=(205,120,72), outline=(255,180,120))
        draw.line([230,330,410,350], fill=(70,70,70), width=3)
        box=(x1-15,y1,x2,y2+10)
    # clamp
    x1,y1,x2,y2=box
    return (max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2))

def make_image(path_img, path_label, idx, cls_id):
    bg=(35+random.randint(-5,5),38+random.randint(-5,5),42+random.randint(-5,5))
    img=Image.new('RGB',(W,H),bg)
    draw=ImageDraw.Draw(img)
    draw_base(draw)
    box=draw_defect(draw, cls_id)
    # add occasional second defect
    labels=[]
    labels.append((cls_id, box))
    if random.random() < 0.2:
        other=(cls_id+random.randint(1,3))%4
        box2=draw_defect(draw, other)
        labels.append((other, box2))
    img=add_noise(img)
    img.save(path_img, quality=92)
    with open(path_label,'w',encoding='utf-8') as f:
        for c, b in labels:
            vals=norm_bbox(b)
            f.write(f"{c} " + " ".join(f"{v:.6f}" for v in vals) + "\n")

# Create dirs
splits = {'train':16,'val':4,'test':4}
for split,n in splits.items():
    (root/f'dataset/images/{split}').mkdir(parents=True, exist_ok=True)
    (root/f'dataset/labels/{split}').mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cls_id=i%4
        stem=f'{split}_{i:03d}_{class_names[cls_id]}'
        make_image(root/f'dataset/images/{split}/{stem}.jpg', root/f'dataset/labels/{split}/{stem}.txt', i, cls_id)

# Copy a few images to demo/test_images
for p in list((root/'dataset/images/test').glob('*.jpg'))[:4]:
    target=root/'demo/test_images'/p.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(p.read_bytes())

# README for dataset
(root/'dataset/README.md').write_text('''# Demo Dataset\n\n这是一个小型合成演示数据集，主要用于让仓库结构完整，并验证 YOLOv8 的 `data.yaml`、训练脚本和推理流程可以跑通。\n\n注意：这些图片是根据 3D 打印机喷嘴、热床、拉丝、喷嘴堆料、支撑失效、模型脱落等视觉特征绘制的 demo 样例，不等同于真实工业数据集。正式训练时应替换为真实采集和标注的数据。\n\n目录结构：\n\n```text\ndataset/\n├── images/\n│   ├── train/\n│   ├── val/\n│   └── test/\n└── labels/\n    ├── train/\n    ├── val/\n    └── test/\n```\n\n类别：\n\n```text\n0 stringing\n1 nozzle_blob\n2 support_failure\n3 detached\n```\n''', encoding='utf-8')
