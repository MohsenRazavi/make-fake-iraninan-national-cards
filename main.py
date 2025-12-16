from PIL import Image, ImageDraw, ImageFont
import random
import os
import cv2
import albumentations as A
import numpy as np

FA_NUMBERS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']

def generate_national_code_fa():
    """تولید یک کد ملی ۱۰ رقمی رندوم با اعداد فارسی"""
    code_en = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    code_fa = ''.join(FA_NUMBERS[int(d)] for d in code_en)
    return code_en, code_fa

def write_text_with_background(image_path, text, position, font_path, font_size, text_color, bg_color):
    """
    نوشتن متن روی تصویر PIL با پس زمینه مستطیلی برای پوشاندن متن قبلی.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"🛑 خطا: فایل تصویر {image_path} پیدا نشد.")
        return None

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"🛑 خطا: فایل فونت {font_path} پیدا نشد.")
        return None

    draw = ImageDraw.Draw(image)
    
    bbox = draw.textbbox(position, text, font=font)
    
    padding = 5
    rect_coords = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    
    draw.rectangle(rect_coords, fill=bg_color)
    
    draw.text(position, text, font=font, fill=text_color)
    
    return image


def get_augmentation_pipeline():
    """تعریف پایپ‌لاین تغییرات واقع‌گرایانه با رفع هشدارهای Albumentations."""
    transform = A.Compose([
        # 1. چرخش، مقیاس و جابجایی (ShiftScaleRotate به Affine تغییر کرد)
        A.Affine(
            scale=(0.95, 1.05), # زوم جزئی
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # جابجایی جزئی
            rotate=(-5, 5),   # چرخش تا 5 درجه
            cval=cv2.BORDER_REPLICATE, # پر کردن حاشیه
            p=0.8
        ),
        
        # 2. تغییرات نوری
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        
        # 3. شبیه‌سازی نقص‌های دوربین
        A.OneOf([
            A.GaussianBlur(blur_limit=1, p=0.1),
            A.MotionBlur(blur_limit=(1, 3), p=0.2),
            A.GaussNoise(var_limit=(1.0, 5.0), p=0.2), 
        ], p=0.6),
        
        # 4. تغییر جزئی در رنگ‌ها
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    ])
    return transform

def make_image_realistic(pil_image, pipeline):
    """
    اعمال تغییرات واقع‌گرایانه Albumentations بر روی یک تصویر PIL.
    """
    image_np = np.array(pil_image)
    
    image_rgb = image_np

    augmented = pipeline(image=image_rgb)
    augmented_image_rgb = augmented['image']

    augmented_pil_image = Image.fromarray(augmented_image_rgb)
    
    return augmented_pil_image


INPUT_IMAGE_PATH = './sample_national_card.png'
FONT_PATH = './Yekan.ttf'      
NUMBER_OF_SAMPLES = 1_000_000
OUTPUT_DIR = 'samples'
TEXT_POSITION = (285, 83)  
FONT_SIZE = 18
TEXT_COLOR = 'black'
BACKGROUND_COLOR = '#8EC5E1' 
MAKE_REALISTIC = True

augmentation_pipeline = get_augmentation_pipeline()
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed()

print(f"شروع تولید {NUMBER_OF_SAMPLES} نمونه...")

for i in range(NUMBER_OF_SAMPLES):
    national_code_en, national_code_fa = generate_national_code_fa()
    filename = f"{national_code_en}.png"

    img_with_text_pil = write_text_with_background(
        INPUT_IMAGE_PATH, 
        national_code_fa, 
        TEXT_POSITION, 
        FONT_PATH, 
        FONT_SIZE, 
        TEXT_COLOR, 
        BACKGROUND_COLOR
    )
    if MAKE_REALISTIC:
        img_final_pil = make_image_realistic(img_with_text_pil, augmentation_pipeline)
    
    img_final_pil.save(os.path.join(OUTPUT_DIR, filename))

    print(f"✅ نمونه {i+1}/{NUMBER_OF_SAMPLES} با کد {national_code_fa} ذخیره شد.")

print("\nعملیات با موفقیت به پایان رسید.")