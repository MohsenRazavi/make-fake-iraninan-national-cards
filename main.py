import os
import random

import albumentations as A
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FA_NUMBERS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']


def generate_national_code_fa():
    """تولید یک کد ملی ۱۰ رقمی رندوم با اعداد فارسی"""
    code_en = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    code_fa = ''.join(FA_NUMBERS[int(d)] for d in code_en)
    return code_en, code_fa


def write_national_code_on_national_card(image_path, text, position, font_path, font_size, text_color, bg_color):
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
        # 1. شبیه‌سازی نقص‌های دوربین
        A.OneOf([
            A.GaussianBlur(blur_limit=1, p=0.1),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussNoise(p=0.2),
        ], p=0.6),

        # 2. چرخش، مقیاس و جابجایی (ShiftScaleRotate به Affine تغییر کرد)
        A.Affine(
            scale=(0.95, 1.05),  # زوم جزئی
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},  # جابجایی جزئی
            rotate=(-1, 1),  # چرخش تا 1 درجه
            p=0.8
        ),

        # 3. تغییرات نوری
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),

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


# ==================CONFIG====================
SAMPLE_NATIONAL_CARD = './sample_national_card.png'
FONT_PATH = './Yekan.ttf'
NUMBER_OF_SAMPLES = 100
OUTPUT_DIR = 'samples'
TEXT_POSITION = (285, 83)
FONT_SIZE = 18
TEXT_COLOR = 'black'
BACKGROUND_COLOR = '#8EC5E1'
MAKE_REALISTIC = True


# ============================================

def main():
    augmentation_pipeline = get_augmentation_pipeline()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed()

    print(f"شروع تولید {NUMBER_OF_SAMPLES} نمونه...")

    for i in range(NUMBER_OF_SAMPLES):
        national_code_en, national_code_fa = generate_national_code_fa()
        fake_national_card = write_national_code_on_national_card(
            SAMPLE_NATIONAL_CARD,
            national_code_fa,
            TEXT_POSITION,
            FONT_PATH,
            FONT_SIZE,
            TEXT_COLOR,
            BACKGROUND_COLOR
        )
        fake_national_card.save(os.path.join(OUTPUT_DIR, f'{national_code_en}.png'))
        if MAKE_REALISTIC:
            realistic_fake_national_card = make_image_realistic(fake_national_card, augmentation_pipeline)
            realistic_fake_national_card.save(os.path.join(OUTPUT_DIR, f'{national_code_en}_realistic.png'))

        print(f"✅ نمونه {i + 1}/{NUMBER_OF_SAMPLES} با کد {national_code_fa} ذخیره شد.")

    print("\nعملیات با موفقیت به پایان رسید.")


if __name__ == "__main__":
    main()
