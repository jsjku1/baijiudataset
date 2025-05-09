
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.getcwd())

from paddleocr import PaddleOCR
import pandas as pd

from jiuxian_com.jiuxian_com_spider import BIZ_DATA_DIR, PRODUCT_DESC_IMAGE_DIR


RESULT_PATH = os.path.join(BIZ_DATA_DIR, 'result.csv')
HANDLE_RESULT_PATH = os.path.join(BIZ_DATA_DIR, 'handle_result.csv')
INGREDIENTS_LIST_PATH = os.path.join(BIZ_DATA_DIR, 'ingredients.txt')  # 原料表
FLAVOR_LIST_PATH = os.path.join(BIZ_DATA_DIR, 'flavor.txt')  # 香型表

PROCESSING_PATTERN = {
    '固态法': re.compile(r'(固态|10781|26760|纯粮)'),  # 数字部分为工艺法对应的执行标准
    '液态法': re.compile(r'(液态|20821|新工艺)'),
    '固液法': re.compile(r'(固液|20822)'),
}

OCR_TEXT_CACHE_DIR = os.path.join(BIZ_DATA_DIR, 'ocr_text_cache')
if not os.path.exists(OCR_TEXT_CACHE_DIR):
    os.makedirs(OCR_TEXT_CACHE_DIR)

ocr = PaddleOCR(use_angle_cls=True, lang='ch')


def main():
    """
    统一清洗处理数据
    """
    df = pd.read_csv(RESULT_PATH)

    # 读取原料表
    ingredient_set = set()
    with open(INGREDIENTS_LIST_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
        for item in content.split(','):
            ingredient_set.add(item.strip())
    
    # 读取香型表
    flavor_set = set()
    with open(FLAVOR_LIST_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
        for item in content.split(','):
            flavor_set.add(item.strip())

    for index, row in df.iterrows():
        product_id = str(row['product_id'])
        name = row['product_name']

        print(f'[处理中] {product_id} {name}')

        # 处理酒精度 统一转化为数字
        alcohol_degree = row['alcohol_degree']
        if not pd.isna(alcohol_degree):
            alcohol_degree = alcohol_degree.replace('°', '').replace('度', '').replace('%vol', '').replace('%', '').replace('vol', '')
            df.loc[index, 'alcohol_degree'] = alcohol_degree
        else:
            extract_alcohol_degree = re.findall(r'(\d+)[°度%]', name)
            if extract_alcohol_degree:
                df.loc[index, 'alcohol_degree'] = float(extract_alcohol_degree[0])
            
        row_ingredients = row['ingredients']
        if not pd.isna(row_ingredients):
            # 统一处理原料
            new_ingredients = []
            for ing in ingredient_set:
                if ing in row_ingredients:
                    new_ingredients.append(ing)
            df.loc[index, 'ingredients'] = ','.join(new_ingredients)
        
        # 白酒风味
        row_flavor = row['flavor']
        if not pd.isna(row_flavor):
            for flavor in flavor_set:
                if flavor in row_flavor:
                    df.loc[index, 'flavor'] = flavor
                    break
        else:
            for flavor in flavor_set:
                if flavor in name:
                    df.loc[index, 'flavor'] = flavor
                    break
        # 制造工艺
        row_process = row['production_process']
        if not pd.isna(row_process):
            if row_process not in PROCESSING_PATTERN.keys():
                for process, pattern in PROCESSING_PATTERN.items():
                    if pattern.search(row_process):
                        df.loc[index, 'production_process'] = process
                        break
                    if pattern.search(name):
                        df.loc[index, 'production_process'] = process
                        break

        # 读取图片
        image_dir = os.path.join(PRODUCT_DESC_IMAGE_DIR, product_id)
        if not os.path.exists(image_dir):
            continue
            
        # 是否缓存了OCR结果
        ocr_text_cache_path = os.path.join(OCR_TEXT_CACHE_DIR, f'{product_id}.txt')
        if os.path.exists(ocr_text_cache_path):
            with open(ocr_text_cache_path, 'r', encoding='utf-8') as f:
                all_text = f.read()
        else:
            image_files = os.listdir(image_dir)
            all_text = []
            for image_file in image_files:
                if image_file in ('0.jpg', '1.jpg'):
                    continue
                image_path = os.path.join(image_dir, image_file)
                try:
                    result = ocr.ocr(image_path)
                    for item in result:
                        for line in item:
                            all_text.append(line[1][0])
                except Exception as e:
                    traceback.print_exc()
            all_text = ','.join(all_text)
            with open(ocr_text_cache_path, 'w', encoding='utf-8') as f:
                f.write(all_text)

        # 如果风味为空，尽可能从OCR文本中提取风味
        flavor = df.loc[index, 'flavor']
        if pd.isna(flavor):
            for f in flavor_set:
                if f in all_text:
                    df.loc[index, 'flavor'] = f
                    print(f'{product_id} 风味: {f}')
                    break
        # 如果原料为空，尽可能从OCR文本中提取原料
        ingredients = df.loc[index, 'ingredients']
        if pd.isna(ingredients):
            new_ingredients = []
            for ing in ingredient_set:
                if ing in all_text:
                    new_ingredients.append(ing)
            if new_ingredients:
                df.loc[index, 'ingredients'] = ','.join(new_ingredients)
                print(f'{product_id} 原料: {new_ingredients}')
        # 如果制造工艺为空，尽可能从OCR文本中提取制造工艺
        process = df.loc[index, 'production_process']
        if process not in PROCESSING_PATTERN.keys():
            for p, pattern in PROCESSING_PATTERN.items():
                if pattern.search(all_text):
                    df.loc[index, 'production_process'] = p
                    print(f'{product_id} 工艺: {p}')
                    break
        
        # 处理风味
        flavor = df.loc[index, 'flavor']
        if flavor not in flavor_set:
            df.loc[index, 'flavor'] = '未知'
        # 处理原料
        ingredients = df.loc[index, 'ingredients']
        if pd.isna(ingredients):
            df.loc[index, 'ingredients'] = '未知'
        # 处理酒精度
        alcohol_degree = df.loc[index, 'alcohol_degree']
        if not pd.isna(alcohol_degree):
            try:
                float(alcohol_degree)
            except Exception as e:
                df.loc[index, 'alcohol_degree'] = '未知'
        else:
            df.loc[index, 'alcohol_degree'] = '未知'
        # 处理产地
        origin = df.loc[index, 'origin']
        if pd.isna(origin):
            df.loc[index, 'origin'] = '未知'
        # 处理制造工艺
        production_process = df.loc[index, 'production_process']
        if production_process not in PROCESSING_PATTERN.keys():
            df.loc[index, 'production_process'] = '未知'
        
    df.to_csv(HANDLE_RESULT_PATH, index=False)

    
if __name__ == '__main__':
    main()


