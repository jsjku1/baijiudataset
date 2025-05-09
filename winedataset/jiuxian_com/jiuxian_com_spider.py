
import csv
import json
import os
import random
import re
import sys
import time

sys.path.append(os.getcwd())

import requests
from pyquery import PyQuery

from config import get_biz_data_dir
from utils.log import get_rotate_logger


BIZ_NAME = 'jiuxian_com'
BIZ_DATA_DIR = get_biz_data_dir(BIZ_NAME)

PRODUCT_DESC_IMAGE_DIR = os.path.join(BIZ_DATA_DIR, 'product_desc_image')
if not os.path.exists(PRODUCT_DESC_IMAGE_DIR):
    os.makedirs(PRODUCT_DESC_IMAGE_DIR)

_CATEGORY_LIST = [
    {
        'name': '白酒',
        'list_url': 'https://list.jiuxian.com/1-0-0-0-0-0-0-0-0-0-0-0.htm?area=6#v2'
    }
]

_LOGGER = get_rotate_logger(BIZ_NAME)

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
}


class BrandCacheManager:
    """
    品牌缓存管理
    """
    def __init__(self):
        self.cache_path = os.path.join(BIZ_DATA_DIR, 'brand_cache.json')
        self.has_crawl_brand_set = None
        self.load_brand_name()

    def load_brand_name(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.has_crawl_brand_set = set(json.load(f))
        else:
            self.has_crawl_brand_set = set()
    
    def save_brand_name(self):  
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.has_crawl_brand_set), f)

    def add_brand_name(self, brand_name: str):
        """
        添加品牌名称
        """
        self.has_crawl_brand_set.add(brand_name)

    def has_crawl_brand_name(self, brand_name: str):
        """
        判断品牌名称是否已爬取
        """
        return brand_name in self.has_crawl_brand_set


class ProductCacheManager:
    """
    商品缓存管理
    """
    def __init__(self):
        self.cache_path = os.path.join(BIZ_DATA_DIR, 'product_cache.json')
        self.has_crawl_product_ids = None
        self.load_product_ids()
    
    def load_product_ids(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.has_crawl_product_ids = set(json.load(f))
        else:
            self.has_crawl_product_ids = set()
    
    def save_product_ids(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.has_crawl_product_ids), f)

    def add_product_id(self, product_id: str):
        """
        添加商品id
        """
        self.has_crawl_product_ids.add(product_id)
    
    def has_crawl_product_id(self, product_id: str):
        """
        判断商品id是否已爬取
        """
        return product_id in self.has_crawl_product_ids
    
    
class Crawler:

    def __init__(self, product_cache_manager: ProductCacheManager):
        self.product_cache_manager = product_cache_manager

    def get_category_band_list(self, category_info: dict):
        """
        获取分类品牌列表
        """
        result = []
        category_list_url = category_info['list_url']

        response = requests.get(category_list_url, headers=_HEADERS)
        doc = PyQuery(response.text)
        brands = doc('li[c] a').items()
        for brand in brands:
            brand_name = brand.text()
            brand_url = brand.attr('href')
            item = {
                'category_name': category_info['name'],
                'brand_name': brand_name,
                'brand_url': 'https://list.jiuxian.com' + brand_url
            }
            result.append(item)
        return result

    def get_brand_product_list(self, brand_info: dict):
        """
        获取品牌商品列表
        """
        result = []
        brand_base_url = brand_info['brand_url']

        response = requests.get(brand_base_url, headers=_HEADERS)
        doc = PyQuery(response.text)
        total_page = doc('span.totalPage em').text()
        try:
            total_page = int(total_page)
        except Exception as e:
            total_page = 1

        for page in range(1, total_page + 1):
            _LOGGER.info(f'爬取品牌{brand_info["brand_name"]}第{page}页/共{total_page}页商品列表...')
            for _ in range(3):
                try:
                    page_url = f'{brand_base_url}&pageNum={page}'
                    response = requests.get(page_url, headers=_HEADERS)
                    doc = PyQuery(response.text)
                    products = doc('li[product-box]').items()
                    for product in products:
                        product_href = product('div.proName > a')
                        product_name = product_href.attr('title')
                        product_url = product_href.attr('href')
                        product_id = re.findall(r'/goods-(.*?).html', product_url)
                        if product_id:
                            product_id = product_id[0]
                        else:
                            product_id = None
                        item = {
                            'brand_name': brand_info['brand_name'],
                            'product_name': product_name,
                            'product_url': product_url,
                            'product_id': product_id,
                        }
                        result.append(item)
                    break
                except Exception as e:
                    time.sleep(random.randint(1, 2))
            # break  # for debug
        return result
    
    def get_product_detail(self, product_info: dict):
        """
        获取商品详情
        """
        product_url = product_info['product_url']
        product_id = product_info['product_id']

        if self.product_cache_manager.has_crawl_product_id(product_id):
            return None

        extend_info = {
            'flavor': None,
            'origin': None,
            'alcohol_degree': None,
            'ingredients': None,
            'production_process': None,
        }

        response = requests.get(product_url, headers=_HEADERS, timeout=10)

        doc = PyQuery(response.text)
        attrs = doc('ul.intrList > li').items()
        for attr in attrs:
            attr_text = attr.text()
            if '香型' in attr_text:
                flavor = attr('em').text()
                extend_info['flavor'] = flavor
            if '产地' in attr_text:
                origin = attr('em').text()
                extend_info['origin'] = origin
            if '酒精度' in attr_text:
                alcohol_degree = attr('em').text()
                extend_info['alcohol_degree'] = alcohol_degree
            if '原料' in attr_text:
                ingredients = attr('em').text()
                extend_info['ingredients'] = ingredients
            if '酿造工艺' in attr_text:
                production_process = attr('em').text()
                extend_info['production_process'] = production_process
        product_info.update(extend_info)

        rating = doc('li.comScore > em').text()
        try:
            rating = float(rating)
        except Exception as e:
            rating = 0
        product_info['rating'] = rating

        product_desc_image_dir = os.path.join(PRODUCT_DESC_IMAGE_DIR, product_id)
        if not os.path.exists(product_desc_image_dir):
            os.makedirs(product_desc_image_dir)
        images = doc('div.infoImg img').items()
        print(len(list(images)))
        for i, image in enumerate(images):
            image_url = image.attr('src')
            image_path = os.path.join(product_desc_image_dir, f'{i}.jpg')
            if os.path.exists(image_path):
                continue
            for _ in range(3):
                try:
                    response = requests.get(image_url, headers=_HEADERS, timeout=5)
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    break
                except Exception as e:
                    time.sleep(random.randint(1, 2))
        
        comment_tags = []
        url = f'https://www.jiuxian.com/pro/listProductEvaluate.htm?id={product_id}&pageNum=1&onlyImg=false'
        response = requests.get(url, headers=_HEADERS, timeout=5)
        doc = PyQuery(response.text)
        comments = doc('ul.tag-s-user-event li').items()    
        for comment in comments:
            comment_text = comment.text()
            comment_tags.append(comment_text)
        product_info['comment_tags'] = comment_tags

        return product_info
    
    def write_brand_product(self, product_info: dict):
        """
        写入品牌商品json
        """
        result_path = os.path.join(BIZ_DATA_DIR, 'result.csv')
        if not os.path.exists(result_path):
            with open(result_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=product_info.keys())
                writer.writeheader()
                writer.writerow(product_info)
        else:
            with open(result_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=product_info.keys())
                writer.writerow(product_info)


def main():
    """
    main
    """
    product_cache_manager = ProductCacheManager()
    brand_cache_manager = BrandCacheManager()
    crawler = Crawler(product_cache_manager)
    for category_info in _CATEGORY_LIST:
        brands = crawler.get_category_band_list(category_info)
        for brand_info in brands:
            brand_name = brand_info['brand_name']
            if brand_cache_manager.has_crawl_brand_name(brand_name):
                _LOGGER.info(f'品牌{brand_name}已爬取，跳过')
                continue
            _LOGGER.info(f'爬取品牌{brand_name}...')
            brand_product_cnt = 0
            products = crawler.get_brand_product_list(brand_info)
            for i, product_info in enumerate(products):
                if i % 5 == 0:
                    product_cache_manager.save_product_ids()
                product_id = product_info['product_id']
                if product_id and product_cache_manager.has_crawl_product_id(product_id):
                    _LOGGER.info(f'商品{product_id}已爬取，跳过')
                    continue
                _LOGGER.info(f'爬取商品{brand_name} {product_id}...')
                for _ in range(3):
                    try:
                        product_info = crawler.get_product_detail(product_info)
                        product_cache_manager.add_product_id(product_id)
                        crawler.write_brand_product(product_info)
                        break
                    except Exception as e:
                        time.sleep(random.randint(1, 2))
                time.sleep(random.randint(1, 2))
                brand_product_cnt += 1
                # break  # for debug
            product_cache_manager.save_product_ids()
            if brand_product_cnt > 0:
                brand_cache_manager.add_brand_name(brand_name)
                brand_cache_manager.save_brand_name()
            # break  # for debug
        


if __name__ == '__main__':
    main()
