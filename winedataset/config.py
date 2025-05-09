
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_biz_data_dir(biz_name: str) -> str:
    """
    获取业务数据目录
    """
    biz_dir = os.path.join(PROJECT_ROOT, 'data', biz_name)
    if not os.path.exists(biz_dir):
        os.makedirs(biz_dir)
    return biz_dir
