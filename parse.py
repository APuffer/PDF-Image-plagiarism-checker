import asyncio
import os
from io import BytesIO

import fitz
from tqdm import tqdm

from utils import is_gray_image, is_small_image


async def async_extract_images_from_pdf(pdf_path: str) -> tuple[str, dict[str, bytes]]:
    """
    异步从PDF中提取图片
    :param pdf_path: PDF文件路径
    :return: 图片字典 {图片ID: 图片二进制数据}
    """
    import aiofiles
    from io import BytesIO

    async with aiofiles.open(pdf_path, 'rb') as f:
        pdf_bytes = await f.read()

    # 必须用BytesIO包装，因为fitz.open需要文件流
    pdf_stream = BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream)

    images = {}
    imgcount = 0
    for page in doc:
        for img in page.get_images():
            imgcount += 1
            img_id = f"{page.number}-{imgcount}"
            img_data = BytesIO()
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            pix.pil_save(img_data, "png")
            images[img_id] = img_data.getvalue()

    return pdf_path, images

async def async_write_image(pdf, images, out_dir, grayimg_dir):
    team = pdf.split('\\')[-1].split('队')[0] + '队'
    ipath = os.path.join(out_dir, team) + '/'
    if not os.path.exists(os.path.dirname(ipath)):
        os.makedirs(os.path.dirname(ipath))
    for name, image in images.items():
        if is_small_image(image, min_size=50 * 50):
            continue
        if is_gray_image(image, method='variance'):
            is_gray, eps = is_gray_image(image, return_eps=True)
            filename = f'{team}_{name}_eps{eps:.2f}.png'
            grayimg_path = os.path.join(grayimg_dir, filename)
            with open(grayimg_path, 'wb') as f:
                f.write(image)
            continue
        filename = name + '.png'
        with open(os.path.join(ipath, filename), 'wb') as f:
            f.write(image)

async def async_parse_main(pdfs, out_dir, grayimg_dir):
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))
    if not os.path.exists(os.path.dirname(grayimg_dir)):
        os.makedirs(os.path.dirname(grayimg_dir))

    parse_tasks = [async_extract_images_from_pdf(pdf) for pdf in pdfs]
    write_tasks = []

    for future in tqdm(asyncio.as_completed(parse_tasks), total=len(parse_tasks), desc="正在从论文中提取并缓存图片……"):
        pdf, images = await future
        write_tasks.append(async_write_image(pdf, images, out_dir, grayimg_dir))

    for future in tqdm(asyncio.as_completed(write_tasks), total=len(write_tasks), desc="正在处理图片……"):
        await future

def parse_main(pdfs, out_dir, grayimg_dir):
    asyncio.run(async_parse_main(pdfs, out_dir, grayimg_dir))


if __name__ == '__main__':
    import glob
    # pdfs = glob.glob('./纯论文/2553队C题.pdf')
    pdfs = glob.glob('./纯论文/*.pdf')
    out_dir = './各论文图片/'
    grayimg_dir = './各论文图片/疑似灰图/'

    parse_main(pdfs, out_dir, grayimg_dir)
