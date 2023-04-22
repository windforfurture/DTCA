# python install pillow
from PIL import Image


# 分割图片
def cut_image(image, row_count,col_count):
    width, height = image.size
    item_width = int(width / row_count)
    item_height = int(height/col_count)
    box_list = []
    # (left, upper, right, lower)
    for j in range(row_count):
        for i in range(col_count):
            box = (i * item_width, j * item_height, (i + 1) * item_width, (j + 1) * item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


# 保存分割后的图片
def save_images(image_list, direction=0):
    total_count = len(image_list)
    width, height = image_list[0].size
    interval = 0
    if direction == 0:
        interval = width // 8
    else:
        interval = width // 8
    if direction == 0:
        t_height = height
        t_width = total_count * width + (total_count - 1) * interval
    else:
        t_height =  total_count * height + (total_count - 1) * interval
        t_width = width
    target = Image.new('RGBA', (t_width,t_height ))
    pre = 0
    nxt = 0
    if direction == 0:
        nxt = width
    else:
        nxt = height
    for image in image_list:
        if direction == 0:
            target.paste(image, (pre, 0, nxt, height))
        else:
            target.paste(image, (0, pre, width, nxt))
        if direction == 0:
            pre += width + interval
            nxt += width + interval
        else:
            pre += height + interval
            nxt += height + interval
        # 图片的质量 0~100
    quantity_value = 100
    target.save("final1" + '.png', 'PNG',quantity=quantity_value)


if __name__ == '__main__':
    # file_path = "../datasets/images/twitter2015_images/228710.jpg"  # 要分割的图片地址
    file_path = "ppt.png"
    image = Image.open(file_path)  # 读取图片

    image_list = cut_image(image, 3, 3)  # 分割图片，分割成7张
    save_images(image_list, direction=1 )  # 保存分割后的图片