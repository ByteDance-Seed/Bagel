from PIL import Image
import os
import json
def merge_with_last_spaced(image_paths, output_path, spacing=10, last_spacing=50, 
                          background_color=(255, 255, 255)):
    """
    水平排列图片，最后一张有更大的间距
    
    Args:
        image_paths: 图片路径列表
        output_path: 输出图片路径
        spacing: 普通图片间距
        last_spacing: 最后一张图与前面的间距
        background_color: 背景颜色
    """
    # 打开所有图片
    images = [Image.open(img_path) for img_path in image_paths]
    
    # 计算总宽度（考虑不同的间距）
    total_width = sum(img.width for img in images)
    
    # 前面的图片间距
    if len(images) > 1:
        total_width += spacing * (len(images) - 2)  # 前面图片之间的间距
        total_width += last_spacing  # 最后一张图的额外间距
    
    # 最大高度
    max_height = max(img.height for img in images)
    
    # 创建新图片
    new_image = Image.new('RGB', (total_width, max_height), background_color)
    
    # 粘贴图片
    x_offset = 0
    for i, img in enumerate(images):
        # 计算垂直居中位置
        y_offset = (max_height - img.height) // 2
        
        # 粘贴图片
        new_image.paste(img, (x_offset, y_offset))
        
        # 更新x偏移量（最后一张图之前使用更大的间距）
        if i < len(images) - 1:
            if i >= len(images) - 2:  # 倒数第二张，后面使用大间距
                x_offset += img.width + last_spacing
            else:  # 其他图片使用普通间距
                x_offset += img.width + spacing
    
    # 保存图片
    new_image.save(output_path)
    print(f"图片已保存到: {output_path}")
    return new_image

# 使用示例
if __name__ == "__main__":
    name = 'step_mix_refine0211_0020000_refine_step_prompt_matterport3d_test'
    image_root = '/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data'
    json_path = '/mnt/petrelfs/linjingli/UMM_Spatial/annotations/refine_step_prompt_matterport3d_test.jsonl'

    tmp_dir = f'/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/{name}'
    outdir = f'/mnt/petrelfs/linjingli/UMM_Spatial/bagel/ex_img/{name}'
    os.makedirs(outdir,exist_ok=True)
    
    
    def read_jsonl(file_path):
        """逐行读取JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        return data

    val_datas = read_jsonl(json_path)
    
    
    for image_file in os.listdir(tmp_dir):
        id_ = int(image_file.split('.')[0])
        for item in val_datas:
            if item['id']==id_:
                available_images =[os.path.join(image_root,t) for t in item['context']+[item['target']]]+ [os.path.join(tmp_dir,image_file)]
                break
    
        result = merge_with_last_spaced(
            available_images, 
            os.path.join(outdir,image_file),
            spacing=20,      
            last_spacing=80  
        )