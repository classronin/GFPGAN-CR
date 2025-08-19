import argparse
import cv2
import glob
import os
import sys
import numpy as np
import torch
from basicsr.utils import imwrite
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer

# 支持的图像文件扩展名
SUPPORTED_IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']

def is_image_file(file_path):
    """检查文件是否为支持的图像格式"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTS

def safe_save_image(image, save_path):
    """尝试多种方法保存图像"""
    # 方法1: 尝试使用OpenCV保存
    if cv2.imwrite(save_path, image):
        return True
    
    # 方法2: 尝试使用basicsr的imwrite保存
    try:
        imwrite(image, save_path)
        return True
    except:
        pass
    
    # 方法3: 尝试使用PIL保存
    try:
        from PIL import Image
        # 转换OpenCV BGR格式为RGB
        if image.shape[2] == 3:  # RGB图像
            image_rgb = image[:, :, ::-1]
            Image.fromarray(image_rgb).save(save_path)
            return True
        elif image.shape[2] == 4:  # RGBA图像
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            Image.fromarray(image_rgba).save(save_path)
            return True
    except ImportError:
        print("警告: PIL库未安装，无法尝试PIL保存方法")
    except Exception as e:
        print(f"PIL保存失败: {str(e)}")
    
    # 方法4: 尝试使用scikit-image保存
    try:
        import skimage.io
        skimage.io.imsave(save_path, image)
        return True
    except ImportError:
        print("警告: scikit-image库未安装，无法尝试此保存方法")
    except Exception as e:
        print(f"scikit-image保存失败: {str(e)}")
    
    return False

def main():
    # 设置默认输入输出为当前目录
    current_dir = os.getcwd()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=current_dir, 
                        help='输入图像或文件夹路径。默认使用当前工作目录')
    parser.add_argument('-o', '--output', type=str, default=current_dir, 
                        help='输出文件夹路径。默认使用当前工作目录')
    parser.add_argument('-s', '--upscale', type=int, default=2, 
                        help='图像的最终上采样比例。默认值: 2')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', 
                        help='背景上采样器选择。默认值: realesrgan')
    parser.add_argument('--bg_tile', type=int, default=400, 
                        help='背景采样器的瓦片大小，0表示不使用瓦片处理。默认值: 400')
    parser.add_argument('--suffix', type=str, default=None, 
                        help='修复后图像的文件名后缀')
    parser.add_argument('--only_center_face', action='store_true', 
                        help='仅修复中心人脸')
    parser.add_argument('--aligned', action='store_true', 
                        help='输入图像是已对齐的人脸')
    parser.add_argument('--ext', type=str, default='auto', 
                        help='输出图像格式。选项: auto | jpg | png, auto表示使用与输入相同的格式。默认值: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, 
                        help='可调节的权重参数。默认值: 0.5')
    args = parser.parse_args()

    # 确保输入输出路径是绝对路径
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")

    # 处理输入路径
    if args.input.endswith('/') or args.input.endswith('\\'):
        args.input = args.input[:-1]
    
    # 收集图像文件列表
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = [f for f in glob.glob(os.path.join(args.input, '*')) 
                   if os.path.isfile(f) and is_image_file(f)]
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 模型URL
    REALSGAN_MODEL_URL = "https://ghfast.top/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    GFPGAN_MODEL_URL = "https://ghfast.top/github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    PARSING_MODEL_URL = "https://ghfast.top/github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
    DETECTION_MODEL_URL = "https://ghfast.top/github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    
    # 设置背景上采样器
    bg_upsampler = None
    if args.bg_upsampler == 'realesrgan' and torch.cuda.is_available():
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
            
        # 下载或加载RealESRGAN模型
        realesrgan_path = load_file_from_url(
            url=REALSGAN_MODEL_URL,
            model_dir="models",
            progress=True,
            file_name="RealESRGAN_x2plus.pth"
        )
            
        # 创建背景上采样器
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path=realesrgan_path,
            model=model,
            tile=args.bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True)

    # 创建模型保存目录
    model_dir = os.path.join(current_dir, "models")  # 修改为根目录下的models文件夹
    os.makedirs(model_dir, exist_ok=True)
    
    # 下载GFPGAN模型
    gfpgan_model_path = os.path.join(model_dir, "GFPGANv1.4.pth")
    if not os.path.isfile(gfpgan_model_path):
        gfpgan_model_path = load_file_from_url(
            url=GFPGAN_MODEL_URL,
            model_dir=model_dir,
            progress=True,
            file_name="GFPGANv1.4.pth"
        )
    
    # 下载新模型
    parsing_model_path = os.path.join(model_dir, "parsing_parsenet.pth")
    if not os.path.isfile(parsing_model_path):
        parsing_model_path = load_file_from_url(
            url=PARSING_MODEL_URL,
            model_dir=model_dir,
            progress=True,
            file_name="parsing_parsenet.pth"
        )
    
    detection_model_path = os.path.join(model_dir, "detection_Resnet50_Final.pth")
    if not os.path.isfile(detection_model_path):
        detection_model_path = load_file_from_url(
            url=DETECTION_MODEL_URL,
            model_dir=model_dir,
            progress=True,
            file_name="detection_Resnet50_Final.pth"
        )

    # 创建修复器
    restorer = GFPGANer(
        model_path=gfpgan_model_path,
        upscale=args.upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler)

    # 图像修复处理
    processed_count = 0
    restored_dir = os.path.join(args.output, "修复图像")
    os.makedirs(restored_dir, exist_ok=True)
    print(f"创建输出目录: {restored_dir}")
    
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        
        # 跳过非图像文件
        if not is_image_file(img_path):
            print(f'跳过非图像文件: {img_name}')
            continue
            
        print(f'处理中: {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        
        # 读取图像并检查有效性
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if input_img is None:
            print(f'无法读取图像文件: {img_name}')
            continue

        try:
            # 修复人脸和背景
            _, _, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight)
        except Exception as e:
            print(f'处理图像失败: {img_name}, 错误: {str(e)}')
            continue

        # 保存修复后的图像到"修复图像"文件夹
        if restored_img is not None:
            extension = ext[1:] if args.ext == 'auto' else args.ext
            suffix = f'_{args.suffix}' if args.suffix else ''
            save_path = os.path.join(restored_dir, f'{basename}{suffix}.{extension}')
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 尝试多种方法保存图像
            if safe_save_image(restored_img, save_path):
                print(f'成功保存: {save_path}')
                processed_count += 1
            else:
                # 尝试保存为PNG格式
                print(f'标准保存失败，尝试保存为PNG格式...')
                save_path_png = os.path.join(restored_dir, f'{basename}{suffix}.png')
                if safe_save_image(restored_img, save_path_png):
                    print(f'成功保存为PNG格式: {save_path_png}')
                    processed_count += 1
                else:
                    print(f'保存图像失败: {save_path} 和 {save_path_png}')

    print(f'处理完成! 成功修复 {processed_count} 张图像')
    print(f'修复图像已保存至: [{restored_dir}] 文件夹')

if __name__ == '__main__':
    # 确保当前目录是脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # 设置系统编码支持中文
    if sys.platform.startswith('win'):
        try:
            import _locale
            _locale._getdefaultlocale = (lambda *args: ('zh_CN', 'utf-8'))
        except:
            pass
    
    # 添加额外的错误处理
    try:
        main()
    except Exception as e:
        print(f"程序发生未捕获的异常: {str(e)}")
        import traceback
        traceback.print_exc()