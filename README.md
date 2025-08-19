
此 [GFPGAN-CR](https://github.com/classronin/GFPGAN-CR) 来自 [GFPGAN](https://github.com/TencentARC/GFPGAN)


些库简易安装，仅限推理。
```
git clone https://github.com/classronin/GFPGAN-CR
cd GFPGAN-CR
uv venv
uv pip install -r requirements.txt
uv run cr.py -i <输入：图像文件/目录路径> -o <输出：目录路径>
模型会自动下载并创建目录下models文件夹。
```

心急如火可下载模型
```
https://ghfast.top/github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
https://ghfast.top/github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
https://ghfast.top/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
https://ghfast.top/github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```
这四个文件都放到根下models文件夹。

创建 CR.bat
```
@echo off
setlocal
set input_dir=%cd%
set output_dir=%cd%
cd /d "你的路径\GFPGAN-CR"
uv run cr.py -i "%input_dir%" -o "%output_dir%"
endlocal
```
在指定任何文件资源器上输入“CMD”打开命令行的窗口，运行“CR”即可。
此CR.bat的效果是输入和输出都指向命令窗口当前目录的路径。



