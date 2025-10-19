# download_with_api.py
import os
from huggingface_hub import snapshot_download, login
import requests

def setup_mirror():
    """设置镜像源"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"已设置镜像源: {os.environ['HF_ENDPOINT']}")

def download_model(repo_id, local_dir, token=None):
    """
    使用 Python API 下载模型
    """
    setup_mirror()
    
    # 登录（如果需要）
    if token and token != "YOUR_HF_TOKEN":
        try:
            login(token=token)
            print("登录成功")
        except Exception as e:
            print(f"登录失败: {e}")
    
    try:
        print(f"开始下载: {repo_id}")
        print(f"保存到: {local_dir}")
        
        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token
        )
        
        print("下载完成!")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

if __name__ == "__main__":
    repo_id = "nvidia/audio-flamingo-2"
    local_dir = "/media/tongji/26_npj_audio/model/Flamingo2"
    token = "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"  # 替换为你的实际token
    
    # 创建目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 下载模型
    success = download_model(repo_id, local_dir, token)