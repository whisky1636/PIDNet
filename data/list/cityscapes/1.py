import glob
import os

def batch_convert_lst(directory='.'):
    # 获取目录下所有 .lst 文件
    # recursive=False 表示只找当前层级，如果不止一层可以设为 True 并改用 '**/*.lst'
    files = glob.glob(os.path.join(directory, '*.lst'))
    
    if not files:
        print("未发现任何 .lst 文件。")
        return

    print(f"找到 {len(files)} 个文件，开始转换...")

    for file_path in files:
        try:
            # 1. 读取内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 2. 替换斜杠
            new_content = content.replace('/', '\\')
            
            # 3. 写回原文件（如果想保留备份，可以修改文件名）
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print(f"成功: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"跳过 {file_path}，原因: {e}")

    print("\n所有操作已完成。")

if __name__ == "__main__":
    # 你可以把 '.' 换成具体的文件夹路径，例如 r'C:\MyFiles'
    batch_convert_lst('.')