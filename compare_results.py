import json
import os
from pathlib import Path

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def compare_json(json1, json2, path=""):
    """递归比较两个JSON对象的差异"""
    differences = []
    
    if type(json1) != type(json2):
        differences.append(f"{path}: 类型不同 - v5: {type(json1).__name__}, v6: {type(json2).__name__}")
        return differences
    
    if isinstance(json1, dict):
        all_keys = set(json1.keys()) | set(json2.keys())
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            if key not in json1:
                differences.append(f"{current_path}: v5中缺少此键")
            elif key not in json2:
                differences.append(f"{current_path}: v6中缺少此键")
            else:
                differences.extend(compare_json(json1[key], json2[key], current_path))
    
    elif isinstance(json1, list):
        if len(json1) != len(json2):
            differences.append(f"{path}: 列表长度不同 - v5: {len(json1)}, v6: {len(json2)}")
        
        min_len = min(len(json1), len(json2))
        for i in range(min_len):
            current_path = f"{path}[{i}]"
            differences.extend(compare_json(json1[i], json2[i], current_path))
    
    else:
        if json1 != json2:
            differences.append(f"{path}: 值不同 - v5: {json1}, v6: {json2}")
    
    return differences

def main():
    v5_dir = Path("e:/temp-wy/outputs/nl_graph/v5")
    v6_dir = Path("e:/temp-wy/outputs/nl_graph/v6")
    
    # 获取所有子文件夹
    v5_subdirs = [d for d in v5_dir.iterdir() if d.is_dir()]
    v6_subdirs = [d for d in v6_dir.iterdir() if d.is_dir()]
    
    v5_names = {d.name for d in v5_subdirs}
    v6_names = {d.name for d in v6_subdirs}
    
    print("=" * 60)
    print("比较 v5 和 v6 目录中的 result.json 文件")
    print("=" * 60)
    
    # 检查目录差异
    only_in_v5 = v5_names - v6_names
    only_in_v6 = v6_names - v5_names
    common_dirs = v5_names & v6_names
    
    if only_in_v5:
        print(f"\n仅在v5中存在的目录: {sorted(only_in_v5)}")
    
    if only_in_v6:
        print(f"\n仅在v6中存在的目录: {sorted(only_in_v6)}")
    
    print(f"\n共同目录数量: {len(common_dirs)}")
    print(f"共同目录: {sorted(common_dirs)}")
    
    # 比较共同目录中的result.json文件
    total_differences = 0
    
    for dir_name in sorted(common_dirs):
        v5_result_path = v5_dir / dir_name / "result.json"
        v6_result_path = v6_dir / dir_name / "result.json"
        
        print(f"\n{'='*40}")
        print(f"比较目录: {dir_name}")
        print(f"{'='*40}")
        
        # 检查文件是否存在
        if not v5_result_path.exists():
            print(f"v5中缺少文件: {v5_result_path}")
            continue
        
        if not v6_result_path.exists():
            print(f"v6中缺少文件: {v6_result_path}")
            continue
        
        # 加载JSON文件
        v5_data = load_json(v5_result_path)
        v6_data = load_json(v6_result_path)
        
        if v5_data is None or v6_data is None:
            print(f"无法加载 {dir_name} 的JSON文件")
            continue
        
        # 比较JSON内容
        differences = compare_json(v5_data, v6_data)
        
        if differences:
            print(f"发现 {len(differences)} 个差异:")
            for diff in differences:
                print(f"  - {diff}")
            total_differences += len(differences)
        else:
            print("✓ 文件内容完全相同")
    
    print(f"\n{'='*60}")
    print(f"总结:")
    print(f"- 共同目录数量: {len(common_dirs)}")
    print(f"- 总差异数量: {total_differences}")
    if total_differences == 0:
        print("✓ 所有对应的result.json文件内容完全相同")
    else:
        print(f"⚠ 发现差异，请查看上述详细信息")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()