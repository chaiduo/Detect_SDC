import json
import sys
import os
import re

def calc_avg_word_count(json_path, strict_space=False):
    """统计 pre_answer 按空格分词后的平均词数"""
    
    # 1. 读取文件
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 统一转为列表（兼容字典包裹的情况）
    items = data if isinstance(data, list) else (
        next((v for v in data.values() if isinstance(v, list)), [data])
    )
    
    # 3. 提取并统计
    lengths = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("pre_answer", ""))
        # 分词：strict_space=True 时严格按单空格切分
        tokens = text.split(' ') if strict_space else text.split()
        lengths.append(len(tokens))
    
    if not lengths:
        print("⚠️ 未找到有效数据")
        return
    
    # 4. 计算统计指标
    avg = sum(lengths) / len(lengths)
    median = sorted(lengths)[len(lengths)//2]
    min_val, max_val = min(lengths), max(lengths)
    
    # 5. 输出结果
    print(f"📊 统计结果（共 {len(lengths)} 条）")
    print(f"   平均词数 : {avg:.2f}")
    print(f"   中位数   : {median}")
    print(f"   最小/最大: {min_val} / {max_val}")
    
    # 6. 可选：输出前10个样本供抽查
    print("\n🔍 前10条样本预览（索引: 词数）:")
    for i, l in enumerate(lengths[:10]):
        print(f"   [{i}] {l}")

if __name__ == "__main__":
    # 用法: python avg_pre_answer.py data.json [--strict]
    path = sys.argv[1] if len(sys.argv) > 1 else input("请输入 JSON 文件路径: ").strip()
    strict = "--strict" in sys.argv
    calc_avg_word_count(path, strict_space=strict)