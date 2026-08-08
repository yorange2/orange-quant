"""兼容 shim：规范位置已移至 orange_quant/data/merge_cn_update.py。

保留本文件使旧路径 `python scripts/merge_cn_update.py ...` 继续可用
（工作区 CLAUDE.md 等外部引用）；新代码请用
`python -m orange_quant.data.merge_cn_update ...`。
"""
from orange_quant.data.merge_cn_update import main

if __name__ == "__main__":
    main()
