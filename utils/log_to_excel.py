"""
通用日志/文本转Excel工具
支持多种格式的表格数据转换
"""

import re
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill


def parse_text_to_dataframe(text,
                            timestamp_pattern=r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
                            skip_patterns=None,
                            delimiter=None,
                            auto_detect=True):
    """
    将文本数据解析为DataFrame

    参数:
        text: 输入文本
        timestamp_pattern: 时间戳的正则表达式模式
        skip_patterns: 要跳过的行的模式列表
        delimiter: 列分隔符(None表示自动检测空白符)
        auto_detect: 是否自动检测表头
    """
    if skip_patterns is None:
        skip_patterns = []

    lines = text.strip().split('\n')
    headers = []
    data_rows = []

    for line in lines:
        # 移除时间戳
        if timestamp_pattern:
            line = re.sub(timestamp_pattern, '', line).strip()

        # 跳过空行
        if not line:
            continue

        # 跳过匹配指定模式的行
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                should_skip = True
                break
        if should_skip:
            continue

        # 分割行
        if delimiter:
            parts = [p.strip() for p in line.split(delimiter)]
        else:
            # 自动检测：使用空白符分割
            parts = line.split()

        # 如果还没有表头，且这行看起来像表头
        if not headers and auto_detect:
            # 检查是否包含非数字列名
            if any(not is_number(p) for p in parts):
                headers = parts
                continue

        # 尝试将数据转换为合适的类型
        processed_parts = []
        for p in parts:
            if is_number(p):
                try:
                    if '.' in p:
                        processed_parts.append(float(p))
                    else:
                        processed_parts.append(int(p))
                except:
                    processed_parts.append(p)
            else:
                processed_parts.append(p)

        data_rows.append(processed_parts)

    # 如果没有检测到表头，创建默认表头
    if not headers and data_rows:
        max_cols = max(len(row) for row in data_rows)
        headers = [f'Column_{i + 1}' for i in range(max_cols)]

    # 确保所有行的列数一致
    max_cols = len(headers)
    for i, row in enumerate(data_rows):
        if len(row) < max_cols:
            data_rows[i] = row + [None] * (max_cols - len(row))
        elif len(row) > max_cols:
            data_rows[i] = row[:max_cols]

    # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    return df


def is_number(s):
    """检查字符串是否为数字"""
    try:
        float(s)
        return True
    except:
        return False


def create_excel(df, output_path,
                 title=None,
                 header_bg_color='4472C4',
                 header_font_color='FFFFFF',
                 freeze_panes=True):
    """
    从DataFrame创建格式化的Excel文件

    参数:
        df: pandas DataFrame
        output_path: 输出文件路径
        title: 工作表标题
        header_bg_color: 表头背景色(十六进制)
        header_font_color: 表头字体色(十六进制)
        freeze_panes: 是否冻结首行
    """
    # 保存为Excel
    df.to_excel(output_path, index=False, sheet_name=title or 'Sheet1')

    # 加载并格式化
    wb = load_workbook(output_path)
    sheet = wb.active

    # 设置表头格式
    header_fill = PatternFill(start_color=header_bg_color,
                              end_color=header_bg_color,
                              fill_type='solid')
    header_font = Font(bold=True, color=header_font_color, name='Arial', size=11)

    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')

    # 设置数据格式
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        for cell in row:
            cell.font = Font(name='Arial', size=10)
            cell.alignment = Alignment(vertical='center')

    # 自动调整列宽
    for column in sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter

        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass

        adjusted_width = min(max_length + 2, 50)
        sheet.column_dimensions[column_letter].width = adjusted_width

    # 冻结首行
    if freeze_panes:
        sheet.freeze_panes = 'A2'

    wb.save(output_path)
    return output_path


def text_to_excel(text, output_path, **kwargs):
    """
    一步完成：文本转Excel

    参数:
        text: 输入文本
        output_path: 输出Excel路径
        **kwargs: 其他参数传递给parse_text_to_dataframe和create_excel
    """
    # 分离参数
    parse_params = ['timestamp_pattern', 'skip_patterns', 'delimiter', 'auto_detect']
    excel_params = ['title', 'header_bg_color', 'header_font_color', 'freeze_panes']

    parse_kwargs = {k: v for k, v in kwargs.items() if k in parse_params}
    excel_kwargs = {k: v for k, v in kwargs.items() if k in excel_params}

    # 解析并创建
    df = parse_text_to_dataframe(text, **parse_kwargs)
    create_excel(df, output_path, **excel_kwargs)

    return df


# 使用示例
if __name__ == '__main__':
    # 示例1：处理原始数据
    sample_text = """2026-01-31 14:03:32
Layers sorted by improvement (higher = more beneficial):
2026-01-31 14:03:32
         layer_name  ssim_score  final_improvement
2026-01-31 14:03:32
     layer3.0.conv2   -0.457737               4.24
2026-01-31 14:03:32
     layer3.2.conv1   -0.538324               4.22
2026-01-31 14:03:32
     layer3.1.conv1   -0.570038               4.03
2026-01-31 14:03:32
     layer3.1.conv2   -0.569057               3.84
2026-01-31 14:03:32
     layer3.0.conv1   -0.099905               3.12
2026-01-31 14:03:32
     layer2.2.conv2    0.068087               2.67
2026-01-31 14:03:32
     layer2.1.conv2    0.474653               2.67
2026-01-31 14:03:32
     layer2.1.conv1   -0.080113               2.18
2026-01-31 14:03:32
     layer2.0.conv2    0.315279               2.16
2026-01-31 14:03:32
     layer2.2.conv1   -0.071820               2.05
2026-01-31 14:03:32
     layer3.2.conv2    0.906409               1.94
2026-01-31 14:03:32
     layer2.0.conv1    0.724397               1.42
2026-01-31 14:03:32
     layer1.2.conv1    0.258307               0.86
2026-01-31 14:03:32
     layer1.1.conv1    0.551186               0.70
2026-01-31 14:03:32
     layer1.2.conv2    0.147868               0.66
2026-01-31 14:03:32
layer3.0.shortcut.0    0.380724               0.45
2026-01-31 14:03:32
     layer1.0.conv2    0.484730               0.34
2026-01-31 14:03:32
     layer1.1.conv2    0.572650               0.18
2026-01-31 14:03:32
             linear    0.500000               0.15
2026-01-31 14:03:32
              conv1    0.446107               0.14
2026-01-31 14:03:32
layer2.0.shortcut.0    0.124001               0.11
2026-01-31 14:03:32
     layer1.0.conv1    0.576959               0.03"""

    df = text_to_excel(
        sample_text,
        'example_output.xlsx',
        skip_patterns=[r'sorted by', r'beneficial'],
        title='Layer Analysis'
    )

    print(f"✓ 已创建 Excel 文件")
    print(f"✓ 共 {len(df)} 行, {len(df.columns)} 列")
    print(f"✓ 列名: {', '.join(df.columns)}")