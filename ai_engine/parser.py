"""
OCR 结果解析器

核心逻辑：将 PaddleOCR 的原始文字识别结果，智能解析为
结构化的 "参数名 → 参数值" 键值对。

解析策略（三层提取）：
1. 行内提取：单个文本框中包含 "参数名:数值" 格式的直接解析
2. 空间配对：根据文本框的位置关系，将参数名和数值配对
3. 独立数值：无法匹配参数名的数值，作为独立结果返回
"""
import re
from config import PARSER_CONFIG


# ============================================================
# 制造业常见单位
# ============================================================
KNOWN_UNITS = [
    # 温度
    "℃", "°C", "°F", "C", "F",
    # 压力
    "MPa", "kPa", "Pa", "bar", "psi", "mbar", "Mpa", "KPa",
    # 长度
    "mm", "cm", "m", "μm", "um", "inch", "in",
    # 重量/力
    "kg", "g", "t", "N", "kN", "KN", "kgf", "ton",
    # 速度/转速
    "rpm", "r/min", "RPM", "mm/s", "m/s", "m/min",
    # 频率
    "Hz", "kHz", "KHz", "MHz",
    # 功率/电气
    "kW", "KW", "W", "A", "mA", "V", "kV", "KV", "VA", "kVA",
    # 时间
    "s", "sec", "min", "h", "ms",
    # 流量
    "L/min", "ml/s", "m³/h", "L/h", "cc",
    # 百分比
    "%",
    # 其他
    "PCS", "pcs", "shot", "次", "件",
]

# 按长度排序（先匹配长单位，避免 "m" 匹配到 "mm" 的情况）
KNOWN_UNITS.sort(key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(u) for u in KNOWN_UNITS)


# ============================================================
# 核心解析函数
# ============================================================

def parse_ocr_results(ocr_results: list, debug: bool = False) -> list[dict]:
    """
    解析 PaddleOCR 结果，提取参数名-值对。

    Args:
        ocr_results: PaddleOCR 返回的原始结果
        debug: 是否输出调试日志

    Returns:
        参数列表, 每个元素格式:
        {
            "name": "模具温度",      # 参数名（可能为空）
            "value": "180.5",        # 参数值
            "unit": "℃",            # 单位（可能为空）
            "confidence": 0.95,      # 置信度
            "source": "inline|spatial|standalone"  # 提取来源
        }
    """
    if not ocr_results or not ocr_results[0]:
        return []

    # Step 1: 解析所有文本框
    boxes = _parse_raw_boxes(ocr_results[0])
    if debug:
        print(f"[PARSE] 识别到 {len(boxes)} 个文本框")

    # 过滤低置信度
    min_conf = PARSER_CONFIG["min_confidence"]
    boxes = [b for b in boxes if b["confidence"] >= min_conf]
    if debug:
        print(f"[PARSE] 过滤后保留 {len(boxes)} 个框 (置信度 >= {min_conf})")

    if not boxes:
        return []

    # Step 2: 行内提取（单框中含参数名和值）
    inline_pairs, remaining = _extract_inline_pairs(boxes)
    if debug:
        print(f"[PARSE] 行内提取: {len(inline_pairs)} 对, 剩余 {len(remaining)} 框")

    # Step 3: 分类剩余文本框
    labels, values, others = _classify_boxes(remaining)
    if debug:
        print(f"[PARSE] 文本分类: {len(labels)} 个标签, {len(values)} 个值, {len(others)} 个其他")

    # Step 4: 空间配对
    spatial_pairs, unmatched_labels, unmatched_values = _spatial_match(labels, values)
    if debug:
        print(f"[PARSE] 空间配对: {len(spatial_pairs)} 对, 未匹配 {len(unmatched_labels)} 标签, {len(unmatched_values)} 值")

    # Step 5: 组装结果
    all_params = []
    all_params.extend(inline_pairs)
    all_params.extend(spatial_pairs)

    # 未匹配的值作为独立结果
    for v in unmatched_values:
        value_str, unit = _extract_value_and_unit(v["text"])
        if value_str:
            all_params.append({
                "name": "",
                "value": value_str,
                "unit": unit,
                "confidence": v["confidence"],
                "source": "standalone",
            })

    # 按置信度排序
    all_params.sort(key=lambda x: x["confidence"], reverse=True)

    if debug:
        print(f"[PARSE] 最终提取: {len(all_params)} 个参数")
        for i, param in enumerate(all_params, 1):
            print(f"       {i}. {param['name']}={param['value']}{param['unit']} ({param['source']})")

    return all_params


def get_raw_texts(ocr_results: list) -> list[dict]:
    """
    获取所有 OCR 识别到的原始文本（用于调试和展示）。

    Returns:
        [{"text": "...", "confidence": 0.95, "position": [x, y]}, ...]
    """
    if not ocr_results or not ocr_results[0]:
        return []

    texts = []
    for line in ocr_results[0]:
        text, confidence = _extract_text_and_confidence(line)
        box = _extract_box_coords(line)
        cx = sum(p[0] for p in box) / 4
        cy = sum(p[1] for p in box) / 4
        texts.append({
            "text": text,
            "confidence": round(confidence, 4),
            "position": [round(cx, 1), round(cy, 1)],
        })

    return texts


# ============================================================
# 内部函数
# ============================================================

def _extract_text_and_confidence(line) -> tuple[str, float]:
    """
    从 OCR 结果行中提取文本和置信度，兼容多种 PaddleOCR 版本格式。

    支持的格式：
    - 旧版: [box, (text, confidence)]
    - PP-OCRv5: [box, text, confidence] 或 [box, text]
    - PaddleX dict: {"dt_polys": ..., "rec_text": ..., "rec_score": ...}
    """
    # Dict 格式 (PaddleX)
    if isinstance(line, dict):
        text = str(line.get("rec_text", line.get("text", ""))).strip()
        confidence = float(line.get("rec_score", line.get("score", line.get("confidence", 0.0))))
        return text, confidence

    # List 格式
    if len(line) >= 3 and isinstance(line[1], str):
        # PP-OCRv5 新格式: [box, text_str, confidence_float]
        return line[1].strip(), float(line[2])

    if len(line) >= 2:
        if isinstance(line[1], (tuple, list)) and len(line[1]) >= 2:
            # 旧版格式: [box, (text, confidence)]
            return str(line[1][0]).strip(), float(line[1][1])
        elif isinstance(line[1], str):
            # [box, text_str] — 无置信度
            return line[1].strip(), 1.0

    return "", 0.0


def _extract_box_coords(line) -> list:
    """
    从 OCR 结果行中提取坐标框，兼容多种格式。
    """
    if isinstance(line, dict):
        return line.get("dt_polys", line.get("box", line.get("points", [])))
    return line[0]


def _parse_raw_boxes(raw_lines: list) -> list[dict]:
    """将 PaddleOCR 原始结果解析为标准化的文本框列表。"""
    boxes = []
    for line in raw_lines:
        text, confidence = _extract_text_and_confidence(line)
        box_coords = _extract_box_coords(line)

        if not text:
            continue

        # 计算中心点和尺寸
        xs = [p[0] for p in box_coords]
        ys = [p[1] for p in box_coords]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        boxes.append({
            "text": text,
            "confidence": confidence,
            "bbox": box_coords,
            "center": (cx, cy),
            "width": width,
            "height": height,
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
        })

    return boxes


def _extract_inline_pairs(boxes: list) -> tuple[list, list]:
    """
    从单个文本框中提取行内键值对。
    例如: "温度：180℃" → name=温度, value=180, unit=℃

    Returns:
        (extracted_pairs, remaining_boxes)
    """
    pairs = []
    remaining = []

    for box in boxes:
        text = box["text"]
        pair = _try_inline_parse(text, box["confidence"])
        if pair:
            pair["source"] = "inline"
            pairs.append(pair)
        else:
            remaining.append(box)

    return pairs, remaining


def _try_inline_parse(text: str, confidence: float) -> dict | None:
    """尝试从单个文本字符串中解析出参数名和值。"""

    # 模式1: "参数名：数值单位" 或 "参数名: 数值 单位"
    # 支持中文冒号、英文冒号、等号作为分隔符
    pattern1 = (
        r"([^\d:：=\-]+?)\s*[：:=]\s*"  # 参数名 + 分隔符
        r"([+-]?\d+\.?\d*)"             # 数值
        r"\s*(" + _UNIT_PATTERN + r")?"  # 可选单位
    )
    m = re.match(pattern1, text)
    if m:
        name = m.group(1).strip()
        value = m.group(2)
        unit = m.group(3) or ""
        if len(name) >= 1 and _is_valid_param_name(name):
            return {
                "name": name,
                "value": value,
                "unit": unit,
                "confidence": confidence,
            }

    # 模式2: "英文缩写 数值" 如 "PV 180.5" "SV 200"
    pattern2 = (
        r"([A-Za-z][A-Za-z0-9_]{0,15})\s+"  # 英文参数名
        r"([+-]?\d+\.?\d*)"                   # 数值
        r"\s*(" + _UNIT_PATTERN + r")?"        # 可选单位
    )
    m = re.match(pattern2, text)
    if m:
        name = m.group(1).strip()
        value = m.group(2)
        unit = m.group(3) or ""
        # 排除纯单位或无意义的情况
        if not _is_unit_like(name):
            return {
                "name": name,
                "value": value,
                "unit": unit,
                "confidence": confidence,
            }

    # 模式3: "中文参数名 数值 单位" 如 "模温 180 ℃"
    pattern3 = (
        r"([\u4e00-\u9fff][\u4e00-\u9fff\w]{0,15})\s+"  # 中文参数名
        r"([+-]?\d+\.?\d*)"                               # 数值
        r"\s*(" + _UNIT_PATTERN + r")?"                    # 可选单位
    )
    m = re.match(pattern3, text)
    if m:
        name = m.group(1).strip()
        value = m.group(2)
        unit = m.group(3) or ""
        return {
            "name": name,
            "value": value,
            "unit": unit,
            "confidence": confidence,
        }

    return None


def _classify_boxes(boxes: list) -> tuple[list, list, list]:
    """
    将文本框分为三类：标签(labels)、数值(values)、其他(others)。

    分类规则：
    - values: 主要由数字组成的文本
    - labels: 主要由中文/英文字母组成的文本
    - others: 无法明确分类的
    """
    labels = []
    values = []
    others = []

    for box in boxes:
        text = box["text"].strip()
        category = _categorize_text(text)

        if category == "value":
            values.append(box)
        elif category == "label":
            labels.append(box)
        else:
            others.append(box)

    return labels, values, others


def _categorize_text(text: str) -> str:
    """对单个文本进行分类。"""
    if not text:
        return "other"

    # 去掉前后空格和已知单位
    clean = text.strip()
    for unit in KNOWN_UNITS:
        if clean.endswith(unit):
            clean = clean[: -len(unit)].strip()
            break

    # 如果去掉单位后是纯数字（含小数点、正负号）
    if re.match(r"^[+-]?\d+\.?\d*$", clean):
        return "value"

    # 如果去掉单位后是带逗号/空格分隔的数字（如 "1,234"）
    if re.match(r"^[+-]?\d{1,3}(,\d{3})*\.?\d*$", clean):
        return "value"

    # 计算字符类型比例
    digits = sum(c.isdigit() for c in clean)
    chinese = sum("\u4e00" <= c <= "\u9fff" for c in clean)
    letters = sum(c.isalpha() and not ("\u4e00" <= c <= "\u9fff") for c in clean)
    total = len(clean.replace(" ", ""))

    if total == 0:
        return "other"

    digit_ratio = digits / total

    # 数字为主（>60%）→ 数值
    if digit_ratio > 0.6:
        return "value"

    # 有中文或英文字母 → 标签
    if chinese > 0 or letters > 1:
        return "label"

    # 单个字母可能是序号或标签
    if letters == 1 and digits == 0:
        return "label"

    return "other"


def _spatial_match(
    labels: list, values: list
) -> tuple[list, list, list]:
    """
    根据空间位置关系，将标签和数值配对。增强版本，支持复杂屏显。

    改进点：
    1. 按行/列分组，处理多行多列的参数布局
    2. 优先同行配对，其次相邻行配对
    3. 同列配对时优先相邻值
    4. 支持参数名在值右边的情况（某些屏显布局）

    Returns:
        (pairs, unmatched_labels, unmatched_values)
    """
    if not labels or not values:
        return [], labels, values

    max_dist = PARSER_CONFIG["max_pair_distance"]
    row_tol = PARSER_CONFIG["row_tolerance_ratio"]

    # 计算所有可能的 (label, value) 配对及其得分
    candidates = []
    for i, label in enumerate(labels):
        for j, value in enumerate(values):
            score = _pair_score(label, value, row_tol)
            if score < max_dist:
                candidates.append((score, i, j))

    # 按得分从低到高排序（贪心：优先匹配最近的）
    candidates.sort(key=lambda x: x[0])

    # 贪心配对
    used_labels = set()
    used_values = set()
    pairs = []

    for score, li, vi in candidates:
        if li in used_labels or vi in used_values:
            continue

        used_labels.add(li)
        used_values.add(vi)

        label = labels[li]
        value = values[vi]

        value_str, unit = _extract_value_and_unit(value["text"])
        if value_str:
            pairs.append({
                "name": label["text"].strip(),
                "value": value_str,
                "unit": unit,
                "confidence": min(label["confidence"], value["confidence"]),
                "source": "spatial",
            })

    # 收集未匹配的
    unmatched_labels = [l for i, l in enumerate(labels) if i not in used_labels]
    unmatched_values = [v for i, v in enumerate(values) if i not in used_values]

    return pairs, unmatched_labels, unmatched_values


def _pair_score(label: dict, value: dict, row_tolerance_ratio: float) -> float:
    """
    计算标签和数值的配对得分（越小越好）。增强版本。

    得分规则（优先级从高到低）：
    1. 值在标签右侧同行 → 最优（低分）
    2. 值在标签右下方 → 次优（中分）
    3. 值在标签下方同列 → 次优（中分）
    4. 其他情况 → 高分或无穷（不配对）
    
    改进点：
    - 更细致的距离计算
    - 考虑文本大小的相对性
    - 对屏显布局的优化
    """
    lx, ly = label["center"]
    vx, vy = value["center"]
    lh = label["height"]
    vh = value["height"]
    lw = label["width"]
    vw = value["width"]

    dx = vx - lx  # 正值 = 值在右边
    dy = vy - ly  # 正值 = 值在下面

    max_height = max(lh, vh)
    row_tol = max_height * row_tolerance_ratio

    # 规则1：值在标签左边太远 → 不配对
    if dx < -lw * 0.5:
        return float("inf")

    # 规则2：值在标签上方太远 → 不配对
    if dy < -max_height * 1.5:
        return float("inf")

    # 规则3：同行判定（Y 距离小）
    if abs(dy) < row_tol:
        # 同一行，优先右侧
        if dx > 0:
            # 右侧同行，最优
            # 距离越短分数越低
            return dx * 0.5 + abs(dy) * 2.0
        else:
            # 左侧同行，降权但仍可能配对
            return abs(dx) * 2.0 + abs(dy) * 2.0

    # 规则4：不同行，值在下方
    if dy > 0:
        # 优先右下方（屏显常见布局）
        if dx >= -lw * 0.3:
            return abs(dx) * 0.8 + dy * 1.5
        else:
            # 左下方
            return abs(dx) * 2.0 + dy * 1.5
    
    # 规则5：值在上方（不太合理），高惩罚
    return abs(dx) * 3.0 + abs(dy) * 3.0


def _extract_value_and_unit(text: str) -> tuple[str, str]:
    """
    从文本中分离数值和单位。

    Examples:
        "180.5℃" → ("180.5", "℃")
        "180.5"  → ("180.5", "")
        "1,234"  → ("1234", "")
    """
    text = text.strip()
    unit = ""

    # 尝试匹配尾部单位
    for u in KNOWN_UNITS:
        if text.endswith(u):
            text = text[: -len(u)].strip()
            unit = u
            break

    # 清理数值
    clean = text.replace(",", "").replace(" ", "")

    # 验证是否为有效数值
    if re.match(r"^[+-]?\d+\.?\d*$", clean):
        return clean, unit

    # 尝试从文本中提取数值
    m = re.search(r"([+-]?\d+\.?\d*)", text)
    if m:
        return m.group(1), unit

    return "", ""


def _is_valid_param_name(text: str) -> bool:
    """检查文本是否像一个有效的参数名。"""
    text = text.strip()
    if not text:
        return False
    # 不应该是纯数字
    if re.match(r"^[\d.]+$", text):
        return False
    # 不应该是纯符号
    if re.match(r"^[^\w\u4e00-\u9fff]+$", text):
        return False
    # 至少包含一个中文字符或英文字母
    has_meaningful = any(
        c.isalpha() or "\u4e00" <= c <= "\u9fff" for c in text
    )
    return has_meaningful


def _is_unit_like(text: str) -> bool:
    """检查文本是否像一个单位名称。"""
    return text.strip() in KNOWN_UNITS or text.strip().lower() in [
        u.lower() for u in KNOWN_UNITS
    ]
