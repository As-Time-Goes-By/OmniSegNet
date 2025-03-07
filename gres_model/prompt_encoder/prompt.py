from .misc import get_stroke_preset, get_random_points_from_mask, get_mask_by_input_strokes

import random
import torch


def get_bounding_boxes_v1(mask):
    """
    Returns:
        boxes: tight bounding boxes around bitmasks.
               If a mask is empty, its bounding box will be all zero.
        box_mask: binary mask where the bounding box area is set to 1.
    """
    # 获取设备和初始化结果
    device = mask.device
    boxes = torch.zeros(mask.shape[0], 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros_like(mask, dtype=torch.float32, device=device)

    for idx in range(mask.shape[0]):
        # 通过 nonzero 找到 mask 中非零的坐标
        non_zero_coords = torch.nonzero(mask[idx])

        if non_zero_coords.size(0) > 0:  # 如果非零元素存在
            # 提取 x 和 y 的最小最大值作为边界框
            ymin, xmin = non_zero_coords.min(dim=0)[0]
            ymax, xmax = non_zero_coords.max(dim=0)[0]

            # 更新边界框
            boxes[idx, :] = torch.tensor([xmin, ymin, xmax + 1, ymax + 1], device=device, dtype=torch.float32)

            # 设置 box_mask 对应区域为 1
            box_mask[idx, ymin:ymax + 1, xmin:xmax + 1] = 1.0

    return boxes, box_mask

def get_bounding_boxes(mask):
    """
    Returns:
        Boxes: tight bounding boxes around bitmasks.
        If a mask is empty, it's bounding box will be all zero.
    """
    boxes = torch.zeros(mask.shape[0], 4, dtype=torch.float32).to(mask.device)
    box_mask = torch.zeros_like(mask).to(mask.device)
    x_any = torch.any(mask, dim=1)
    y_any = torch.any(mask, dim=2)
    for idx in range(mask.shape[0]):
        x = torch.where(x_any[idx, :])[0].int()
        y = torch.where(y_any[idx, :])[0].int()
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = torch.as_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
            )
            x1, y1, x2, y2 = x[0], y[0], x[-1] + 1, y[-1] + 1

            box_mask[idx, y1:y2, x1:x2] = 1
    return boxes, box_mask


def get_bounding_boxes_kaung(mask, thickness=4):
    """
    获取边界框，并生成仅包含边框轮廓的 mask（内部为 0）。

    Args:
        mask (torch.Tensor): 输入的二值 mask，形状为 (B, H, W)。
        thickness (int): 线框的厚度。

    Returns:
        boxes (torch.Tensor): 每个 mask 的 bounding box，形状为 (B, 4)。
        box_mask (torch.Tensor): 仅包含边框的 mask，形状为 (B, H, W)。
    """
    B, H, W = mask.shape
    boxes = torch.zeros(B, 4, dtype=torch.float32, device=mask.device)
    box_mask = torch.zeros_like(mask, device=mask.device)

    x_any = torch.any(mask, dim=1)  # 在高度方向找到非零区域
    y_any = torch.any(mask, dim=2)  # 在宽度方向找到非零区域

    for idx in range(B):
        x = torch.where(x_any[idx, :])[0].int()
        y = torch.where(y_any[idx, :])[0].int()

        if len(x) > 0 and len(y) > 0:
            x1, x2 = x[0], x[-1] + 1
            y1, y2 = y[0], y[-1] + 1
            boxes[idx, :] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=mask.device)

            # 仅绘制框的边缘
            box_mask[idx, y1:y1+thickness, x1:x2] = 1  # 顶边
            box_mask[idx, y2-thickness:y2, x1:x2] = 1  # 底边
            box_mask[idx, y1:y2, x1:x1+thickness] = 1  # 左边
            box_mask[idx, y1:y2, x2-thickness:x2] = 1  # 右边

    return boxes, box_mask

def get_point_mask(mask, training, max_points=20):
    """
    Returns:
        Point_mask: random 20 point for train and test.
        If a mask is empty, it's Point_mask will be all zero.
    """
    max_points = min(max_points, mask.sum().item())
    if training:
        num_points = random.Random().randint(1, max_points)  # get a random number of points
    else:
        num_points = max_points
    b, h, w = mask.shape
    point_masks = []

    for idx in range(b):
        view_mask = mask[idx].view(-1)
        non_zero_idx = view_mask.nonzero()[:, 0]  # get non-zero index of mask
        selected_idx = torch.randperm(len(non_zero_idx))[:num_points]  # select id
        non_zero_idx = non_zero_idx[selected_idx]  # select non-zero index
        rand_mask = torch.zeros(view_mask.shape).to(mask.device)  # init rand mask
        rand_mask[non_zero_idx] = 1  # get one place to zero
        point_masks.append(rand_mask.reshape(h, w).unsqueeze(0))
    return torch.cat(point_masks, 0)


def get_scribble_mask(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
    """
    Returns:
        Scribble_mask: random 20 point for train and test.
        If a mask is empty, it's Scribble_mask will be all zero.
    """
    if training:
        stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random().randint(1, min(20, mask.sum().item()))
    else:
        stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
        # nStroke =1
    preset = get_stroke_preset(stroke_preset_name)

    b, h, w = mask.shape

    scribble_masks = []
    for idx in range(b):
        points = get_random_points_from_mask(mask[idx].bool(), n=nStroke)
        rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h,
                                              nStroke=min(nStroke, len(points)), **preset)
        rand_mask = (~torch.from_numpy(rand_mask)) * mask[idx].bool().cpu()
        scribble_masks.append(rand_mask.float().unsqueeze(0))
    return torch.cat(scribble_masks, 0).to(mask.device)

def get_scribble_mask_single(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
    """
    Returns:
        Scribble_mask: random 20 point for train and test.
        If a mask is empty, it's Scribble_mask will be all zero.
    """
    if training:
        stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random().randint(1, min(20, mask.sum().item()))
    else:
        stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
        # nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
        nStroke =1
    preset = get_stroke_preset(stroke_preset_name)
    preset['brushWidthBound']=(10, 15)

    b, h, w = mask.shape

    scribble_masks = []
    for idx in range(b):
        points = get_random_points_from_mask(mask[idx].bool(), n=nStroke)
        rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h,
                                              nStroke=min(nStroke, len(points)), **preset)
        rand_mask = (~torch.from_numpy(rand_mask)) * mask[idx].bool().cpu()
        scribble_masks.append(rand_mask.float().unsqueeze(0))
    return torch.cat(scribble_masks, 0).to(mask.device)