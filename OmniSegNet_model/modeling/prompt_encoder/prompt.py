from .misc import get_stroke_preset, get_random_points_from_mask, get_mask_by_input_strokes

import random
import torch



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

