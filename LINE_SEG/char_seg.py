import cv2
import os
import numpy as np

def get_initial_blocks(line_binary):
    """
    First pass: Identify raw character blocks using vertical projection.
    Returns a list of block dictionaries without splitting.
    """
    h_line, w_line = line_binary.shape
    v_proj = np.sum(line_binary == 255, axis=0)
    text_regions = v_proj > 0
    
    blocks = []
    in_char = False
    start_x = 0
    
    for x in range(len(text_regions)):
        if text_regions[x] and not in_char:
            start_x = x
            in_char = True
        elif not text_regions[x] and in_char:
            end_x = x
            in_char = False
            char_band = line_binary[:, start_x:end_x]
            pts = np.where(char_band == 255)
            if len(pts[0]) > 0:
                y_min, y_max = np.min(pts[0]), np.max(pts[0])
                blocks.append({'x': start_x, 'y': y_min, 'w': end_x - start_x, 'h': y_max - y_min + 1})
                
    if in_char:
        char_band = line_binary[:, start_x:]
        pts = np.where(char_band == 255)
        if len(pts[0]) > 0:
            y_min, y_max = np.min(pts[0]), np.max(pts[0])
            blocks.append({'x': start_x, 'y': y_min, 'w': w_line - start_x, 'h': y_max - y_min + 1})
            
    return blocks

def merge_overlapping_boxes(chars, line_binary):
    """
    Merges bounding boxes that overlap in 2D space.
    """
    if not chars:
        return []

    # Sort by x-coordinate to help grouping
    chars.sort(key=lambda c: c['x'])
    
    merged_happened = True
    while merged_happened:
        merged_happened = False
        new_chars = []
        skip_indices = set()
        
        for i in range(len(chars)):
            if i in skip_indices:
                continue
            
            current = chars[i]
            for j in range(i + 1, len(chars)):
                if j in skip_indices:
                    continue
                
                next_char = chars[j]
                
                # Check for intersection
                x1, y1, w1, h1 = current['x'], current['y'], current['w'], current['h']
                x2, y2, w2, h2 = next_char['x'], next_char['y'], next_char['w'], next_char['h']
                
                # Intersection condition
                intersect = not (x2 >= x1 + w1 or x1 >= x2 + w2 or y2 >= y1 + h1 or y1 >= y2 + h2)
                
                if intersect:
                    # Merge
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    
                    # Re-crop from original binary
                    new_crop = line_binary[new_y:new_y+new_h, new_x:new_x+new_w]
                    
                    current = {
                        'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h,
                        'crop': new_crop, 'is_merged': False # Re-evaluated later
                    }
                    skip_indices.add(j)
                    merged_happened = True
            
            new_chars.append(current)
        chars = new_chars
        
    return chars

def adaptive_split_and_save(line_binary, line_blocks, avg_w, avg_h, out_dir, line_source=None):
    """
    Finalize segmentation: Process initial blocks using contours to separate overlapping characters,
    detect merged blobs, combine overlapping boxes, and save.
    
    Args:
        line_binary: The cleaned binary image used for segmentation logic.
        line_blocks: Initial blocks detected.
        avg_w, avg_h: Average dimensions for merge detection.
        out_dir: Directory to save characters.
        line_source: Optional. If provided, characters are cropped from this image (e.g., raw thresh) 
                     instead of line_binary. Defaults to line_binary.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Use line_binary as source if line_source is not provided
    if line_source is None:
        line_source = line_binary

    final_chars = []
    
    # Process each initial block
    for i, b in enumerate(line_blocks):
        crop = line_binary[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']]
        
        # Use Contours to separate disjoint characters within the block
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter noise and create sub-blocks
        sub_blocks = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10: # Minimum area filter
                x, y, w, h = cv2.boundingRect(cnt)
                # bounding rect is relative to crop
                sub_blocks.append({'x': x, 'y': y, 'w': w, 'h': h, 'cnt': cnt})
        
    # Sort sub-blocks by x-coordinate (left to right)
        sub_blocks.sort(key=lambda sb: sb['x'])
        
        for sb in sub_blocks:
            # Global coordinates
            global_x = b['x'] + sb['x']
            global_y = b['y'] + sb['y']
            
            # Extract content from crop (masks out other chars in same block)
            # Create a mask for this specific contour to ensure we only get this character
            mask = np.zeros_like(crop)
            cv2.drawContours(mask, [sb['cnt']], -1, 255, -1)
            
            # Use the mask logic on the CLEANED image to define the shape
            # But we want to extract pixels from the SOURCE image
            
            # We need to map the mask to the source image context if we want to mask it precisely.
            # However, for simple bounding box visualization on source, we might just want the crop.
            # But the user asked for "bounding box on thresh image", usually implies the content.
            
            # If line_source is provided, we cut from it.
            # Note: The mask is derived from line_binary, so it matches line_binary shapes.
            # If line_source is 'thresh', it might have more noise/pixels. Masking it with 
            # the clean mask might lose the "raw" look the user wants, or it might be exactly what is needed 
            # (clean shape, raw pixels).
            # Usually "box values get from cleaned" means we stick to the box.
            
            # Let's extract the crop from line_source using the box coordinates.
            char_crop = line_source[global_y:global_y+sb['h'], global_x:global_x+sb['w']]
            
            # Optional: Apply the mask from cleaned image to the source crop? 
            # If we don't, we might get neighboring noise.
            # If we do, we might clip raw pixels.
            # Implementation decision: Just crop the box from source. 
            # The contour separation logic already separated them spatially.
            
            final_chars.append({
                'x': global_x, 'y': global_y, 'w': sb['w'], 'h': sb['h'], 
                'crop': char_crop, 'is_merged': False
            })

    # NEW: Merge overlapping boxes
    final_chars = merge_overlapping_boxes(final_chars, line_source)

    # Re-evaluate merge status based on new dimensions
    # Threshold for determining if a character is "merged" (too wide or tall)
    # Changed from 1.5 to 1.2 as per user request to be more sensitive
    MERGE_THRESHOLD = 1.2
    
    for c in final_chars:
        if c['w'] > MERGE_THRESHOLD * avg_w or c['h'] > MERGE_THRESHOLD * avg_h:
            c['is_merged'] = True
    
    # Sort results
    final_chars.sort(key=lambda c: (c['x'], c['y']))
    
    bboxes, x_mins, x_maxs, y_mins, y_maxs, widths, heights, areas, avgs, is_merged_list, crops_list = [], [], [], [], [], [], [], [], [], [], []
    merged_indices = []
    
    for i, c in enumerate(final_chars):
        crop = c['crop']
        
        # Invert to standard black-on-white (or white-on-black depending on input)
        # Original logic was `255 - crop`. Assuming 'crop' is 0=black, 255=white (binary).
        # But 'crop' comes from 'line_source' which is `~thresh` (inverted).
        # So `255 - crop` flips it back to normal reading (black text on white bg).
        char_img = 255 - crop
        
        # Create Title Header
        h, w = char_img.shape
        header_h = 20
        # Create white background for header
        header = np.full((header_h, w), 255, dtype=np.uint8) 
        
        # Put Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        text = f"{i+1}"
        (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thickness)
        # Center text
        t_x = (w - t_w) // 2
        t_y = (header_h + t_h) // 2
        cv2.putText(header, text, (t_x, t_y), font, scale, 0, thickness) # 0 = Black text
        
        # Stack vertically
        final_img = np.vstack((header, char_img))
        
        # cv2.imwrite(os.path.join(out_dir, f"char_{i+1}.png"), final_img)
        
        bboxes.append((c['x'], c['y'], c['w'], c['h']))
        x_mins.append(c['x'])
        x_maxs.append(c['x'] + c['w'])
        y_mins.append(c['y'])
        y_maxs.append(c['y'] + c['h'])
        widths.append(c['w'])
        heights.append(c['h'])
        areas.append(c['w'] * c['h'])
        avgs.append(float(np.mean(crop)))
        is_merged_list.append(c['is_merged'])
        crops_list.append(c['crop'])
        if c['is_merged']:
            merged_indices.append(i + 1)

    # if merged_indices:
    #     print(f"  [!] Merged characters suspected at indices: {merged_indices} (Directory: {os.path.basename(out_dir)})")

    # Montage for visualization
    if final_chars:
        max_h_c = max(c['h'] for c in final_chars)
        total_w_c = sum(c['w'] for c in final_chars) + (len(final_chars) * 6)
        montage = np.full((max_h_c + 20, total_w_c + 20, 3), 220, dtype=np.uint8)
        curr_x = 10
        for c in final_chars:
            disp_char = cv2.cvtColor(255 - c['crop'], cv2.COLOR_GRAY2BGR)
            # Use RED for merged, GRAY for normal
            color = (0, 0, 255) if c['is_merged'] else (150, 150, 150)
            cv2.rectangle(disp_char, (0, 0), (c['w'] - 1, c['h'] - 1), color, 1)
            montage[10:10+c['h'], curr_x:curr_x+c['w']] = disp_char
            curr_x += c['w'] + 6
        # cv2.imshow(f"Chars (Global) - {os.path.basename(out_dir)}", montage)

    metrics = {
        'bboxes': bboxes, 'x_min': x_mins, 'x_max': x_maxs, 'y_min': y_mins, 'y_max': y_maxs,
        'widths': widths, 'heights': heights, 'areas': areas, 'avgs': avgs, 'is_merged': is_merged_list,
        'crops': crops_list
    }
    return len(final_chars), metrics
