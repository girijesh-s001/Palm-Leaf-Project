import cv2
import os
import numpy as np
import glob
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import char_seg

def preprocess(img, is_crop=False, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sharpen the image to enhance edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 11)
    inv = ~thresh
    
    # Adjust thresholds if processing a character crop
    min_area_cont = 2 if is_crop else 10
    max_area_cont = 14000 if not is_crop else 100000 # essentially disabled for crops
    cc_area_thresh = 10 if is_crop else 800
    
    # contours for removing noise by boundary-based filtering
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = thresh.copy()
    for cnt in contours:
        if not (min_area_cont < cv2.contourArea(cnt) < max_area_cont):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(cleaned, (x, y), (x + w, y + h), 255, -1)
            
    # CC for removing noise by size-based(connected pixels) filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(~cleaned, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > cc_area_thresh:
            if not is_crop: # Logic for standard page/line: remove large blobs
                cleaned[labels == i] = 255
        else:
            if is_crop: # Logic for crop: remove tiny noise components
                 cleaned[labels == i] = 255

    # Visualizations (only if requested)
    if show:
        cv2.imshow("Preprocess Input", img)
        cv2.imshow("Preprocess Cleaned", ~cleaned)

    # Return both clean binary (for logic) and raw threshold (for visualization)
    return ~cleaned, ~thresh, sharpened

def estimate_height(binary):
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, len(stats)) 
               if 5 < stats[i, cv2.CC_STAT_HEIGHT] < 80 and stats[i, cv2.CC_STAT_WIDTH] > 3]
    return max(int(np.median(heights)), 10) if heights else 25

def detect_separators(binary, text_h):
    profile = np.sum(binary == 255, axis=1).astype(np.float64)
    smooth = gaussian_filter1d(profile, sigma=max(text_h / 6, 1.5))
    min_dist = max(int(text_h * 0.6), 8)
    best_v = []
    for f in [0.15, 0.1, 0.05, 0.02, 0.01]:
        peaks, _ = find_peaks(smooth, distance=min_dist, prominence=f * np.max(smooth))
        if len(peaks) < 2: continue
        valleys = []
        for j in range(len(peaks) - 1):
            seg = smooth[peaks[j]:peaks[j+1]+1]
            v = peaks[j] + np.argmin(seg)
            if (min(smooth[peaks[j]], smooth[peaks[j+1]]) - smooth[v]) > 0.02 * min(smooth[peaks[j]], smooth[peaks[j+1]]):
                valleys.append(v)
        if len(valleys) > len(best_v): best_v = valleys
    bg = (binary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
    row_dist = np.mean(dist, axis=1)
    r = max(text_h // 3, 5)
    return [y - r + np.argmax(row_dist[max(0, y-r):min(len(row_dist), y+r+1)]) for y in best_v]

def dp_trace(binary, y_start):
    h, w = binary.shape
    # Cost based on distance to nearest text (prefer large white gaps)
    bg = (binary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
    
    # Text crossing penalty (increase to discourage cutting through characters)
    # Background cost is inverse of distance transform (farther from text is cheaper)
    cost = (np.max(dist) + 1 - dist).astype(np.float32)
    cost[binary > 0] += 250 # Higher penalty for text
    
    path = np.zeros(w, dtype=np.int32)
    path[0] = np.clip(y_start, 0, h - 1)
    
    for x in range(1, w):
        best_y, min_c = path[x-1], cost[path[x-1], x]
        # Search range + penalty for vertical deviation to stay smooth
        for dy in [-2, -1, 0, 1, 2]:
            ny = path[x-1] + dy
            if 0 <= ny < h:
                # Add vertical distance penalty and bias towards starting y
                bias = abs(ny - y_start) * 0.1
                c = cost[ny, x] + abs(dy) * 5 + bias
                if c < min_c:
                    min_c, best_y = c, ny
        path[x] = best_y
    return path

def process_image(path):
    img = cv2.imread(path)
    if img is None: return
    print(f"Processing: {os.path.basename(path)}")
    cleaned, thresh_inv, sharpened = preprocess(img, show=False)
    
    text_h = estimate_height(cleaned)
    reinforced = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(text_h * 2, 30), 1)))
    seps = detect_separators(cleaned, text_h)
    paths = [dp_trace(reinforced, y) for y in seps]
    if len(paths) > 1:
        paths.sort(key=lambda p: np.median(p))
        merged = [paths[0]]
        for p in paths[1:]:
            if np.mean(np.abs(p.astype(float) - merged[-1].astype(float))) > max(text_h // 3, 6):
                merged.append(p)
        paths = merged

    # 1. First Pass: Extract lines and get initial blocks
    h, w = cleaned.shape
    inner = sorted(paths, key=lambda p: np.median(p))
    full_paths = [np.zeros(w, dtype=np.int32)] + inner + [np.full(w, h - 1, dtype=np.int32)]
    
    line_data = []
    all_initial_blocks = []
    
    for i in range(len(full_paths) - 1):
        if np.median(full_paths[i+1]) - np.median(full_paths[i]) < 8: continue
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.vstack((np.column_stack((np.arange(w), full_paths[i])), np.flipud(np.column_stack((np.arange(w), full_paths[i+1]))))).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        y_idx, x_idx = np.where(mask == 255)
        if len(y_idx) == 0 or np.sum(cleaned[mask == 255] == 255) < 50: continue
        
        line_binary = np.zeros((h, w), dtype=np.uint8)
        line_binary[mask == 255] = cleaned[mask == 255]
        
        # Also extract line from raw thresh and sharpened image
        line_thresh = np.zeros((h, w), dtype=np.uint8)
        line_thresh[mask == 255] = thresh_inv[mask == 255]
        line_sharpened = np.zeros((h, w), dtype=np.uint8)
        line_sharpened[mask == 255] = sharpened[mask == 255]
        
        y_min, y_max, x_min, x_max = np.min(y_idx), np.max(y_idx), np.min(x_idx), np.max(x_idx)
        line_crop_bin = line_binary[y_min:y_max+1, x_min:x_max+1]
        line_crop_thresh = line_thresh[y_min:y_max+1, x_min:x_max+1]
        line_crop_sharpened = line_sharpened[y_min:y_max+1, x_min:x_max+1]
        
        initial_blocks = char_seg.get_initial_blocks(line_crop_bin)
        all_initial_blocks.extend(initial_blocks)
        line_data.append({
            'bin': line_crop_bin, 'thresh': line_crop_thresh, 'sharpened': line_crop_sharpened, 'mask': mask, 
            'y_min': y_min, 'x_min': x_min, 'initial_blocks': initial_blocks
        })

    if not all_initial_blocks:
        print("  No text detected.")
        return

    # 2. Calculate Global Stats
    avg_w = np.median([b['w'] for b in all_initial_blocks])
    avg_h = np.median([b['h'] for b in all_initial_blocks])

    # 3. Second Pass: Splitting and Saving
    out = os.path.join("output", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "cleaned.png"), 255 - cleaned)
    cv2.imwrite(os.path.join(out, "thresh.png"), 255 - thresh_inv)
    
    vis_crops = []
    all_final_chars = []
    total_char_count = 0
    
    for i, L in enumerate(line_data):
        line_num = i + 1
        
        # PER-LINE STATS instead of global
        if L['initial_blocks']:
            line_avg_w = np.median([b['w'] for b in L['initial_blocks']])
            line_avg_h = np.median([b['h'] for b in L['initial_blocks']])
            # Blend with global if count is low
            if len(L['initial_blocks']) < 5:
                line_avg_w = (line_avg_w + avg_w) / 2
                line_avg_h = (line_avg_h + avg_h) / 2
        else:
            line_avg_w, line_avg_h = avg_w, avg_h

        cv2.imwrite(os.path.join(out, f"line_{line_num}.png"), 255 - L['thresh'])
        char_dir = os.path.join(out, f"line_{line_num}_chars")
        
        # Pass both binary (for logic) and thresh (for potential visualization/cropping)
        # Note: function signature update required in char_seg.py
        cnt, metrics = char_seg.adaptive_split_and_save(L['bin'], L['initial_blocks'], line_avg_w, line_avg_h, char_dir, line_source=L['thresh'])
        
        # Collect data for final logic
        for j in range(cnt):
            total_char_count += 1
            all_final_chars.append({
                's_no': total_char_count,
                'line_idx': i, # Keep track of which line this belongs to
                'local_bbox': metrics['bboxes'][j], # (bx, by, bw, bh) relative to line
                'abs_x': L['x_min'] + metrics['x_min'][j],
                'abs_y': L['y_min'] + metrics['y_min'][j],
                'w': metrics['widths'][j],
                'h': metrics['heights'][j],
                'area': metrics['areas'][j],
                'aspect_ratio': round(metrics['widths'][j] / metrics['heights'][j], 2) if metrics['heights'][j] > 0 else 0,
                'crop': metrics['crops'][j],
                'is_merged': metrics['is_merged'][j]
            })

    print(f"  Saved {len(line_data)} lines to {out}")

    # --- TWO-STAGE FILTERING & IN-PLACE SPLITTING ---
    if all_final_chars:
        # 1. Global Averages
        avg_w_global = sum(c['w'] for c in all_final_chars) / len(all_final_chars)
        avg_h_global = sum(c['h'] for c in all_final_chars) / len(all_final_chars)
        
        # 2. Stage 1: Shortlist
        stage1_thresh_w = 1.6 * avg_w_global
        stage1_thresh_h = 1.6 * avg_h_global
        shortlist = [c for c in all_final_chars if c['w'] > stage1_thresh_w or c['h'] > stage1_thresh_h]
        
        # 3. Stage 2: Calculate average width of shortlisted characters
        if shortlist:
            avg_w_shortlisted = sum(c['w'] for c in shortlist) / len(shortlist)
            # 4. Final Threshold (8% buffer)
            final_thresh_w = 1.08 * avg_w_shortlisted
        else:
            final_thresh_w = 9999 # Nothing meets global criteria

        # --- NEW: Save each character as separate image & Save Joint Characters as RGB ---
        final_char_dir = os.path.join(out, "final_characters")
        joint_rgb_dir = os.path.join(out, "joint_characters_rgb")
        os.makedirs(final_char_dir, exist_ok=True)
        
        # Track which original characters were identified as joints
        joint_bboxes_to_save = []
        
        # 5. Process ALL characters, splitting where necessary
        complete_final_list = []
        for c in all_final_chars:
            # Check if this character should be split
            should_split = False
            if c['w'] > stage1_thresh_w or c['h'] > stage1_thresh_h: # Passed Stage 1
                if c['w'] > final_thresh_w: # Passed Stage 2
                    should_split = True
            
            if should_split:
                # Store for RGB saving (identified by average values before any structural split check)
                joint_bboxes_to_save.append({
                    's_no': c['s_no'],
                    'abs_x': c['abs_x'],
                    'abs_y': c['abs_y'],
                    'w': c['w'],
                    'h': c['h']
                })
                
                crop = c['crop']
                h_c, w_c = crop.shape
                v_hist = np.sum(crop == 255, axis=0)
                
                # Search in the middle 30% region (from 35% to 65% of width)
                start_x = int(w_c * 0.35)
                end_x = int(w_c * 0.65)
                
                if end_x > start_x:
                    search_area = v_hist[start_x:end_x]
                    min_val = np.min(search_area)
                    # Find the leftmost index of the absolute minimum within the middle 30%
                    split_idx = start_x + np.where(search_area == min_val)[0][0]
                else:
                    # Fallback for very narrow characters
                    split_idx = w_c // 2

                # NEW: Refined Joint character detection
                # Calculate peaks outside the search area to check for a significant valley
                peak_left = np.max(v_hist[:start_x]) if start_x > 0 else 0
                peak_right = np.max(v_hist[end_x:]) if end_x < w_c else 0
                max_peak = max(peak_left, peak_right)
                
                # Criteria for a "true" joint between two characters:
                # 1. Aspect Ratio > 1.6
                # 2. Significant valley: min value in middle area is < 30% of BOTH neighboring peaks
                # 3. Significant peaks: neighboring peaks must be at least 50% of char height
                aspect_ratio = w_c / h_c if h_c > 0 else 0
                min_neighbor_peak = min(peak_left, peak_right) if peak_left > 0 and peak_right > 0 else 0
                is_joint = (aspect_ratio > 1.6) and \
                           (min_neighbor_peak > 0.5 * h_c) and \
                           (min_val < 0.3 * min_neighbor_peak)
                
                if is_joint and end_x > start_x:
                    bx, by, bw, bh = c['local_bbox']
                    
                    # Part A
                    crop_a = crop[:, :split_idx]
                    complete_final_list.append({
                        **c, 's_no': f"{c['s_no']}a", 'w': split_idx, 'crop': crop_a,
                        'local_bbox': (bx, by, split_idx, bh),
                        'area': int(np.sum(crop_a == 255)),
                        'aspect_ratio': round(split_idx / h_c, 2) if h_c > 0 else 0,
                        'is_split': True
                    })
                    
                    # Part B
                    crop_b = crop[:, split_idx:]
                    complete_final_list.append({
                        **c, 's_no': f"{c['s_no']}b", 'abs_x': c['abs_x'] + split_idx,
                        'local_bbox': (bx + split_idx, by, w_c - split_idx, bh),
                        'w': w_c - split_idx, 'crop': crop_b,
                        'area': int(np.sum(crop_b == 255)),
                        'aspect_ratio': round((w_c - split_idx) / h_c, 2) if h_c > 0 else 0,
                        'is_split': True
                    })
                else:
                    complete_final_list.append(c)
            else:
                complete_final_list.append(c)

        # Save RGB Joint Characters if any found
        if joint_bboxes_to_save:
            os.makedirs(joint_rgb_dir, exist_ok=True)
            joint_vis_data = [] # Store (rgb_crop, cleaned_crop, s_no)
            for j in joint_bboxes_to_save:
                # Crop from original RGB image
                x, y, w, h = j['abs_x'], j['abs_y'], j['w'], j['h']
                # Safety check for boundaries
                y_end, x_end = min(y + h, img.shape[0]), min(x + w, img.shape[1])
                rgb_crop = img[y:y_end, x:x_end]
                
                # Preprocess the individual character crop
                cleaned_c, _, _ = preprocess(rgb_crop, is_crop=True)
                
                rgb_path = os.path.join(joint_rgb_dir, f"joint_char_{j['s_no']}_rgb.png")
                cv2.imwrite(rgb_path, rgb_crop)
                joint_vis_data.append((rgb_crop, cleaned_c, j['s_no']))
            
            print(f"  Saved {len(joint_bboxes_to_save)} joint RGB characters to {joint_rgb_dir}")
            
            # Visualization for Joint Characters removed to display only specified images

        # 6. Console Output (Full List)
        print(f"\n" + "="*80)
        print(f"COMPLETE CHARACTER LIST (Two-Stage Buffer: 60% Stage1, 8% Stage2)")
        print(f"Global Avg: W={avg_w_global:.2f}, H={avg_h_global:.2f}")
        if shortlist:
            print(f"Refined Shortlist Avg Width: {avg_w_shortlisted:.2f} (Threshold: {final_thresh_w:.2f})")
        print(f"Total Detected: {len(all_final_chars)} | Final Count (Splits Included): {len(complete_final_list)}")
        print(f"="*80)
        
        print(f"{'S.No':<10} {'Abs X':<8} {'Abs Y':<8} {'Width':<8} {'Height':<8} {'Area':<8} {'Aspect'}")
        print("-" * 80)
        for c in complete_final_list:
            tag = " [SPLIT]" if c.get('is_split') else ""
            print(f"{str(c['s_no']):<10} {c['abs_x']:<8} {c['abs_y']:<8} {c['w']:<8} {c['h']:<8} {c['area']:<8} {c['aspect_ratio']}{tag}")
            
            # Save individual image
            char_filename = f"char_{c['s_no']}.png"
            char_path = os.path.join(final_char_dir, char_filename)
            # Invert back to black-on-white for saving if desired, but typical is 255 for text
            # The current crop is binary with 255 as text. Let's save as is.
            cv2.imwrite(char_path, c['crop'])
            
        print("="*80)
        print(f"  Saved {len(complete_final_list)} character images to {final_char_dir}")
        print("="*80)

        # ── SEGMENTATION ACCURACY / QUALITY REPORT ────────────────────────────
        print(f"\n{'='*80}")
        print("SEGMENTATION ACCURACY REPORT")
        print(f"{'='*80}")

        total_final = len(complete_final_list)
        split_chars  = sum(1 for c in complete_final_list if c.get('is_split'))
        merged_chars = sum(1 for c in complete_final_list if c.get('is_merged') and not c.get('is_split'))
        
        # Good character: not merged and not split-produced (i.e., confidently single)
        good_chars = total_final - merged_chars - split_chars

        # Aspect-ratio quality: ideal Tamil/script chars ~ 0.4–1.8
        ar_ok   = sum(1 for c in complete_final_list if 0.3 <= c.get('aspect_ratio', 0) <= 2.0)
        ar_bad  = total_final - ar_ok

        # Width outlier: chars whose width deviates > 80% from global avg
        w_ok    = sum(1 for c in complete_final_list
                      if 0.2 * avg_w_global <= c['w'] <= 1.8 * avg_w_global)
        w_bad   = total_final - w_ok

        # Accuracy score:  penalise merged (under-seg) and bad-AR chars
        # Score = (good + split) / total  (splits were resolved, so credit them)
        accuracy_score = ((good_chars + split_chars) / total_final * 100) if total_final > 0 else 0.0

        # Per-line breakdown
        print(f"\n{'Line':<8} {'Chars':<10} {'Merged':<10} {'Split':<10} {'Good':<10}")
        print("-"*50)
        for li, L in enumerate(line_data):
            lc = [c for c in complete_final_list if c['line_idx'] == li]
            lm = sum(1 for c in lc if c.get('is_merged') and not c.get('is_split'))
            ls = sum(1 for c in lc if c.get('is_split'))
            lg = len(lc) - lm - ls
            print(f"  L{li+1:<6} {len(lc):<10} {lm:<10} {ls:<10} {lg:<10}")

        print("-"*50)
        print(f"\n  Total Characters (final)  : {total_final}")
        print(f"  Initially Detected        : {len(all_final_chars)}")
        print(f"  Split (joint resolved)    : {split_chars}")
        print(f"  Likely Merged (flagged)   : {merged_chars}")
        print(f"  Clean / Good Chars        : {good_chars}")
        print(f"  Aspect-Ratio OK  (0.3–2.0): {ar_ok}  |  Outliers: {ar_bad}")
        print(f"  Width Within 80% of avg  : {w_ok}  |  Outliers: {w_bad}")
        print(f"\n  *** Segmentation Accuracy Score : {accuracy_score:.2f}% ***")
        print(f"  (Score = clean+split chars / total final chars)")
        print(f"{'='*80}\n")

        # 7. Final Visualization: Input Image & Segmented Characters
        h_img, w_img = img.shape[:2]
        segmented_img = np.full((h_img, w_img, 3), 255, dtype=np.uint8)
        
        # Create line segmented image
        line_segmented_img = img.copy()
        for p in inner:
            pts = np.column_stack((np.arange(len(p)), p)).astype(np.int32)
            cv2.polylines(line_segmented_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            
        for c in complete_final_list:
            x, y, w, h = c['abs_x'], c['abs_y'], c['w'], c['h']
            crop = c['crop']
            
            y_start = max(0, y)
            y_end = min(h_img, y + h)
            x_start = max(0, x)
            x_end = min(w_img, x + w)
            
            crop_y_start = y_start - y
            crop_y_end = crop_y_start + (y_end - y_start)
            crop_x_start = x_start - x
            crop_x_end = crop_x_start + (x_end - x_start)
            
            if crop_y_end > crop_y_start and crop_x_end > crop_x_start:
                roi = segmented_img[y_start:y_end, x_start:x_end]
                roi_crop = crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                
                # Draw the character in black (where crop is 255, set to 0)
                for ch in range(3):
                    roi[:, :, ch][roi_crop == 255] = 0
                
                # Draw bounding box and label
                color = (0, 0, 255) if c.get('is_split') or c.get('is_merged') else (255, 0, 0)
                cv2.rectangle(segmented_img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(segmented_img, str(c['s_no']), (x, max(0, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Save the outputs to the folder
        cv2.imwrite(os.path.join(out, "input_image.png"), img)
        cv2.imwrite(os.path.join(out, "line_segmented.png"), line_segmented_img)
        cv2.imwrite(os.path.join(out, "char_segmented.png"), segmented_img)

        cv2.imshow("Input Image", img)
        cv2.imshow("Line Segmented Image", line_segmented_img)
        cv2.imshow("Segmented Characters", segmented_img)

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    print("Please select a palm leaf image file...")
    img_path = filedialog.askopenfilename(
        title="Select Palm Leaf Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if img_path:
        process_image(img_path)
        print("\nDone. Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image selected. Exiting.")