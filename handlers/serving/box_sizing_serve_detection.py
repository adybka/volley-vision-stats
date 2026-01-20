def box_sizing_serve_detection(bounding_box, inputHeight):
    # arbitrary threshold at 57% of frame height
    threshold = inputHeight * 0.57
    x1, y1, x2, y2 = map(int, bounding_box)
    box_height = y2 - y1
    if box_height > threshold:
        team_label = 0 #close
        color = (0, 255, 0)  # Green
    else:
        team_label = 1 #far
        color = (0, 0, 255)  # Red
    return team_label, color
