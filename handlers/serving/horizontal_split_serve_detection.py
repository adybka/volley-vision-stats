def horizontal_split_serve_detection(bounding_box, inputHeight):
    threshold = inputHeight / 2
    x1, y1, x2, y2 = map(int, bounding_box)
    center_y = (y1 + y2) / 2
    if center_y < threshold:
        team_label = 0 #close
        color = (0, 255, 0)  # Green
    else:
        team_label = 1 #far
        color = (0, 0, 255)  # Red
    return team_label, color
