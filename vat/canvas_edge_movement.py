from PyQt5.QtCore import Qt, QRect

# Edge detection constants
EDGE_NONE = 0
EDGE_TOP = 1
EDGE_RIGHT = 2
EDGE_BOTTOM = 3
EDGE_LEFT = 4

def detect_edge(rect, pos, threshold=8):
    """
    Detect if the cursor is near an edge of the rectangle.
    
    Args:
        rect: QRect object
        pos: QPoint cursor position
        threshold: Distance threshold to consider cursor near an edge
        
    Returns:
        Edge constant (EDGE_NONE, EDGE_TOP, EDGE_RIGHT, EDGE_BOTTOM, EDGE_LEFT)
    """
    if not rect.adjusted(-threshold, -threshold, threshold, threshold).contains(pos):
        return EDGE_NONE
        
    # Check if cursor is near any edge
    if abs(pos.y() - rect.top()) <= threshold:
        return EDGE_TOP
    elif abs(pos.x() - rect.right()) <= threshold:
        return EDGE_RIGHT
    elif abs(pos.y() - rect.bottom()) <= threshold:
        return EDGE_BOTTOM
    elif abs(pos.x() - rect.left()) <= threshold:
        return EDGE_LEFT
    
    return EDGE_NONE

def get_edge_cursor(edge):
    """Return the appropriate cursor for the given edge"""
    if edge in (EDGE_TOP, EDGE_BOTTOM):
        return Qt.SizeVerCursor
    elif edge in (EDGE_LEFT, EDGE_RIGHT):
        return Qt.SizeHorCursor
    return Qt.ArrowCursor

def move_edge(rect, edge, pos, start_pos):
    """
    Move a specific edge of the rectangle based on cursor movement.
    
    Args:
        rect: QRect to modify
        edge: Which edge to move (EDGE_TOP, EDGE_RIGHT, etc.)
        pos: Current cursor position
        start_pos: Starting cursor position
        
    Returns:
        Modified QRect
    """
    delta_x = pos.x() - start_pos.x()
    delta_y = pos.y() - start_pos.y()
    
    new_rect = QRect(rect)
    
    if edge == EDGE_TOP:
        new_top = rect.top() + delta_y
        if new_top < rect.bottom():
            new_rect.setTop(new_top)
    elif edge == EDGE_RIGHT:
        new_right = rect.right() + delta_x
        if new_right > rect.left():
            new_rect.setRight(new_right)
    elif edge == EDGE_BOTTOM:
        new_bottom = rect.bottom() + delta_y
        if new_bottom > rect.top():
            new_rect.setBottom(new_bottom)
    elif edge == EDGE_LEFT:
        new_left = rect.left() + delta_x
        if new_left < rect.right():
            new_rect.setLeft(new_left)
    
    return new_rect