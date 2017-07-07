# x, y, w, hの4パラメータを保持するだけのクラス
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def int_left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x - half_width)), int(round(self.y - half_height)))

    def left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x - half_width, self.y - half_height]

    def int_right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x + half_width)), int(round(self.y + half_height)))

    def right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x + half_width, self.y + half_height]

    def crop_region(self, h, w):
        left, top = self.left_top()
        right, bottom = self.right_bottom()
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        self.w = right - left
        self.h = bottom - top
        self.x = (right + left) / 2
        self.y = (bottom + top) / 2
        return self

