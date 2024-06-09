
from PyQt6.QtWidgets import QSplitter

class QCustomSplitter(QSplitter):
    def __init__(self, orientation, min_width_c1, fixed_width_c2, *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)
        self.fixed_width_c2 = fixed_width_c2
        self.min_width_c1 = min_width_c1
        self.user_resized = False
        self.setCollapsible(0, False)
        self.setCollapsible(1, False)
        self.initial_size_set = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.user_resized and not self.initial_size_set:
            self.set_initial_sizes()

    def set_initial_sizes(self):
        sizes = self.sizes()
        total_width = self.width()
        first_size = total_width - self.fixed_width_c2
        if first_size < self.min_width_c1:
            first_size = self.min_width_c1
        self.setSizes([first_size, self.fixed_width_c2])
        self.initial_size_set = True

    def splitterMoved(self, pos, index):
        super().splitterMoved(pos, index)
        self.user_resized = True