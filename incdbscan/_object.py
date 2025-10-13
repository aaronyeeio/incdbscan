NodeId = int
ObjectId = int


class Object:
    def __init__(self, id_, min_pts, weight=1.0):
        self.id: ObjectId = id_
        self.node_id: NodeId = None
        self.weight = weight
        self.neighbors = {self}
        self.merge_neighbors = {self}
        self.neighbor_count = 0
        self.min_pts = min_pts

    @property
    def is_core(self):
        return self.neighbor_count >= self.min_pts

    def __repr__(self):
        return f'{self.id}_'
