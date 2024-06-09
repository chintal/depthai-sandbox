

import numpy as np
from vedo import Plotter, Grid, Cone, Box, Plane, Sphere


class RenderableObject(object):
    element_type = 'generic'

    def __init__(self, name, color=None):
        self.name = name
        self.color = color
        self._elems = []

    def render(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.element_type:10} {self.name:12}'

class RenderableCamera(RenderableObject):
    element_type = 'camera'

    def __init__(self, geom, *args, **kwargs):
        self.geom = geom
        super().__init__(*args, **kwargs)

    def render(self):
        if not self._elems:
            #TODO Render all camera origins with cones
            camera_pos = (0, 0, -10)
            self._elems.append(Box(pos=camera_pos, length=80, width=30, height=20, c="black"))
            cone = Cone(axis=(0, 0, -1), height=1000, r=1200, res=100, pos=(0, 0, +500))
            cone.color("lightgray").alpha(0.1)
            self._elems.append(cone)
        return self._elems

class RenderablePlane(RenderableObject):
    element_type = 'plane'

    def __init__(self, centroid, normal, *args, **kwargs):
        self._centroid = centroid
        self._normal = normal
        super().__init__(*args, **kwargs)

    @property
    def centroid(self):
        return self._centroid
    
    @centroid.setter
    def centroid(self, value):
        if not np.array_equal(value, self._centroid):
            self._elems = []
        self._centroid = value

    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, value):
        if not np.array_equal(value, self._normal):
            self._elems = []
        self._normal = value

    def render(self):
        if not self._elems:
            self._elems.append(Plane(pos=self.centroid, normal=self.normal, s=(1000,1000),
                                     c=self.color, alpha=0.4))
        return self._elems

    def __repr__(self):
        return super().__repr__() + f' centroid {self.centroid}; normal {self.normal}'


class RenderableMarker(RenderableObject):
    element_type = 'marker'

    def __init__(self, tvec, size=50, *args, **kwargs):
        self._tvec = None
        self.tvec = tvec
        self.size = size
        super().__init__(*args, **kwargs)

    @property
    def tvec(self):
        return self._tvec
    
    @tvec.setter
    def tvec(self, value):
        if isinstance(value, dict):
            value = np.array([value['x'], value['y'], value['z']])
        if self._tvec is None or not np.array_equal(value, self._tvec):
            self._elems = []
        self._tvec = value

    def render(self):
        if not self._elems:
            pos = self.tvec.flatten()
            print(pos)
            self._elems.append(Sphere(pos=pos, r=self.size/2, c=self.color))
        return self._elems

    def __repr__(self):
        return super().__repr__() + f' tvec {self.tvec.tolist()}'


class SceneReconstruction(object):
    def __init__(self):
        self.elements = {}
        self._target : Plotter = None

    def _clear_elements(self, name):
        to_remove = []
        for e in self.elements.keys():
            if e.startswith(name):
                to_remove.append(e)
        for e in to_remove:
            del self.elements[e]

    def upsert_camera(self, geom):
        if 'camera' in self.elements.keys():
            # We assume the coordinate systems are fixed. 
            pass
        else:
            self.elements['camera'] = RenderableCamera(name='camera', geom=geom)

    def clear_plane(self, name):
        self._clear_elements(name)

    def upsert_plane(self, name, centroid, normal, color=None):
        if centroid is None or normal is None:
            return
        if name in self.elements.keys():
            self.elements[name].centroid = centroid
            self.elements[name].normal = normal
        else:
            self.elements[name] = RenderablePlane(centroid=centroid, normal=normal, name=name, color=color)

    def clear_markers(self, name):
        self._clear_elements(name)

    def upsert_marker(self, name, tvec, color=None):
        if name in self.elements.keys():
            self.elements[name].tvec = tvec
        else:
            self.elements[name] = RenderableMarker(tvec=tvec, name=name, color=color)

    def remove_element(self, name):
        if name in self.elements:
            del self.elements[name]

    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, target):
        self._target = target

    def update_scene(self):
        self.target.clear(at=0)
        elements_to_show = False
        
        for name, elem_spec in self.elements.items():
            elem_spec : RenderableObject
            elems = elem_spec.render()
            for elem in elems:
                self.target.at(0).show(elem)
                elements_to_show = True

        if not elements_to_show:
            empty_grid = Grid()
            self.target.at(0).show(empty_grid)

        self.target.axes = 1
        print(self.target)
        
    def close(self):
        pass
