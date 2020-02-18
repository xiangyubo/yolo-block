# -*- coding: UTF-8 -*-
"""
聚合预测晶格中间的晶格类
"""


class Grid:

    def __init__(self, x, y, links, bbox, conf, end_type):
        self._x = x
        self._y = y
        self._links = links
        self._bbox = bbox
        self._conf = conf
        self._end_type = end_type

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def links(self):
        return self._links

    @property
    def bbox(self):
        return self._bbox

    @property
    def conf(self):
        return self._conf

    @property
    def end_type(self):
        return self._end_type

    @end_type.setter
    def end_type(self, end_type):
        self._end_type = end_type
