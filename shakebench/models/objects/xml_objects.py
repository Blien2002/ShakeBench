"""ShakeBench task objects selected from the RoboCasa asset library."""

from robosuite.models.objects import MujocoXMLObject
from shakebench.models import xml_path_completion


class FoodCanObject(MujocoXMLObject):
    """RoboCasa canned-food visual with a non-rolling box collision primitive."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/food_can.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class CookieBoxObject(MujocoXMLObject):
    """RoboCasa boxed-food visual with a simple rectangular collision primitive."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/cookie_box.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
