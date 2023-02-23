import pyglet
import os
import sys
import pathlib

def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width / 2
    image.anchor_y = image.height / 2

path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
# Tell pyglet where to find the resources
pyglet.resource.path = [str(path.joinpath(r"../resources"))]
pyglet.resource.reindex()

agent_image = pyglet.resource.image("drone.png")
center_image(agent_image)

target_image = pyglet.resource.image("target.png")
center_image(target_image)

protected_space = pyglet.resource.image("protected_space.png")
center_image(protected_space)