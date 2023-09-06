# ash2048
---

> Implementation of the 2048 Game for Gym Environments with state rendering through PyGame.

## Usage and Notes:

The package follows gymnasium conventions.
To render the board while playing, set `display=True` in the constructor.
To save the render to a gif, set `save_animation=True` in the constructor.
To save the generated animation to a file, call `save_render(filename)` where the filaname is the outfile gif file.
The animation is reset upon calling `reset()`.