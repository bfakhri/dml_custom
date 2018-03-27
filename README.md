## Original

The original [DeepMind Lab github](https://github.com/deepmind/lab).

## Troubleshooting Notes: 

If you get a checksum error, replace the corresponding checksum in WORKSPACE with the checksum that is expected

If you get "configure: error: unsafe absolute working directory name", you must set a custom working directory that doesn't have any weird characters. This is done with "bazel --output_user_root=/path/to/directory YOUR_ORIGINAL_COMMAND"

## Running an environment

It is important to note that asset generation can take **up to 10 minutes** when performed for the first time. Just be patient!

### Play as a human

```shell
cd lab

lab$ bazel run :game -- -l tests/empty_room_test -s logToStdErr=false
# is short for
lab$ bazel run :game -- --level_script=tests/empty_room_test --level_setting=logToStdErr=false
```

### Run a random agent

```shell
lab$ bazel run :python_random_agent --define graphics=sdl -- \
	       --length=10000 --width=640 --height=480
```

### Train an agent

*Taken directly from [Deepmind Lab](https://github.com/deepmind/lab).*

*DeepMind Lab* ships with an example random agent in
[`python/random_agent.py`](python/random_agent.py)
which can be used as a starting point for implementing a learning agent. To let
this agent interact with DeepMind Lab for training, run

```shell
lab$ bazel run :python_random_agent
```

The Python API for the agent-environment interaction is described
in [docs/python_api.md](docs/python_api.md).

*DeepMind Lab* ships with different levels implementing different tasks. These
tasks can be configured using Lua scripts,
as described in [docs/lua_api.md](docs/lua_api.md).

-----------------

## Editing the files

### Reward Values

```lab/game_scripts/common/pickups.lua```

pickups.lua is where all of the player-available pickups are listed, along with their names, 3D model reference, and their reward (refered to as quantity in the code). This is also where each pickup is defined as either a pickup item or a goal item. Goal items are the same as pickup items, with the exception that they will reset the level upon pickup.

### Level Layout

```lab/game_scripts/levels```

This is where the playable levels exist. For this example, we will be using the test level empty_room_test.lua, located at:

```lab/game_scripts/levels/tests/empty_room_test.lua```

DeepMind includes a text-to-tile level-generating parser. The level layout in empty_room_test.lua is as follows:

```
local MAP_ENTITIES = [[
***********
*         *
*    P    *
*         *
*** A S ***
*MI     IG*
*** L F ***
*         *
*   *H*   *
*   *W*   *
***********
]]
```

The key is as follows:

Environment:
- Asterisk (*): Wall
- Space ( ): Empty tile
- I: North-South sliding door
- H: East-West sliding door

Player:
- P: Player spawn

Pickups:
- A: Apple
- F: Fungus
- L: Lemon
- S: Strawberry

Goals:
- G: Goal
- M: Mango
- W: Watermelon

### Extra Entities

DeepMind provides an additional function to place extra pickups onto the map in completely custom locations, and with completely custom rewards.

An example of the function in use is given in: 
```lab/game_scripts/levels//tests/extra_entities_test.lua```

In this file, note that the ```MAP_ENTITIES``` is a simple 7x7 room with the player spawn at the center. The function ```api:extraEntities``` allows the user to define a list of additional entities to be included, following the form:

```
return {
      {
          classname = 'apple_reward',
          origin = '550 450 30',
          count = '1',
      },
      ...
}
```

The ```classname``` is the same that is defined in pickups.lua. The ```origin``` is the x, y, z coordinates in 3D space that the pickup will spawn at. The ```count``` is the reward the object will give. Note that this reward will only apply to this entity; If apples are defined in ```pickups.lua``` to have a reward (quantity) of 7, then all apples placed using ```MAP_ENTITIES``` will have a reward of 7, but this apple placed using ```api:extraEntities()``` will have a reward (count) of 1.
