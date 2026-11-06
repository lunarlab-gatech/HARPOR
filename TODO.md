TO DO LIST
- instead of relying on DA3 for camera intrinsics and extrinsics, the user should pass that data in if they have it. Eventually, in the multi user version, they should be able to have customizable preset scanners they can choose
- support for reconstructing multiple objects at once (or many in a queue from a single command)

- run psi0 on reconstructed objects for a quick baseline
- LiDAR supervision scales the entire reconstructed object, can’t fix scale of specific sides of an object, need to improve this

IN PROGRESS
- refine lidar supervision
- have an option to visualize the difference between actual LiDAR points and DA3 depth estimations
- remove bloat and hard-coded features from HARPOR

