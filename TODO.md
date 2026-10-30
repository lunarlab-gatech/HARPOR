TO DO LIST
- pass the known K to DA3 when a job has one (`inference(intrinsics=...)`, from the calibration file, camera_info or Stray's K), so the mesh fit stops using DA3's estimated K
- add an optional input for external camera poses (a bag's `/tf` or a pose file like the Pocket2's `img_pos.txt`) and pass them to DA3 as `extrinsics`
- add a check on s that doesn't depend on the calibration: fit DA3's camera centers times s to the external poses, and warn when they disagree. A calibration error shifts every view the same way, so the spread and outlier checks can't catch it
- let a video job upload intrinsics, since a video job has no calibration input
- remove the hard-coded GPU arch
- physagent3D mujoco running
- have an option to visualize the difference between actual LiDAR points and DA3 depth estimations
- have a total time indicator after the entire run finishes

IN PROGRESS
- refine lidar supervision