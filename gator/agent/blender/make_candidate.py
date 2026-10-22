"""Build candidate.blend from original.glb before the agent's clock starts (build_plan.md note 5).

blender -b --factory-startup --python-exit-code 1 --python make_candidate.py -- ORIGINAL.glb CANDIDATE.blend

The stock glTF importer maps glTF to W, as in review.py. Images are packed, so the .blend does not
depend on file paths.
"""

import sys

import bpy

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=dst)
