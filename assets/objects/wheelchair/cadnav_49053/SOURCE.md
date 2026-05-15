# Wheelchair with Shackle Asset

Source: https://www.cadnav.com/3d-models/model-49053.html

Downloaded: 2026-05-15

Original package:
- `source/Wheelchair.obj`
- `source/Wheelchair.mtl`
- `source/Wheelchair.bmp`
- `source/Metal_Clips.bmp`
- `source/Restraints.bmp`
- `source/readme.txt`

Converted runtime asset:
- `wheelchair_cadnav_49053.glb`

CadNav lists this model as "Wheelchair with Shackle", model ID 49053. The page lists the format as Wavefront OBJ with MTL and BMP textures, and the model page marks the license as non-commercial. The included readme says the files may be used commercially as part of artwork or a project, should not be resold or redistributed as a standalone model, may be modified, and should indicate cadnav.com as the source.

Conversion notes:

```bash
convert Metal_Clips.bmp Metal_Clips.png
convert Restraints.bmp Restraints.png
convert Wheelchair.bmp Wheelchair.png
perl -0pi -e 's/\\.bmp/.png/g' Wheelchair.mtl
npx --yes obj2gltf -i Wheelchair.obj -o wheelchair_cadnav_49053.glb -b --secure
```
