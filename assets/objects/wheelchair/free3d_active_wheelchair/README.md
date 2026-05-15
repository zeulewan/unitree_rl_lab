# Free3D Active Manual Wheelchair

Local asset wrapper for the Free3D `active-wheelchair-82422` manual wheelchair model.

The downloaded Free3D files are intentionally ignored by Git because the fork is public and the asset is listed as personal-use. Keep the archive and normalized visual files local unless the repository is made private or the license is cleared for redistribution.

Expected local archive:

```bash
assets/objects/wheelchair/free3d_active_wheelchair/source/y2ztdkf6sg00-ActiveWheelchair_likeKueschall.rar
```

Regenerate the normalized OBJ/MTL/textures from the archive:

```bash
python scripts/assets/import_free3d_active_wheelchair.py
```

The passive physics wrapper is committed at:

```bash
assets/objects/wheelchair/free3d_active_wheelchair/urdf/active_manual_wheelchair.urdf
```

The URDF uses simple primitive collisions and joints for stable simulation. The detailed Free3D OBJ is visual-only.
