# Credits

ComfyUI-GeometryPack is made possible by wrapping and building upon the following
amazing open-source geometry processing libraries. We are deeply grateful to the
authors and communities behind each of these projects.

## Currently Integrated Libraries

Ordered from least to most stringent license.

| Library | License | Description | Homepage |
|---------|---------|-------------|----------|
| [trimesh](https://github.com/mikedh/trimesh) | MIT | Core mesh I/O, queries, convex hull, proximity, and boolean operations | https://trimsh.org |
| [fast-simplification](https://github.com/pyvista/fast-simplification) | MIT | Fast quadric mesh simplification (~10x faster than classic implementations) | https://github.com/pyvista/fast-simplification |
| [PyVista](https://github.com/pyvista/pyvista) | MIT | VTK-based 3D visualization, mesh analysis, and DecimatePro decimation | https://pyvista.org |
| [xatlas](https://github.com/jpcy/xatlas) | MIT | Automatic UV atlas generation and lightmap UV unwrapping for triangle meshes | https://github.com/jpcy/xatlas |
| [xatlas-python](https://github.com/mworchel/xatlas-python) | MIT | Python bindings for xatlas | https://github.com/mworchel/xatlas-python |
| [Open3D](https://github.com/isl-org/Open3D) | MIT | 3D data processing — point clouds, meshes, RGBD; used via mesh-to-sdf | https://www.open3d.org |
| [mesh-to-sdf](https://github.com/marian42/mesh_to_sdf) | MIT | Converts triangle meshes to signed distance fields | https://github.com/marian42/mesh_to_sdf |
| [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils) | MIT | Point cloud and mesh utilities — Hausdorff/Chamfer distance, sampling | https://github.com/fwilliams/point-cloud-utils |
| [CuMesh](https://github.com/JeffreyXiang/CuMesh) | MIT | CUDA-accelerated mesh processing — GPU UV unwrapping and hole filling | https://github.com/JeffreyXiang/CuMesh |
| [pyQuadriFlow](https://github.com/hjwdzh/QuadriFlow) | MIT | Scalable and robust quad remeshing (used in Blender) | https://github.com/hjwdzh/QuadriFlow |
| [pmp-library](https://github.com/pmp-library/pmp-library) | MIT | Polygon Mesh Processing Library — uniform and adaptive isotropic remeshing | https://www.pmp-library.org |
| [pypmp](https://github.com/PozzettiAndrea/pypmp) | MIT | Python bindings for pmp-library (nanobind) | https://github.com/PozzettiAndrea/pypmp |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | Fundamental N-dimensional array operations and scientific computing | https://numpy.org |
| [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause | Scientific algorithms — Delaunay triangulation, spatial operations, optimization | https://scipy.org |
| [PyNanoInstantMeshes](https://github.com/vork/PyNanoInstantMeshes) | BSD-3-Clause | Python bindings for Instant Meshes field-aligned quad remeshing | https://github.com/vork/PyNanoInstantMeshes |
| [Instant Meshes](https://github.com/wjakob/instant-meshes) | BSD-3-Clause | Field-aligned quad-dominant remeshing (original C++ implementation) | https://github.com/wjakob/instant-meshes |
| [embreex](https://github.com/trimesh/embreex) | BSD-3-Clause | Python wrapper for Intel Embree high-performance ray tracing | https://github.com/trimesh/embreex |
| [geogram](https://github.com/BrunoLevy/geogram) | BSD-3-Clause | Voronoi diagrams, centroidal Voronoi tessellation, parameterization, remeshing (INRIA) | https://github.com/BrunoLevy/geogram |
| [pygeogram-ap](https://github.com/PozzettiAndrea/pygeogram) | BSD-3-Clause | Python bindings for geogram CVT remeshing, booleans, repair, reconstruction (nanobind) | https://github.com/PozzettiAndrea/pygeogram |
| [libigl](https://github.com/libigl/libigl) | MPL-2.0 | Geometry processing — signed distance, UV parameterization (LSCM/harmonic), winding number, booleans | https://libigl.github.io |
| [skeletor](https://github.com/navis-org/skeletor) | GPL-3.0 | Skeleton extraction from triangle meshes (wavefront, vertex clusters, edge collapse, TEASAR) | https://github.com/navis-org/skeletor |
| [pymeshfix](https://github.com/pyvista/pymeshfix) | GPL-3.0 | Automatic mesh repair — hole filling, self-intersection removal, watertight conversion | https://github.com/pyvista/pymeshfix |
| [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) | GPL-3.0 | Python bindings for MeshLab — isotropic remeshing, Poisson/ball-pivoting reconstruction, smoothing, decimation | https://github.com/cnr-isti-vclab/PyMeshLab |
| [MeshLab](https://github.com/cnr-isti-vclab/meshlab) | GPL-3.0 | The open-source mesh processing system (CNR-ISTI Visual Computing Lab) | https://www.meshlab.net |
| [CGAL](https://www.cgal.org) | GPL-3.0 / LGPL-3.0 | Computational Geometry Algorithms Library — isotropic remeshing, alpha wrap, booleans, edge collapse, self-intersection repair. Dual licensed: algorithms are GPL-3.0, kernel/support components are LGPL-3.0. Commercial licenses available from GeometryFactory. | https://www.cgal.org |
| [quadwild](https://github.com/nicopietroni/quadwild) | GPL-3.0 | Feature-line driven quad remeshing for producing quad-dominant meshes (BiMDF solver) | https://github.com/nicopietroni/quadwild |
| [pyquadwild](https://github.com/PozzettiAndrea/pyquadwild) | GPL-3.0 | Python bindings for QuadWild BiMDF tri-to-quad remeshing (nanobind) | https://github.com/PozzettiAndrea/pyquadwild |
| [Blender (bpy)](https://www.blender.org) | GPL-3.0 | Blender's Python module — voxel/modifier remeshing, booleans, UV projection, texture remeshing | https://www.blender.org |

## Future Integrations

We hope to also wrap the following outstanding libraries in future releases:

| Library | License | Description | Homepage |
|---------|---------|-------------|----------|
| [manifold3d](https://github.com/elalish/manifold) | Apache-2.0 | Robust, GPU-friendly CSG/boolean operations on manifold meshes | https://github.com/elalish/manifold |
| [gmsh](https://gmsh.info) | GPL-2.0+ | 3D finite element mesh generator with built-in CAD engine; linking exception available | https://gmsh.info |

## License Notes

- **CGAL** uses a dual-license model per component. Geometric algorithm packages (Surface Mesh Simplification, Polygon Mesh Processing, etc.) are GPL-3.0. Kernel and support components are LGPL-3.0. Commercial licenses are available from [GeometryFactory](https://geometryfactory.com).
- **PyMesh** license situation is ambiguous — `setup.py` references MPL-2.0 but no formal LICENSE file is present in the repository.
- **MeshLib** is source-available but not OSI-approved open source. Free for evaluation and non-commercial use; commercial use requires a paid license.
- **gmsh** is GPL-2.0-or-later but includes a linking exception that allows use as a library without triggering GPL obligations on the calling code.
- **GeometryPack itself** is licensed under GPL-3.0-or-later, which is compatible with all of the above.