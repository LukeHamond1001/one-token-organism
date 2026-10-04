"""the eye's own copies of the G1's visual meshes, decimated (speed only: the physics keeps Unitree's files; body/sim/g1scene.py
load_model points each visual geom at its copy here). Run once; the copies are kept in git so every machine renders the same."""
import pathlib, sys
import vtk
SRC = pathlib.Path("body/sim/assets/unitree_g1/assets"); DST = pathlib.Path("body/sim/assets/unitree_g1/assets_vis")
MIN_FACES = 4000          # a mesh under this is left alone
KEEP = 0.12               # the share of faces kept
FLOOR = 1500              # and never under this many
DST.mkdir(exist_ok=True)
for f in sorted(SRC.glob("*.STL")):
    r = vtk.vtkSTLReader(); r.SetFileName(str(f)); r.Update()
    n = r.GetOutput().GetNumberOfPolys()
    if n < MIN_FACES:
        continue
    cl = vtk.vtkCleanPolyData(); cl.SetInputConnection(r.GetOutputPort())
    d = vtk.vtkQuadricDecimation(); d.SetInputConnection(cl.GetOutputPort())
    d.SetTargetReduction(1.0 - max(KEEP, FLOOR / n)); d.VolumePreservationOn(); d.Update()
    w = vtk.vtkSTLWriter(); w.SetFileTypeToBinary(); w.SetFileName(str(DST / f.name)); w.SetInputConnection(d.GetOutputPort()); w.Write()
    print(f"{f.name}: {n} -> {d.GetOutput().GetNumberOfPolys()} faces")
