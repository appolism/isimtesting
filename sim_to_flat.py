# Flat File is named 'sim_to_flat.txt'
import omni.usd
from pxr import UsdGeom

#variables
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/jetbot")

xform = UsdGeom.Xformable(prim)
matrix = xform.ComputeLocalToWorldTransform(0)
position = matrix.ExtractTranslation()

pos_list = list(position)

#code
with open("sim_to_flat.txt", "w") as f:
	f.write(f"Position(x, y, z): {pos_list[0]}, {pos_list[1]}, {pos_list[2]}")
	
with open("sim_to_flat.txt") as f:
	print(f.read())









































