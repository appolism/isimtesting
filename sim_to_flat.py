# Flat File is named 'sim_to_flat.txt'
import omni.usd
import omni.timeline
import omni.kit.app
from pxr import UsdGeom
from isaacsim.core.experimental.utils import xform

try:
	subscription.unsubscribe()
except:
	pass

#variables
#robot_pos, robot_orientation = xform.get_world_pose("/World/jetbot/chassis")
#robot_pos_np = robot_pos.numpy()
app = omni.kit.app.get_app()
frame_count = 0
robot_pos_np = [0, 0, 0]

#functions
def robot_pos_update(event):
	global frame_count
	global robot_pos_np
	timeline = omni.timeline.get_timeline_interface()
	if not timeline.is_playing():
		return
		
	frame_count += 1
	
	if frame_count % 45 != 0:
		return
	
	robot_pos, robot_orientation = xform.get_world_pose("/World/jetbot/chassis")
	robot_pos_np = robot_pos.numpy()
	with open("sim_to_flat.txt", "w") as f:
		f.write(f"Position(x, y, z): {robot_pos_np[0]}, {robot_pos_np[1]}, {robot_pos_np[2]}")
	
	with open("sim_to_flat.txt") as f:
		print(f.read())

#code
subscription = app.get_update_event_stream().create_subscription_to_pop( robot_pos_update )

#with open("sim_to_flat.txt", "w") as f:
	#f.write(f"Position(x, y, z): {robot_pos_np[0]}, {robot_pos_np[1]}, {robot_pos_np[2]}")
	
#with open("sim_to_flat.txt") as f:
	#print(f.read())
	
#print("lalala")

































































































