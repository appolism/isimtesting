from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


class DroneTakeoff(BehaviorScript):

    TARGET_HEIGHT = 1.0
    KP = 4.0
    KD = 3.0
    GRAVITY = 9.81
    MAX_ACCELERATION = 20.0

    def on_init(self):
        self.rigid_body = UsdPhysics.RigidBodyAPI(self.prim)

        if not self.rigid_body:
            raise RuntimeError(
                f"{self.prim.GetPath()} does not have Rigid Body API"
            )

        mass_api = UsdPhysics.MassAPI(self.prim)
        self.mass = mass_api.GetMassAttr().Get()

        if self.mass is None or self.mass <= 0:
            raise RuntimeError(
                f"{self.prim.GetPath()} does not have a valid mass"
            )

        self.mass = float(self.mass)

        self.force_api = PhysxSchema.PhysxForceAPI.Apply(self.prim)
        self.force_api.CreateWorldFrameEnabledAttr().Set(True)
        self.force_api.CreateForceAttr().Set(
            Gf.Vec3f(0.0, 0.0, 0.0)
        )

        self.target_z = None

    def on_play(self):
        transform = UsdGeom.Xformable(
            self.prim
        ).ComputeLocalToWorldTransform(0.0)

        start_z = float(
            transform.ExtractTranslation()[2]
        )

        self.target_z = start_z + self.TARGET_HEIGHT

    def on_update(self, current_time, delta_time):
        if self.target_z is None:
            return

        transform = UsdGeom.Xformable(
            self.prim
        ).ComputeLocalToWorldTransform(current_time)

        current_z = float(
            transform.ExtractTranslation()[2]
        )

        velocity = self.rigid_body.GetVelocityAttr().Get()

        if velocity is None:
            vertical_velocity = 0.0
        else:
            vertical_velocity = float(velocity[2])

        height_error = self.target_z - current_z

        acceleration = (
            self.GRAVITY
            + self.KP * height_error
            - self.KD * vertical_velocity
        )

        acceleration = max(
            0.0,
            min(acceleration, self.MAX_ACCELERATION)
        )

        thrust = self.mass * acceleration

        self.force_api.GetForceAttr().Set(
            Gf.Vec3f(0.0, 0.0, thrust)
        )

    def on_stop(self):
        self.force_api.GetForceAttr().Set(
            Gf.Vec3f(0.0, 0.0, 0.0)
        )

    def on_destroy(self):
        if hasattr(self, "force_api"):
            self.force_api.GetForceAttr().Set(
                Gf.Vec3f(0.0, 0.0, 0.0)
            )
