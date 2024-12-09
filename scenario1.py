# Cell
LOCAL_PATH_ROHAN = "file:///home/rbosworth/Diffusion/ManipFinalProject/assets/"
LOCAL_PATH_RICHARD = "file:///Users/earlight/Desktop/MIT/-Fall-2024/6.4210/project/ManipulationFinal/assets/"

LOCAL_PATH = LOCAL_PATH_RICHARD # Change this to LOCAL_PATH_ROHAN if you are Rohan

total_moles_contacted = 0

# Cell
import numpy as np
import time
TOTAL_START_TIME = time.time()
from sklearn.cluster import DBSCAN
from pydrake.geometry import StartMeshcat
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.visualization import MeshcatPoseSliders
from manipulation import running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant, AddPointClouds
from manipulation.systems import AddIiwaDifferentialIK, MultibodyPositionToBodyPose
from manipulation.utils import FindResource
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import (
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    PointCloud,
    SceneGraph,
    Simulator,
    Parser,
    RigidTransform,
    SpatialVelocity,
    AbstractValue,
    MeshcatPointCloudVisualizer,
    InverseKinematics,
    AddMultibodyPlantSceneGraph,
    Solve,
    SnoptSolver,
    MosekSolver,
    KinematicTrajectoryOptimization,
    Toppra,
    PathParameterizedTrajectory,
    ContactResults,
    DiscreteTimeDelay,
    Concatenate,
    GcsTrajectoryOptimization,
    GraphOfConvexSetsOptions,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile,
    IrisOptions,
    Point
)

import multiprocessing as mp
import os.path
import time
from collections import OrderedDict
from typing import Dict

import numpy as np
import pydot
from IPython.display import SVG, display
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    Meshcat,
    MeshcatVisualizer,
    QueryObject,
    Rgba,
    Role,
    SceneGraph,
    Sphere,
    StartMeshcat,
)
from pydrake.geometry.optimization import (
    HPolyhedron,
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import PackageMap, Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import Body
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer
from manipulation.station import LoadScenario, MakeMultibodyPlant
from manipulation.utils import FindDataResource

# Cell
# Start the visualizer.
meshcat = StartMeshcat()
meshcat.SetProperty("/Background", "top_color", [0.17254901960784313, 0.403921568627451, 0.9490196078431372])
meshcat.SetProperty("/Background", "bottom_color", [0.3843137254901961, 0.8117647058823529, 0.9568627450980393])
meshcat.SetProperty("/Lights/SpotLight", "visible", True)
meshcat.SetProperty("/Grid", "visible", False)
meshcat.SetProperty("/Axes", "visible", False)
meshcat.SetProperty("/drake/contact_forces", "visible", False)

# Cell
MOLE_COUNT = 1  # Change this value to spawn more moles
MAIN_TABLE_COORDS = [0, 0, 0]
STORAGE_TABLE_COORDS = [100, 0, 0]
MOLE_COORDS = [[STORAGE_TABLE_COORDS[j] + (-1 + 0.2 * i if j == 0 else 0) for j in range(3)] for i in range(MOLE_COUNT)]
SCHUNK_WSG_DRIVER = "{}"

# Generate mole directives dynamically
mole_directives = "\n".join(
    f"""
    - add_model:
        name: sphere_mole_{i + 1:02}
        file: {LOCAL_PATH}sphere_mole.sdf
        default_free_body_pose:
            __model__:
                translation: [{MOLE_COORDS[i][0]}, {MOLE_COORDS[i][1]}, 0.015]
    """ for i in range(MOLE_COUNT)
)

scenario_data_multiple = f'''
    directives:
    - add_directives:
        file: package://manipulation/iiwa_and_wsg.dmd.yaml

    - add_model:
        name: main_table
        file: {LOCAL_PATH}round_table.sdf
    - add_weld:
        parent: world
        child: main_table::table_top_center
        X_PC:
            translation: {MAIN_TABLE_COORDS}

    - add_model:
        name: storage_table
        file: {LOCAL_PATH}round_table.sdf
    - add_weld:
        parent: world
        child: storage_table::table_top_center
        X_PC:
            translation: {STORAGE_TABLE_COORDS}

    {mole_directives}

    # Camera 1
    - add_frame:
        name: camera1_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{ deg: [-120.0, 0, 90.0]}}
            translation: [2.5, 0, .8]

    - add_model:
        name: camera1
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera1_origin
        child: camera1::base

    # Camera 2
    - add_frame:
        name: camera2_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{ deg: [-120.0, 0, 180.0]}}
            translation: [0, 2.5, .8]

    - add_model:
        name: camera2
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera2_origin
        child: camera2::base

    # Camera 3
    - add_frame:
        name: camera3_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{ deg: [-120.0, 0, -90.0]}}
            translation: [-2.5, 0, .8]

    - add_model:
        name: camera3
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera3_origin
        child: camera3::base

    # Camera 4
    - add_frame:
        name: camera4_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{ deg: [-120.0, 0, 0.0]}}
            translation: [0, -2.5, .8]

    - add_model:
        name: camera4
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera4_origin
        child: camera4::base

    cameras:
        camera1:
            name: camera1
            depth: True
            X_PB:
                base_frame: camera1::base
        camera2:
            name: camera2
            depth: True
            X_PB:
                base_frame: camera2::base
        camera3:
            name: camera3
            depth: True
            X_PB:
                base_frame: camera3::base
        camera4:
            name: camera4
            depth: True
            X_PB:
                base_frame: camera4::base

    plant_config:
        time_step: 0.02
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            hand_model_name: wsg
            control_mode: position_only
        wsg: !SchunkWsgDriver {SCHUNK_WSG_DRIVER}
    lcm_buses:
        default:
            lcm_url: ""
'''


# Cell
MAX_SPAWN_DISTANCE = 0.6
MIN_SPAWN_DISTANCE = 0.9

def spawn_mole(instance, plant, simulator):
    "Returns the position of the spawned mole"

    # Generate random position on the table
    mole_distance = np.random.uniform(MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE)
    mole_angle = np.random.uniform(np.pi / 4, np.pi * 3 / 4) # scenario 1: spawn moles in 90 degrees in front of the robot
    mole_position = [
        mole_distance * np.sin(mole_angle), 
        mole_distance * np.cos(mole_angle), 
        0.015
    ]

    # Get the model instance of the mole
    mole_instance = f"sphere_mole_{instance+1:02d}"
    mole_model_instance = plant.GetModelInstanceByName(mole_instance)

    # Get the body of the mole
    mole_body = plant.GetBodyByName("sphere_link", mole_model_instance)

    # Move the mole to the new position
    context = plant.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    plant.SetFreeBodyPose(context, mole_body, RigidTransform(mole_position))

    return mole_position

def despawn_mole(instance, plant, simulator):
    # Get the model instance of the mole
    mole_instance = f"sphere_mole_{instance+1:02d}"
    mole_model_instance = plant.GetModelInstanceByName(mole_instance)

    # Get the body of the mole
    mole_body = plant.GetBodyByName("sphere_link", mole_model_instance)

    # Move the mole to the new position
    context = plant.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    plant.SetFreeBodyPose(context, mole_body, RigidTransform(MOLE_COORDS[instance]))

def despawn_close_mole(plant, simulator, ee_position):
    for i in range(MOLE_COUNT):
        mole_instance = f"sphere_mole_{i+1:02d}"
        mole_model_instance = plant.GetModelInstanceByName(mole_instance)
        mole_body = plant.GetBodyByName("sphere_link", mole_model_instance)
        context = plant.GetMyMutableContextFromRoot(simulator.get_mutable_context())
        mole_position = plant.EvalBodyPoseInWorld(context, mole_body).translation()
        if np.linalg.norm(mole_position[:2] - ee_position[:2]) < 0.1:
            despawn_mole(i, plant, simulator)
            spawn_mole(i, plant, simulator)

# Cell
iris_options = IrisOptions()
iris_options.iteration_limit = 10
iris_options.configuration_space_margin = 0.00001
# increase num_collision_infeasible_samples to improve the (probabilistic)
# certificate of having no collisions.
iris_options.num_collision_infeasible_samples = 3
iris_options.require_sample_point_is_contained = True
iris_options.relative_termination_threshold = 0.01
iris_options.termination_threshold = -1

# Additional options for this notebook:

# If use_existing_regions_as_obstacles is True, then iris_regions will be
# shrunk by regions_as_obstacles_margin, and then passed to
# iris_options.configuration_obstacles.
use_existing_regions_as_obstacles = False
regions_as_obstacles_scale_factor = 0.95

# Cell
iris_regions = LoadIrisRegionsYamlFile("iiwa_regions.yaml")

def LoadRobotIRIS(plant: MultibodyPlant):
    """Setup your plant, and return the body corresponding to your
    end-effector."""
    parser = Parser(plant)

    # We'll use some tables, shelves, and bins from a remote resource.
    parser.package_map().AddRemote(
        package_name="gcs",
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/mpetersen94/gcs/archive/refs/tags/arxiv_paper_version.tar.gz"
            ],
            sha256=("6dd5e841c8228561b6d622f592359c36517cd3c3d5e1d3e04df74b2f5435680c"),
            strip_prefix="gcs-arxiv_paper_version",
        ),
    )

    model_directives = f"""    
directives:

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.3]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1]
        iiwa_joint_7: [1.57]

- add_weld:
    parent: world
    child: iiwa::iiwa_link_0

# Add schunk
- add_model:
    name: wsg
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.06]
      rotation: !Rpy {{ deg: [90.0, 0.0, 90.0 ]}}

- add_model:
    name: main_table
    file: {LOCAL_PATH}round_table.sdf
- add_weld:
    parent: world
    child: main_table::table_top_center
    X_PC:
        translation: [0, 0, 0]
"""

    parser.AddModelsFromString(model_directives, ".dmd.yaml")

    return plant

def LoadRobot(plant: MultibodyPlant):
    """Setup your plant, and return the body corresponding to your
    end-effector."""
    parser = Parser(plant)

    # We'll use some tables, shelves, and bins from a remote resource.
    parser.package_map().AddRemote(
        package_name="gcs",
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/mpetersen94/gcs/archive/refs/tags/arxiv_paper_version.tar.gz"
            ],
            sha256=("6dd5e841c8228561b6d622f592359c36517cd3c3d5e1d3e04df74b2f5435680c"),
            strip_prefix="gcs-arxiv_paper_version",
        ),
    )

    model_directives = f"""    
directives:

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.3]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1]
        iiwa_joint_7: [1.57]

- add_weld:
    parent: world
    child: iiwa::iiwa_link_0

# Add schunk
- add_model:
    name: wsg
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.06]
      rotation: !Rpy {{ deg: [90.0, 0.0, 90.0 ]}}

- add_model:
    name: main_table
    file: {LOCAL_PATH}round_table.sdf
- add_weld:
    parent: world
    child: main_table::table_top_center
    X_PC:
        translation: [0, 0, 0]
"""

    parser.AddModelsFromString(model_directives, ".dmd.yaml")
    gripper = plant.GetModelInstanceByName("wsg")
    end_effector_body = plant.GetBodyByName("body", gripper)
    return end_effector_body


def ScaleHPolyhedron(hpoly, scale_factor):
    # Shift to the center.
    xc = hpoly.ChebyshevCenter()
    A = hpoly.A()
    b = hpoly.b() - A @ xc
    # Scale
    b = scale_factor * b
    # Shift back
    b = b + A @ xc
    return HPolyhedron(A, b)


def _CheckNonEmpty(region):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(region.ambient_dimension())
    region.AddPointInSetConstraints(prog, x)
    result = Solve(prog)
    assert result.is_success()


def _CalcRegion(name, seed):
    builder = DiagramBuilder()
    plant = AddMultibodyPlantSceneGraph(builder, 0.0)[0]
    LoadRobotIRIS(plant)
    plant.Finalize()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, seed)
    if use_existing_regions_as_obstacles:
        iris_options.configuration_obstacles = [
            ScaleHPolyhedron(r, regions_as_obstacles_scale_factor)
            for k, r in iris_regions.items()
            if k != name
        ]
        for h in iris_options.configuration_obstacles:
            _CheckNonEmpty(h)

    display(f"Computing region for seed: {name}")
    start_time = time.time()
    hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)
    display(
        f"Finished seed {name}; Computation time: {(time.time() - start_time):.2f} seconds"
    )

    _CheckNonEmpty(hpoly)
    reduced = hpoly.ReduceInequalities()
    _CheckNonEmpty(reduced)

    return reduced


def GenerateRegion(name, seed):
    global iris_regions
    iris_regions[name] = _CalcRegion(name, seed)
    #SaveIrisRegionsYamlFile(f"iiwa_regions.yaml", iris_regions)

# Cell
class PlantIK():
    def __init__(self):
        builder = DiagramBuilder()
        self.plant = AddMultibodyPlantSceneGraph(builder, 0.0)[0]
        LoadRobot(self.plant)
        self.plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(diagram_context)

        self.gripper_frame = self.plant.GetFrameByName("body")

        self.eps = np.array([0.001, 0.001, 0.001])
    
    def do_ik(self, q):
        ik = InverseKinematics(self.plant, self.plant_context)

        ik.AddMinimumDistanceLowerBoundConstraint(0.001)

        ik.AddPositionConstraint(self.gripper_frame, [0, 0, 0], self.plant.world_frame(),
                             np.array(q) - self.eps, np.array(q) + self.eps)

        prog = ik.get_mutable_prog()
        q = ik.q()

        q0 = [0 for i in range(7)]

        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)
        result = Solve(ik.prog())
        if not result.is_success():
            print("IK failed")
            return [None]
        q1 = result.GetSolution(q)
        return np.concatenate((q1, [0, 0]))

class Planner():
    def __init__(self, plant):
        super().__init__()
        # self.trajOpt = KinematicTrajectoryOptimization(plant.num_positions(), 10)
        self.trajOpt = GcsTrajectoryOptimization(7) # testing
        self.trajOpt.AddPathLengthCost()
        self.trajOpt.AddTimeCost()

        velocity_lb = plant.GetVelocityLowerLimits()[:7]
        velocity_ub = plant.GetVelocityUpperLimits()[:7]
        velocity_lb = np.where(np.isinf(velocity_lb), -1e6, velocity_lb)
        velocity_ub = np.where(np.isinf(velocity_ub), 1e6, velocity_ub)

        self.regions = iris_regions

        self.trajOpt.AddVelocityBounds(velocity_lb, velocity_ub) # testing
        self.trajOptRegions = self.trajOpt.AddRegions(list(self.regions.values()), order=3)
        self.plant = plant
        self.options=GraphOfConvexSetsOptions()
        self.options.solver = MosekSolver()

    def solve_path(self, x0, xf):
        start_included = False
        end_included = False
        for reg in self.regions.values():
            if reg.PointInSet(x0):
                start_included = True
            if reg.PointInSet(xf):
                end_included = True

        if not start_included:
            name = f"start_reg_{len(self.regions)}"
            GenerateRegion(name, x0)
            self.trajOpt.RemoveSubgraph(self.trajOptRegions)
            self.trajOptRegions = self.trajOpt.AddRegions(list(self.regions.values()), order=3)

        if not end_included:
            name = f"end_reg_{len(self.regions)}"
            GenerateRegion(name, xf)
            self.trajOpt.RemoveSubgraph(self.trajOptRegions)
            self.trajOptRegions = self.trajOpt.AddRegions(list(self.regions.values()), order=3)        

        start_pt = self.trajOpt.AddRegions([Point(x0)], order=0)
        end_pt = self.trajOpt.AddRegions([Point(xf)], order=0)

        self.trajOpt.AddEdges(start_pt, self.trajOptRegions)
        self.trajOpt.AddEdges(self.trajOptRegions, end_pt)

        traj, result = self.trajOpt.SolvePath(start_pt, end_pt)

        self.trajOpt.RemoveSubgraph(start_pt)
        self.trajOpt.RemoveSubgraph(end_pt)

        builder = DiagramBuilder()
        plant = AddMultibodyPlantSceneGraph(builder, 0.0)[0]
        LoadRobot(plant)
        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()

        toppra = Toppra(
                traj,
                plant,
                np.linspace(traj.start_time(), traj.end_time(), 1000),
            )
            
        toppra.AddJointVelocityLimit(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
        time_traj = toppra.SolvePathParameterization()
        return PathParameterizedTrajectory(traj, time_traj)
        

class PositionOutput(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorOutputPort("fixed_pos", 7, self.return_zeros)
        self.dummy_in = self.DeclareVectorInputPort("dummy", 1)

    def return_zeros(self, context, output):
        self.dummy_in.Eval(context)
        output.SetFromVector(np.zeros(7))

class WSGPositionOutput(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorOutputPort("fixed_pos", 1, self.return_zeros)

    def return_zeros(self, context, output):
        output.SetFromVector(np.zeros(1))

# Cell
class TrajectoryFollower(LeafSystem):
    def __init__(self, plant, plant_full):
        super().__init__()
        self.DeclareVectorOutputPort("iiwa_position", 7, self.calculate_positions)

        self.iiwa_pos = self.DeclareVectorInputPort("iiwa_pos", 7)
        self.ik_sol = self.DeclareVectorInputPort("IK_solution", 7)

        self.planner = Planner(plant)

        self.plant = plant
        self.plant_full = plant_full

        self.traj = None

        self.first = True

        self.prior = False

        self.traj_start = 0

        """
        States:
        0 - running to mole
        1 - Diff IK down to hit mole
        2 - running to start
        1.5  - waiting for mole
        """

        self.state = 0

        self.start_height = None

        # self.DeclareAbstractInputPort("time", AbstractValue.Make(0.0))

    def calculate_positions(self, context, output):            
        if self.first:
            self.first = False
            target_pos = self.ik_sol.Eval(context)

            start_pos = self.iiwa_pos.Eval(context)

            self.traj = self.planner.solve_path(start_pos, target_pos)

            
        if self.state == 0:
            current_time = context.get_time()

            if current_time - self.traj_start > self.traj.end_time():
                self.traj = None
                self.state = 1
                output.SetFromVector(self.iiwa_pos.Eval(context))
            else:
                # Evaluate the trajectory at the current time
                joint_positions = self.traj.value(current_time - self.traj_start)
                
                # Set the output to the evaluated positions
                output.SetFromVector(joint_positions)

        elif self.state == 1:
            plant_cont = self.plant.CreateDefaultContext()
            curr_pos = np.concatenate((self.iiwa_pos.Eval(context), [0, 0]))

            self.plant.SetPositions(plant_cont, curr_pos)
            current_height = self.plant.EvalBodyPoseInWorld(plant_cont, self.plant.GetBodyByName("body")).translation()[2]
            if self.start_height == None:
                self.start_height = current_height

            contact = False
            mole_indices = self.plant_full.GetBodyIndices(self.plant_full.GetModelInstanceByName("sphere_mole_01"))
            wsg_indices = self.plant_full.GetBodyIndices(self.plant_full.GetModelInstanceByName("wsg"))

            pairs = [(a, b) for a in mole_indices for b in wsg_indices]


            contact = self.start_height - current_height > 0.005
            
            if contact:
                print("=========================================================================")
                print("contact made")
                global total_moles_contacted
                total_moles_contacted += 1
                despawn_close_mole(plant, simulator, self.plant.EvalBodyPoseInWorld(plant_cont, self.plant.GetBodyByName("body")).translation())
                if self.prior:
                    for i in range(MOLE_COUNT):
                        despawn_mole(i, plant, simulator)
                    for i in range(MOLE_COUNT):
                        spawn_mole(i, plant, simulator)

                target_pos = self.ik_sol.Eval(context)
                
                if sum(target_pos) == 0:
                    self.prior = True
                    output.SetFromVector(self.iiwa_pos.Eval(context))
                    return


                self.prior = False
                start_pos = self.iiwa_pos.Eval(context)                

                self.traj = self.planner.solve_path(start_pos, target_pos)

                self.traj_start = context.get_time()

                self.state = 0

                output.SetFromVector(self.iiwa_pos.Eval(context))
            
            else:
                params = DifferentialInverseKinematicsParameters(self.plant.num_positions(),
                                                                self.plant.num_velocities())
                newPos = np.array(DoDifferentialInverseKinematics(self.plant, plant_cont, [0, 0, 0, 0, 0,-1], self.plant.GetFrameByName("body"), params).joint_velocities)[0:7]
                newPos = np.array(curr_pos)[0:7] + 0.1 * newPos
                output.SetFromVector(newPos)



# Cell
PROCESSING_TIME = 0.5

class Perception(LeafSystem):
    def __init__(self, plant):
        super().__init__()

        self.means = [] # mole locations

        self.point_cloud1 = self.DeclareAbstractInputPort(
            "camera1", AbstractValue.Make(PointCloud()))
        self.point_cloud2 = self.DeclareAbstractInputPort(
            "camera2", AbstractValue.Make(PointCloud()))
        self.point_cloud3 = self.DeclareAbstractInputPort(
            "camera3", AbstractValue.Make(PointCloud()))
        self.point_cloud4 = self.DeclareAbstractInputPort(
            "camera4", AbstractValue.Make(PointCloud()))

        self.DeclareAbstractOutputPort(
            "crop_cloud", lambda: AbstractValue.Make(PointCloud()), self.crop_cloud)

        self.DeclareVectorOutputPort("IK_sol", 7, self.identify_moles) # testing

        self.crop_lower = [-10,-10,0.01]
        self.crop_upper = [10,10,0.05]

        # print(plant.GetPositions(plant.CreateDefaultContext()))

        self.ik = PlantIK()

        self._last_processed_time = None

    def crop_cloud(self, context, output):
        current_time = context.get_time()
        if (self._last_processed_time and current_time - self._last_processed_time < PROCESSING_TIME):
            return
        self._last_processed_time = current_time

        # Retrieve the point cloud, but Eval(context) too slow
        point_cloudO = self.get_input_port(0).Eval(context)
        
        # Process the point cloud: crop and downsample
        point_cloudO = point_cloudO.Crop(self.crop_lower, self.crop_upper)#.VoxelizedDownSample(voxel_size=0.005)
        point_cloud = point_cloudO.xyzs()
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.1, min_samples=3).fit(point_cloud.T).labels_
        
        # Group points by labels into lists
        clustered_points = {label : [] for label in set(dbscan)}

        pc_rgb = np.zeros(point_cloud.shape).T
        
        for i, label in enumerate(dbscan):
            if label == 0:
                pc_rgb[i] = [0, 255, 0]
            elif label == 1:
                pc_rgb[i] = [0, 0, 255]
            else:
                pc_rgb[i] = [255, 0, 0]

        point_cloudO.mutable_rgbs()[:] = pc_rgb.T.astype(np.uint8)
        output.set_value(point_cloudO)

    def identify_moles(self, context, output):
        ik_found = False

        current_time = context.get_time()
        if (self._last_processed_time and current_time - self._last_processed_time < PROCESSING_TIME):
            ...
        self._last_processed_time = current_time

        # Retrieve the point cloud, but Eval(context) too slow
        point_clouds = [self.point_cloud1.Eval(context), self.point_cloud2.Eval(context), self.point_cloud3.Eval(context), self.point_cloud4.Eval(context)]

        point_clouds = [point_cloud.Crop(self.crop_lower, self.crop_upper) for point_cloud in point_clouds]

        merged_cloud = Concatenate(point_clouds)
        
        # Process the point cloud: crop and downsample
        rgbs_1 = merged_cloud.rgbs().T
        point_cloud_1 = merged_cloud.xyzs().T
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.1, min_samples=3).fit(point_cloud_1).labels_
        
        # Group points by labels into lists
        clustered_points = {label : [] for label in set(dbscan)}
        rgb_points = {label : [] for label in set(dbscan)}
        
        for i, label in enumerate(dbscan):

            clustered_points[label].append(point_cloud_1[i])
            rgb_points[label].append(rgbs_1[i])

        for k in list(clustered_points.keys()):

            v = clustered_points[k]

            clust_center = np.mean(v, axis=0)

            rgb = np.mean(rgb_points[k], axis=0)

            print("cluster: ", k, "mean: ", clust_center, "rgb: ", rgb)

            if np.linalg.norm(clust_center[:2]) < 0.1 or rgb[0] < 80 or np.var(rgb) < 10:
                del clustered_points[k]
        
        self.means = []
        for k, v in clustered_points.items():
            self.means.append(np.mean(v, axis=0))
            # print("cluster: ", k, "mean: ", self.means[-1])

        ind = np.random.randint(0, len(self.means))
        
        print("Perceived mole location:", np.array(self.means[ind]) + np.array([0, 0, 0.25]))
    
        ik_sol = self.ik.do_ik(np.array(self.means[ind]) + np.array([0, 0, 0.25])) # length 9 numpy array

        if ik_sol[0] == None:
            pass
        else:
            print("ik sol: ", ik_sol)
            ik_found = True

        if not ik_found:
            ik_sol = np.zeros(9)

        output.SetFromVector(ik_sol[:-2])
    
    def get_mole_pos_and_ik(self):
        pos = self.means[0]

        return pos, self.ik.do_ik(pos)

# Cell
MOLE_SPAWN_TIME = 2

builder = DiagramBuilder()

# set up main simulation
scenario = LoadScenario(data=scenario_data_multiple)
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
plant = station.GetSubsystemByName("plant")  # Access the plant from the station

scenario_int = LoadScenario(filename=FindResource("models/iiwa_and_wsg.dmd.yaml"))
plant_robot = MakeMultibodyPlant(
    scenario=scenario_int, model_instance_names=["iiwa", "wsg"]
)

# init the planner, trajectory follower, and perception
trajectory_follower = builder.AddSystem(TrajectoryFollower(plant_robot, plant))
perception = builder.AddSystem(Perception(plant_robot))



builder.Connect(
    trajectory_follower.get_output_port(),
    station.GetInputPort("iiwa.position"),
)


builder.Connect(
    station.GetOutputPort("iiwa.position_measured"),
    trajectory_follower.GetInputPort("iiwa_pos")        
)

builder.Connect(
    perception.GetOutputPort("IK_sol"),
    trajectory_follower.GetInputPort("IK_solution")
)

wsg_out = builder.AddSystem(WSGPositionOutput())

# builder.Connect(
#     trajectory_follower.GetOutputPort("iiwa_position"),
#     station.GetInputPort("iiwa.position"),
# )
builder.Connect(
    wsg_out.get_output_port(), 
    station.GetInputPort("wsg.position")
)

# Visualize the cropped point cloud
#point_cloud_visualizer = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, f"cropped cloud"))

#builder.Connect(perception.GetOutputPort("crop_cloud"), point_cloud_visualizer.cloud_input_port())
# builder.Connect(perception.GetOutputPort("dummy1"), pos_out.get_input_port())

to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder)
builder.Connect(to_point_cloud["camera1"].point_cloud_output_port(), perception.GetInputPort("camera1"))
builder.Connect(to_point_cloud["camera2"].point_cloud_output_port(), perception.GetInputPort("camera2"))
builder.Connect(to_point_cloud["camera3"].point_cloud_output_port(), perception.GetInputPort("camera3"))
builder.Connect(to_point_cloud["camera4"].point_cloud_output_port(), perception.GetInputPort("camera4"))


# Finalize the plant and build the simulation
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.get_mutable_context()

simulator.set_target_realtime_rate(0) # supposedly 0 makes it run as fast as possible

# Do simulation
meshcat.StartRecording()
for i in range(MOLE_COUNT):
    spawn_mole(i, plant, simulator)

try:
    simulator.AdvanceTo(simulator.get_context().get_time() + 60)
except Exception as e:
    print(e)
    pass
meshcat.PublishRecording()

TOTAL_END_TIME = time.time()
print("Total execution time: ", TOTAL_END_TIME - TOTAL_START_TIME)
print("Total moles contacted: ", total_moles_contacted)

# don't end the simulation until the user inputs to end it

while True:
    user_input = input("Enter any key to end the simulation: ")
    if user_input:
        break