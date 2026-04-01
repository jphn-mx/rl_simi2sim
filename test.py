import time
import math

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/home/woan/mujoco_sim/model/dog/dog_V2_description/urdf/scene.xml')
d = mujoco.MjData(m)
m.opt.timestep = 0.005

print(f"驱动器数量: {m.nu}")
for i in range(m.nu):
    actuator = m.actuator(i)
    actuator_name = actuator.name if actuator.name else f"actuator_{i}"
    joint_id = actuator.trnid[0]  # 驱动器对应的关节ID
    joint_name = m.joint(joint_id).name if joint_id >= 0 else "None"
    print(f"  ctrl[{i}]: {actuator_name} -> 关节: {joint_name}")
print(f"关节数量: {m.nq}")
for i in range(m.nq):
    # 找到对应关节
    for j in range(m.njnt):
        joint = m.joint(j)
        if joint.qposadr[0] == i:
            print(f"  qpos[{i}]: {joint.name}")
            break
    else:
        print(f"  qpos[{i}]: 未找到对应关节")
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  cnt = 0
  while viewer.is_running() and time.time() - start < 300:
    step_start = time.time()

    '''测试step '''
    # d.ctrl = [0.1,0.1,-0.1,-0.1,
    #               0.8,0.,1.5,0.8,
    #               -1.5,-1.5,-1.5,-1.8]
    # d.ctrl index 11->YH4 10->ZH4
    mujoco.mj_step(m, d)
    
    '''测试step1 step2 '''
    # mujoco.mj_step1(m, d)
    # d.ctrl[1] = math.sin(cnt)
    # mujoco.mj_step2(m, d)
    
    '''测试forward '''
    # d.ctrl[0] = math.sin(cnt)
    # d.qpos[0] = math.sin(cnt)
    # mujoco.mj_forward(m, d)
    # print("qvel:",d.qvel)
    # print("qacc:",d.qacc)
    # print("qpos:",d.qpos)
    
    '''测试inverse '''
    # d.qacc[0] = math.sin(cnt)
    print(d.qpos)
    # Actuator names
    # print(" Actuator names:", m.actuator_names)  # 直接读取

    # # Joint names  
    # print(" Joint names:", m.joint_names)        # 直接读取
    
    # d.qvel[0] = 0
    # mujoco.mj_inverse(m, d)
    # print("qfrc_inverse",d.qfrc_inverse)
    
    cnt += 0.005

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = 0.005 - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)