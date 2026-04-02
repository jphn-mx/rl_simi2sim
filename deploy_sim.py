
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import onnxruntime as ort

# 导入功能模块
from utils.math_utils import get_gravity_orientation, pd_control

import sys
import termios
import tty
def normalize_policy_output(policy_output):
    if isinstance(policy_output, torch.Tensor):
        return policy_output
    if isinstance(policy_output, np.ndarray):
        return torch.from_numpy(policy_output)
    if isinstance(policy_output, (list, tuple)):
        for item in policy_output:
            if isinstance(item, torch.Tensor):
                return item
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item)
        raise TypeError(f"Policy output sequence does not contain a tensor or ndarray: {type(policy_output)}")
    raise TypeError(f"Unsupported policy output type: {type(policy_output)}")

class obs_history_lab:
    #term
    def __init__(self,num_obs,hist_len,obs_index):
        self.hist_len = hist_len
        self.omeg_buffer = [0.]*3*self.hist_len
        self.gravity_orientation_buffer = [0.]*3*self.hist_len
        self.cmd_buffer = [0.]*3*self.hist_len
        self.position_buffer = [0.]*12*self.hist_len
        self.velocity_buffer = [0.]*12*self.hist_len
        self.last_cation_buffer = [0.]*12*self.hist_len
        self.obs_index = obs_index  
    
    def update(self,new_obs):
        self.omeg_buffer.extend(new_obs[:3])
        self.omeg_buffer = self.omeg_buffer[3:]
        
        self.gravity_orientation_buffer.extend(new_obs[3:6])
        self.gravity_orientation_buffer = self.gravity_orientation_buffer[3:]
        self.cmd_buffer.extend(new_obs[6:9])
        self.cmd_buffer = self.cmd_buffer[3:]
        self.position_buffer.extend(new_obs[9:21])
        self.position_buffer = self.position_buffer[12:]
        self.velocity_buffer.extend(new_obs[21:33])
        self.velocity_buffer = self.velocity_buffer[12:]
        self.last_cation_buffer.extend(new_obs[33:45])
        self.last_cation_buffer = self.last_cation_buffer[12:]

        total_obs = []
        for obs in self.obs_index:
            if obs == 'base_ang_vel':
                total_obs += self.omeg_buffer
            elif obs == 'joint_pos_rel':
                total_obs += self.position_buffer
            elif obs == 'joint_vel_rel':
                total_obs += self.velocity_buffer
            elif obs == 'last_action':
                total_obs += self.last_cation_buffer
            elif obs == 'velocity_commands':
                total_obs += self.cmd_buffer
            elif obs == 'projected_gravity':
                total_obs += self.gravity_orientation_buffer
        
        # total_obs = self.omeg_buffer + self.gravity_orientation_buffer + self.cmd_buffer  \
        #             + self.position_buffer + self.velocity_buffer + self.last_cation_buffer
        
        return np.array(total_obs)

class obs_history_gym:
    #time
    def __init__(self,num_obs,hist_len):
        self.hist_len = hist_len
        self.num_obs = num_obs
        self.total_obs = [0.]*self.num_obs*self.hist_len
    
    def update(self,new_obs):
        self.total_obs[self.num_obs:] = self.total_obs[:-self.num_obs]
        self.total_obs[:self.num_obs] = new_obs

        return np.array(self.total_obs)


import fcntl
import os
fd = sys.stdin.fileno()

# 保存终端状态
old_term = termios.tcgetattr(fd)
tty.setcbreak(fd)

# 设置为非阻塞
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

def restore_terminal():
    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
def get_key():
    global x_vel, y_vel, yaw,reset_requested

    try:
        ch = sys.stdin.read(1)
    except IOError:
        return  # 没有输入时直接返回

    if not ch:
        return

    if ch == 'w':
        x_vel += 0.1
    elif ch == 's':
        x_vel -= 0.1
    elif ch == 'd':
        yaw -= 0.1
    elif ch == 'a':
        yaw += 0.1
    elif ch == 'l':
        y_vel -= 0.1
    elif ch == 'j':
        y_vel += 0.1
    elif ch == ' ':
        x_vel = y_vel = yaw = 0.0
    elif ch == 'r':
        reset_requested = True


# ==================== 主程序 ====================
if __name__ == "__main__":
    x_vel = 0.0
    y_vel = 0.0
    yaw = 0.0
    # get config file name from command line
    config_file = 'dog.yaml'
    with open(f"configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        num_history = config["num_history"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        print("default_angles = ", default_angles)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        obs_index = config["obs_index"]
        # 如果有相位，在基础观测维度上加2
        num_obs = config["num_obs"]
        model_type = config["model_type"]
        simulate_type = config["simulate_type"]
        if simulate_type == 'lab':
            obs_index_in_lab = config["obs_index_in_lab"]
            obs_index_in_mj = config["obs_index_in_mj"]
            action_index_in_lab = config["action_index_in_lab"]
            action_index_in_mj = config["action_index_in_mj"]

            mj2lab = [obs_index_in_mj.index(name) for name in obs_index_in_lab]
            lab2mj = [action_index_in_lab.index(name) for name in action_index_in_mj]
            print("mj2lab = ", mj2lab)
            print("lab2mj = ", lab2mj)
        else:
            obs_index_in_gym = config["obs_index_in_gym"]
            obs_index_in_mj = config["obs_index_in_mj"]
            action_index_in_gym = config["action_index_in_gym"]
            action_index_in_mj = config["action_index_in_mj"]

            mj2gym = [obs_index_in_mj.index(name) for name in obs_index_in_gym]
            gym2mj = [action_index_in_gym.index(name) for name in action_index_in_mj]
            print("mj2gym = ", mj2gym)
            print("gym2mj = ", gym2mj)
        print("num_actions = ", num_actions)
        print("num_obs = ", num_obs)
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        reset_requested = False

    # define context variables
    if simulate_type == 'lab':
        obs_hist = obs_history_lab(num_obs,num_history,obs_index) #lab
    else:
        obs_hist = obs_history_gym(num_obs,num_history) # gym
    actions = np.zeros(num_actions, dtype=np.float32)
    # target_dof_pos = default_angles.copy()
    if simulate_type == 'lab':
        target_dof_pos = default_angles[lab2mj]
    else:
        target_dof_pos = default_angles[gym2mj]
    obs = []

    counter = 0
    print("xml_path = ", xml_path)
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # 保存初始状态用于reset
    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()
    initial_cmd = [0.,0.,0.]  # 保存初始速度命令

    # print(initial_qpos)
    # load policy
    
    if model_type == "jit":
        policy = torch.jit.load(policy_path)
        print(f"Loaded JIT model from {policy_path}")
    elif model_type == "onnx":
        # 检查文件扩展名是否为.onnx
        if not policy_path.endswith('.onnx'):
            policy_path += '.onnx'
        policy = ort.InferenceSession(policy_path)
        print(f"Loaded ONNX model from {policy_path}")
    # init_gamepad(config)  # 初始化游戏手柄（传入配置）
    
    # 查找pelvis body的ID
    pelvis_body_id = None
    try:
        pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    except Exception:
        try:
            pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        except Exception:
            print("警告: 未找到pelvis_link或base_link，将使用body 0")
            pelvis_body_id = 0
      
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # 处理reset请求
            if reset_requested:
                # 重置机器人状态
                
                d.qvel[:] = initial_qvel
                d.qpos[:7] = initial_qpos[:7]
                # 重置速度命令
                cmd[0] = 0.
                cmd[1] = 0.
                cmd[2] = 0.
                # 重置动作和目标位置
                actions[:] = 0.0

                if simulate_type == 'lab':
                    obs_hist.__init__(num_obs,num_history,obs_index)
                else:
                    obs_hist.__init__(num_obs,num_history)
                # 重置相位
                phase = 0.0
                # 重置计数器
                counter = 0
                if simulate_type == 'lab':
                    target_dof_pos = default_angles[lab2mj]
                else:
                    target_dof_pos = default_angles[gym2mj]
                # 重置数据记录
                # reset_recording()
                # 清除外力
                d.xfrc_applied[:] = 0.0
                # d.qpos[7:19] = target_dof_pos
                # 重置reset标志
                reset_requested = False
                print("机器人状态已重置")
                # 执行一步仿真以应用重置
                mujoco.mj_step(m, d)
                viewer.sync()
                # time.sleep(0.5)
                continue
            
            get_key()
            cmd[0] = x_vel
            cmd[1] = y_vel
            cmd[2] = yaw
            
            # tau = pd_control(target_dof_pos, np.array([d.qpos[0+7],d.qpos[9+7],d.qpos[3+7],d.qpos[6+7],d.qpos[1+7],d.qpos[10+7],d.qpos[4+7],d.qpos[7+7],d.qpos[2+7],d.qpos[11+7],d.qpos[5+7],d.qpos[8+7]]), kps, np.zeros_like(kds), np.array([d.qvel[0+6],d.qvel[9+6],d.qvel[3+6],d.qvel[6+6],d.qvel[1+6],d.qvel[10+6],d.qvel[4+6],d.qvel[7+6],d.qvel[2+6],d.qvel[11+6],d.qvel[5+6],d.qvel[8+6]]), kds)
            if simulate_type == 'lab':
                tau = pd_control(target_dof_pos, np.array(d.qpos[7:][mj2lab][lab2mj]), kps[lab2mj], np.zeros_like(kds), np.array(d.qvel[6:][mj2lab][lab2mj]), kds[lab2mj])
                d.ctrl = tau
            else:
                tau = pd_control(target_dof_pos, np.array(d.qpos[7:][mj2gym][gym2mj]), kps[gym2mj], np.zeros_like(kds), np.array(d.qvel[6:][mj2gym][gym2mj]), kds[gym2mj])
                d.ctrl = tau
            # d.ctrl = target_dof_pos
            
            # 执行一步仿真
            print(f'x_vel:{cmd[0]}     y_vel:{cmd[1]}     yaw:{cmd[2]}     \r',end="")
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0  and counter > 400:
                # create observation
                if simulate_type == 'lab':
                    qj = d.qpos[7:][mj2lab]
                    dqj = np.array(d.qvel[6:][mj2lab])
                else:
                    qj = d.qpos[7:][mj2gym]
                    dqj = np.array(d.qvel[6:][mj2gym])
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                obs = []
                for idx in obs_index:
                    if idx == 'last_action':
                        obs.append(actions)
                    if idx == 'velocity_commands':
                        obs.append(cmd * cmd_scale)
                    if idx == 'base_ang_vel':
                        obs.append(omega)
                    if idx == 'projected_gravity':
                        obs.append(gravity_orientation)
                    if idx == 'joint_pos_rel':
                        obs.append(qj)
                    if idx == 'joint_vel_rel':
                        obs.append(dqj)

                obs = np.concatenate(obs,axis=0)
                
                total_obs = obs_hist.update(obs)
                # print(total_obs)
                if simulate_type=='lab':
                    # print(torch.from_numpy(total_obs).unsqueeze(0).shape)
                    # obs_tensor = torch.clip(torch.from_numpy(total_obs).squeeze(),-100.0,100.0)
                    obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0),-100.0,100.0)
                else:
                    obs_tensor = torch.clip(torch.from_numpy(total_obs).unsqueeze(0),-100.0,100.0)
                # policy inference
                if model_type == "jit":
                    # actions = torch.clip(policy(obs_tensor.float()),-100.0,100.0).detach().numpy().squeeze()
                    policy_output = normalize_policy_output(policy(obs_tensor.float()))
                    actions = torch.clip(policy_output,-100.0,100.0).detach().cpu().numpy().squeeze()
                elif model_type == "onnx":
                    # ONNX 推理
                    input_name = policy.get_inputs()[0].name
                    obs_np = obs_tensor.float().numpy()
                    if obs_np.ndim == 1:
                        obs_np = obs_np[np.newaxis, :]  # 添加批次维度
                    outputs = policy.run(None, {input_name: obs_np})
                    actions = np.clip(outputs[0], -100.0, 100.0).squeeze()
                # print(actions)
                ### dog
                if simulate_type == 'lab':
                    action = np.array(actions[lab2mj])
                    target_dof_pos = action * action_scale + default_angles
                else:
                    
                    target_dof = actions * action_scale + default_angles
                    # print(default_angles)
                    target_dof_pos = np.array(target_dof[gym2mj])
            target_dof_pos[0] = np.clip(target_dof_pos[0],-0.4,0.4)
            target_dof_pos[1] = np.clip(target_dof_pos[1],-0.4,0.4)
            target_dof_pos[2] = np.clip(target_dof_pos[2],-0.4,0.4)
            target_dof_pos[3] = np.clip(target_dof_pos[3],-0.4,0.4)

            target_dof_pos[4] = np.clip(target_dof_pos[4],-0.0,2.)
            target_dof_pos[5] = np.clip(target_dof_pos[5],-0.0,2.)
            target_dof_pos[6] = np.clip(target_dof_pos[6],-0.0,2.)
            target_dof_pos[7] = np.clip(target_dof_pos[7],-0.0,2.)

            target_dof_pos[8] = np.clip(target_dof_pos[8],-2.,-0.8)
            target_dof_pos[9] = np.clip(target_dof_pos[9],-2.,-0.8)
            target_dof_pos[10] = np.clip(target_dof_pos[10],-2.,-0.8)
            target_dof_pos[11] = np.clip(target_dof_pos[11],-2.,-0.8) ## clip 
            
            viewer.sync()

            # 控制步长
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

restore_terminal()

