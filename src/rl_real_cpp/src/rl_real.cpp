#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <yaml-cpp/yaml.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <Eigen/Dense>
#include <iomanip>


using namespace std::chrono_literals;

// Observation history class (lab version)
class ObsHistoryLab {
private:
    int hist_len;
    std::vector<double> omeg_buffer;
    std::vector<double> gravity_orientation_buffer;
    std::vector<double> cmd_buffer;
    std::vector<double> position_buffer;
    std::vector<double> velocity_buffer;
    std::vector<double> last_cation_buffer;

public:
    ObsHistoryLab(int /*num_obs*/, int hist_len) : hist_len(hist_len) {
        omeg_buffer.resize(3 * hist_len, 0.0);
        gravity_orientation_buffer.resize(3 * hist_len, 0.0);
        cmd_buffer.resize(3 * hist_len, 0.0);
        position_buffer.resize(12 * hist_len, 0.0);
        velocity_buffer.resize(12 * hist_len, 0.0);
        last_cation_buffer.resize(12 * hist_len, 0.0);
    }

    std::vector<double> update(const std::vector<double>& new_obs) {
        // Update omega buffer
        omeg_buffer.insert(omeg_buffer.end(), new_obs.begin(), new_obs.begin() + 3);
        omeg_buffer.erase(omeg_buffer.begin(), omeg_buffer.begin() + 3);

        // Update gravity orientation buffer
        gravity_orientation_buffer.insert(gravity_orientation_buffer.end(), new_obs.begin() + 3, new_obs.begin() + 6);
        gravity_orientation_buffer.erase(gravity_orientation_buffer.begin(), gravity_orientation_buffer.begin() + 3);

        // Update cmd buffer
        cmd_buffer.insert(cmd_buffer.end(), new_obs.begin() + 6, new_obs.begin() + 9);
        cmd_buffer.erase(cmd_buffer.begin(), cmd_buffer.begin() + 3);

        // Update position buffer
        position_buffer.insert(position_buffer.end(), new_obs.begin() + 9, new_obs.begin() + 21);
        position_buffer.erase(position_buffer.begin(), position_buffer.begin() + 12);

        // Update velocity buffer
        velocity_buffer.insert(velocity_buffer.end(), new_obs.begin() + 21, new_obs.begin() + 33);
        velocity_buffer.erase(velocity_buffer.begin(), velocity_buffer.begin() + 12);

        // Update last action buffer
        last_cation_buffer.insert(last_cation_buffer.end(), new_obs.begin() + 33, new_obs.end());
        last_cation_buffer.erase(last_cation_buffer.begin(), last_cation_buffer.begin() + 12);

        // Combine all buffers
        std::vector<double> total_obs;
        total_obs.insert(total_obs.end(), omeg_buffer.begin(), omeg_buffer.end());
        total_obs.insert(total_obs.end(), gravity_orientation_buffer.begin(), gravity_orientation_buffer.end());
        total_obs.insert(total_obs.end(), cmd_buffer.begin(), cmd_buffer.end());
        total_obs.insert(total_obs.end(), position_buffer.begin(), position_buffer.end());
        total_obs.insert(total_obs.end(), velocity_buffer.begin(), velocity_buffer.end());
        total_obs.insert(total_obs.end(), last_cation_buffer.begin(), last_cation_buffer.end());

        return total_obs;
    }

    void reset(int /*num_obs*/, int hist_len) {
        this->hist_len = hist_len;
        omeg_buffer.assign(3 * hist_len, 0.0);
        gravity_orientation_buffer.assign(3 * hist_len, 0.0);
        cmd_buffer.assign(3 * hist_len, 0.0);
        position_buffer.assign(12 * hist_len, 0.0);
        velocity_buffer.assign(12 * hist_len, 0.0);
        last_cation_buffer.assign(12 * hist_len, 0.0);
    }
};

// Math utility functions
std::vector<double> get_gravity_orientation(const std::vector<double>& quaternion) {
    double qw = quaternion[0];
    double qx = quaternion[1];
    double qy = quaternion[2];
    double qz = quaternion[3];

    std::vector<double> gravity_orientation(3, 0.0);
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy);
    gravity_orientation[1] = -2 * (qz * qy + qw * qx);
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz);

    return gravity_orientation;
}

// RL Real node class
class RLReal : public rclcpp::Node {
private:
    // ROS2 subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr obs_subscriber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr commands_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Configuration parameters
    std::string policy_path_;
    int num_actions_;
    int num_obs_;
    int num_hist_;
    double simulation_dt_;
    int control_decimation_;
    std::vector<double> default_angles_;
    double ang_vel_scale_;
    double dof_pos_scale_;
    double dof_vel_scale_;
    double action_scale_;
    std::vector<double> cmd_scale_;
    std::string simulator_type_;

    // State variables
    ObsHistoryLab obs_hist_;
    std::vector<double> actions_;
    torch::jit::script::Module policy_;
    double x_vel_ = 0.0;
    double y_vel_ = 0.0;
    double yaw_ = 0.0;
    std::vector<double> get_obs_;
    bool reset_symbol_ = false;
    std::vector<double> commands_;
    int counter_ = 0;
    std::vector<double> current_pos_;

    // Terminal settings
    int fd_;
    termios old_term_;
    int old_flags_;

public:
    RLReal(const std::string& name) : Node(name), obs_hist_(45, 10) {
        // Initialize subscribers and publishers
        obs_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/left_joint_states", 5, std::bind(&RLReal::obs_callback, this, std::placeholders::_1));
        
        commands_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/dog_joint_pos", 1);
        
        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 5, std::bind(&RLReal::imu_callback, this, std::placeholders::_1));

        // Load configuration
        load_config();

        // Reinitialize obs_hist_ with correct num_hist_
        obs_hist_.reset(num_obs_, num_hist_);

        // Initialize state variables
        actions_.resize(num_actions_, 0.0);
        get_obs_.resize(31, 0.0);
        current_pos_.resize(12, 0.0);

        // Load policy model
        try {
            policy_ = torch::jit::load(policy_path_);
            RCLCPP_INFO(this->get_logger(), "Policy model loaded successfully");
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading policy model: %s", e.what());
        }

        // Set default commands
        commands_ = lab2real(default_angles_);

        // Setup terminal for keyboard input
        setup_terminal();

        // Create and start threads
        std::thread commands_thread(&RLReal::commands_thread, this);
        std::thread model_thread(&RLReal::model_thread, this);
        
        // Detach threads to run independently
        commands_thread.detach();
        model_thread.detach();
        
        // Create timer
        // timer_ = this->create_wall_timer(5ms, std::bind(&RLReal::timer_callback, this));
    }

    ~RLReal() {
        // Restore terminal settings
        tcsetattr(fd_, TCSANOW, &old_term_);
        fcntl(fd_, F_SETFL, old_flags_);
    }

private:
    void load_config() {
        std::string config_file = "dog.yaml";
        std::string package_path = "/home/woan/rl_real/src/rl_real_py";
        std::string config_path = package_path + "/configs/" + config_file;

        try {
            YAML::Node config = YAML::LoadFile(config_path);
            policy_path_ = package_path + "/" + config["policy_path"].as<std::string>();
            num_actions_ = config["num_actions"].as<int>();
            num_obs_ = config["num_obs"].as<int>();
            num_hist_ = config["num_hist"].as<int>();
            simulation_dt_ = config["simulation_dt"].as<double>();
            control_decimation_ = config["control_decimation"].as<int>();

            // Load default angles
            const YAML::Node& default_angles_node = config["default_angles"];
            for (size_t i = 0; i < default_angles_node.size(); ++i) {
                default_angles_.push_back(default_angles_node[i].as<double>());
            }

            std::cout << "default_angles_: " << default_angles_ << std::endl;

            ang_vel_scale_ = config["ang_vel_scale"].as<double>();
            dof_pos_scale_ = config["dof_pos_scale"].as<double>();
            dof_vel_scale_ = config["dof_vel_scale"].as<double>();
            action_scale_ = config["action_scale"].as<double>();

            // Load cmd scale
            const YAML::Node& cmd_scale_node = config["cmd_scale"];
            for (size_t i = 0; i < cmd_scale_node.size(); ++i) {
                cmd_scale_.push_back(cmd_scale_node[i].as<double>());
            }

            simulator_type_ = config["simulator_type"].as<std::string>();

            RCLCPP_INFO(this->get_logger(), "Configuration loaded successfully");
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading config file: %s", e.what());
        }
    }

    void setup_terminal() {
        fd_ = STDIN_FILENO;
        tcgetattr(fd_, &old_term_);
        termios new_term = old_term_;
        new_term.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(fd_, TCSANOW, &new_term);
        old_flags_ = fcntl(fd_, F_GETFL);
        fcntl(fd_, F_SETFL, old_flags_ | O_NONBLOCK);
    }

    void obs_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        // Update joint positions
        for (int i = 0; i < 12; ++i) {
            get_obs_[7 + i] = msg->position[i];
        }
        current_pos_ = msg->position;

        // Update joint velocities
        get_obs_[19] = msg->velocity[0];
        get_obs_[20] = msg->velocity[1];
        get_obs_[21] = msg->velocity[2];
        get_obs_[22] = msg->velocity[3];
        get_obs_[23] = -msg->velocity[4];
        get_obs_[24] = -msg->velocity[5];
        get_obs_[25] = -msg->velocity[6];
        get_obs_[26] = msg->velocity[7];
        get_obs_[27] = msg->velocity[8];
        get_obs_[28] = -msg->velocity[9];
        get_obs_[29] = -msg->velocity[10];
        get_obs_[30] = -msg->velocity[11];
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // Update angular velocity
        get_obs_[0] = msg->angular_velocity.x;
        get_obs_[1] = msg->angular_velocity.y;
        get_obs_[2] = msg->angular_velocity.z;

        // Update orientation
        get_obs_[3] = msg->orientation.w;
        get_obs_[4] = msg->orientation.x;
        get_obs_[5] = msg->orientation.y;
        get_obs_[6] = msg->orientation.z;
    }

    std::vector<double> real2lab(const std::vector<double>& real) {
        std::vector<double> lab(12, 0.0);
        lab[0] = real[9];
        lab[1] = real[3];
        lab[2] = real[6];
        lab[3] = real[0];
        lab[4] = real[10];
        lab[5] = real[4];
        lab[6] = real[7];
        lab[7] = real[1];
        lab[8] = real[11];
        lab[9] = real[5];
        lab[10] = real[8];
        lab[11] = real[2];
        return lab;
    }

    std::vector<double> lab2real(const std::vector<double>& lab) {
        std::vector<double> real(12, 0.0);
        real[0] = lab[3];
        real[1] = lab[7];
        real[2] = lab[11];
        real[3] = lab[1];
        real[4] = lab[5];
        real[5] = lab[9];
        real[6] = lab[2];
        real[7] = lab[6];
        real[8] = lab[10];
        real[9] = lab[0];
        real[10] = lab[4];
        real[11] = lab[8];
        return real;
    }

    void get_key() {
        char ch;
        int n = read(fd_, &ch, 1);
        if (n == -1) {
            return; // No input
        }

        switch (ch) {
            case 'w':
                x_vel_ += 0.1;
                break;
            case 's':
                x_vel_ -= 0.1;
                break;
            case 'd':
                yaw_ -= 0.1;
                break;
            case 'a':
                yaw_ += 0.1;
                break;
            case 'l':
                y_vel_ -= 0.1;
                break;
            case 'j':
                y_vel_ += 0.1;
                break;
            case ' ':
                x_vel_ = 0.0;
                y_vel_ = 0.0;
                yaw_ = 0.0;
                break;
            case 'r':
                reset_symbol_ = true;
                break;
        }
    }

    void commands_thread() {
        while (rclcpp::ok()) {
            auto start_commands_time = std::chrono::high_resolution_clock::now();

            auto msg = std::make_shared<std_msgs::msg::Float64MultiArray>();
            if (reset_symbol_) {
                obs_hist_.reset(num_obs_, num_hist_);
                x_vel_ = 0.0;
                y_vel_ = 0.0;
                yaw_ = 0.0;
                commands_ = lab2real(default_angles_);
                RCLCPP_INFO(this->get_logger(), "reset");
                reset_symbol_ = false;
            }
            
            // Calculate commands with clipping
            std::vector<double> lab_commands(num_actions_, 0.0);
            for (int i = 0; i < num_actions_; ++i) {
                lab_commands[i] = actions_[i] * action_scale_ + default_angles_[i];
            }
            std::vector<double> commands = lab2real(lab_commands);

            for (int i = 0; i < 12; ++i) {
                std::cout << commands[i] << " ";
            }
            
            std::cout << std::endl;
            // Clip commands
            commands[0] = std::max(std::min(commands[0], 0.4), -0.4);
            commands[3] = std::max(std::min(commands[3], 0.4), -0.4);
            commands[6] = std::max(std::min(commands[6], 0.4), -0.4);
            commands[9] = std::max(std::min(commands[9], 0.4), -0.4);
            
            commands[1] = std::max(std::min(commands[1], 2.0), 0.0);
            commands[4] = std::max(std::min(commands[4], 2.0), 0.0);
            commands[7] = std::max(std::min(commands[7], 2.0), 0.0);
            commands[10] = std::max(std::min(commands[10], 2.0), 0.0);
            
            commands[2] = std::max(std::min(commands[2], -0.8), -2.0);
            commands[5] = std::max(std::min(commands[5], -0.8), -2.0);
            commands[8] = std::max(std::min(commands[8], -0.8), -2.0);
            commands[11] = std::max(std::min(commands[11], -0.8), -2.0);
            
            commands_ = commands;
            msg->data = commands_;

            // Publish commands
            commands_publisher_->publish(*msg);
            get_key();
            std::cout << "x_vel: " << std::fixed << std::setprecision(2) << x_vel_ << 
                      "     y_vel: " << std::fixed << std::setprecision(2) << y_vel_ << 
                      "     yaw: " << std::fixed << std::setprecision(2) << yaw_ << "\r" << std::flush;

            auto end_commands_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = end_commands_time - start_commands_time;
            if (dt.count() < 0.005) {
                std::this_thread::sleep_for(std::chrono::duration<double>(0.005 - dt.count()));
            }
        }
    }

    void model_thread() {
        while (rclcpp::ok()) {
            auto start_time = std::chrono::high_resolution_clock::now();

            std::vector<double> obs(45, 0.0);
            // Omega
            for (int i = 0; i < 3; ++i) {
                obs[i] = get_obs_[i];
            }
            // Gravity orientation
            std::vector<double> quaternion(get_obs_.begin() + 3, get_obs_.begin() + 7);
            std::vector<double> gravity_orientation = get_gravity_orientation(quaternion);
            for (int i = 0; i < 3; ++i) {
                obs[3 + i] = gravity_orientation[i];
            }
            // Command
            obs[6] = x_vel_ * cmd_scale_[0];
            obs[7] = y_vel_ * cmd_scale_[1];
            obs[8] = yaw_ * cmd_scale_[2];
            // Position
            std::vector<double> real_pos(get_obs_.begin() + 7, get_obs_.begin() + 19);
            std::vector<double> lab_pos = real2lab(real_pos);
            for (int i = 0; i < num_actions_; ++i) {
                obs[9 + i] = (lab_pos[i] - default_angles_[i]) * dof_pos_scale_;
            }
            // Velocity
            std::vector<double> real_vel(get_obs_.begin() + 19, get_obs_.begin() + 31);
            std::vector<double> lab_vel = real2lab(real_vel);
            for (int i = 0; i < num_actions_; ++i) {
                obs[9 + num_actions_ + i] = lab_vel[i] * dof_vel_scale_;
            }
            // Last action
            for (int i = 0; i < num_actions_; ++i) {
                obs[9 + 2 * num_actions_ + i] = actions_[i];
            }
            
            // Update observation history
            std::vector<double> total_obs = obs_hist_.update(obs);
            
            // Convert to torch tensor
            torch::Tensor obs_tensor = torch::from_blob(total_obs.data(), {1, static_cast<long>(total_obs.size())}, torch::kFloat32);
            obs_tensor = torch::clamp(obs_tensor, -100.0, 100.0);
            
            // Run policy
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(obs_tensor);
            torch::Tensor output = policy_.forward(inputs).toTensor();
            output = torch::clamp(output, -100.0, 100.0);
            
            // Update actions
            for (int i = 0; i < num_actions_; ++i) {
                actions_[i] = output[0][i].item<double>();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = end_time - start_time;
            if (dt.count() < 0.02) {
                std::this_thread::sleep_for(std::chrono::duration<double>(0.02 - dt.count()));
            }
        }
    }
    
    void timer_callback() {
        get_key();
        counter_++;

        if (reset_symbol_) {
            obs_hist_.reset(num_obs_, num_hist_);
            x_vel_ = 0.0;
            y_vel_ = 0.0;
            yaw_ = 0.0;
            commands_ = lab2real(default_angles_);
            reset_symbol_ = false;
        }

        if (counter_ % control_decimation_ == 0) {
            // Prepare observation
            std::vector<double> obs(45, 0.0);
            
            // Omega
            for (int i = 0; i < 3; ++i) {
                obs[i] = get_obs_[i];
            }
            
            // Gravity orientation
            std::vector<double> quaternion(get_obs_.begin() + 3, get_obs_.begin() + 7);
            std::vector<double> gravity_orientation = get_gravity_orientation(quaternion);
            for (int i = 0; i < 3; ++i) {
                obs[3 + i] = gravity_orientation[i];
            }
            
            // Command
            obs[6] = x_vel_ * cmd_scale_[0];
            obs[7] = y_vel_ * cmd_scale_[1];
            obs[8] = yaw_ * cmd_scale_[2];
            
            // Position
            std::vector<double> real_pos(get_obs_.begin() + 7, get_obs_.begin() + 19);
            std::vector<double> lab_pos = real2lab(real_pos);
            for (int i = 0; i < num_actions_; ++i) {
                obs[9 + i] = (lab_pos[i] - default_angles_[i]) * dof_pos_scale_;
            }
            
            // Velocity
            std::vector<double> real_vel(get_obs_.begin() + 19, get_obs_.begin() + 31);
            std::vector<double> lab_vel = real2lab(real_vel);
            for (int i = 0; i < num_actions_; ++i) {
                obs[9 + num_actions_ + i] = lab_vel[i] * dof_vel_scale_;
            }
            
            // Last action
            for (int i = 0; i < num_actions_; ++i) {
                obs[9 + 2 * num_actions_ + i] = actions_[i];
            }
            
            // Update observation history
            std::vector<double> total_obs = obs_hist_.update(obs);
            
            // Convert to torch tensor
            torch::Tensor obs_tensor = torch::from_blob(total_obs.data(), {1, static_cast<long>(total_obs.size())}, torch::kFloat32);
            obs_tensor = torch::clamp(obs_tensor, -100.0, 100.0);
            
            // Run policy
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(obs_tensor);
            torch::Tensor output = policy_.forward(inputs).toTensor();
            output = torch::clamp(output, -100.0, 100.0);
            
            // Update actions
            for (int i = 0; i < num_actions_; ++i) {
                actions_[i] = output[0][i].item<double>();
            }
            
            // Calculate commands with clipping
            std::vector<double> lab_commands(num_actions_, 0.0);
            for (int i = 0; i < num_actions_; ++i) {
                lab_commands[i] = actions_[i] * action_scale_ + default_angles_[i];
            }
            std::vector<double> commands = lab2real(lab_commands);
            
            for (int i = 0; i < num_actions_; ++i) {
                std::cout << commands[i] << " ";
            }
            // Clip commands
            commands[0] = std::max(std::min(commands[0], 0.4), -0.4);
            commands[3] = std::max(std::min(commands[3], 0.4), -0.4);
            commands[6] = std::max(std::min(commands[6], 0.4), -0.4);
            commands[9] = std::max(std::min(commands[9], 0.4), -0.4);
            
            commands[1] = std::max(std::min(commands[1], 2.0), 0.0);
            commands[4] = std::max(std::min(commands[4], 2.0), 0.0);
            commands[7] = std::max(std::min(commands[7], 2.0), 0.0);
            commands[10] = std::max(std::min(commands[10], 2.0), 0.0);
            
            commands[2] = std::max(std::min(commands[2], -0.8), -2.0);
            commands[5] = std::max(std::min(commands[5], -0.8), -2.0);
            commands[8] = std::max(std::min(commands[8], -0.8), -2.0);
            commands[11] = std::max(std::min(commands[11], -0.8), -2.0);
            
            commands_ = commands;
            
        }
        
        std::cout << std::endl;
        // Publish commands
        auto msg = std::make_shared<std_msgs::msg::Float64MultiArray>();
        msg->data = commands_;
        commands_publisher_->publish(*msg);
        
        // Print velocity information
        // std::cout << "x_vel: " << std::fixed << std::setprecision(2) << x_vel_ << 
        //           "     y_vel: " << std::fixed << std::setprecision(2) << y_vel_ << 
        //           "     yaw: " << std::fixed << std::setprecision(2) << yaw_ << "\r" << std::flush;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RLReal>("rl_real");
    RCLCPP_INFO(node->get_logger(), "rl_real start ...");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}