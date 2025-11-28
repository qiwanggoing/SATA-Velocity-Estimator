
#include <deploy_real/low_level_ctrl.hpp>
double get_diff_norm(vector<double> &vec1, vector<double> &vec2)
{
    assert(vec1.size() == vec2.size());
    double sum = 0.0;
    int n = vec1.size();
    for (int i = 0; i < n; i++)
    {
        sum += pow((vec1[i] - vec2[i]), 2);
    }
    return sqrt(sum);
}

double get_norm(vector<double> &vec)
{
    double sum = 0.0;
    for (double num : vec)
        sum += pow(num, 2);
    return sqrt(sum);
}
float clip_val(float val, int index)
{
    if ((index + 1) % 3 == 0)
    {
        if (val > 35.55)
            val = 35.55;
        else if (val < -35.55)
            val = -35.55;
    }
    else
    {
        if (val > 23.7)
            val = 23.7;
        else if (val < -23.7)
            val = -23.7;
    }
    return val;
}
LowLevelControl::LowLevelControl() : Node("finite_state_machine_node")
{
    this->declare_parameter("is_simulation", true);
    this->get_parameter("is_simulation", is_simulation);
    if (is_simulation)
    {
        RCLCPP_INFO(this->get_logger(), "Running in simulation mode.");
        cmd_puber_ = this->create_publisher<unitree_go::msg::LowCmd>("/mujoco/lowcmd", 10);
        state_suber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/mujoco/lowstate", 10, std::bind(&LowLevelControl::state_callback, this, std::placeholders::_1)); // 500HZ
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Running in real mode.");
        cmd_puber_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);
        state_suber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", 10, std::bind(&LowLevelControl::state_callback, this, std::placeholders::_1)); // 500HZ
    }
    
    // target_pos_puber_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/pos", 10); // 可选的调试发布器
    init_cmd();
    
    // !!!修改!!!: 订阅新的 /rl/target_torques 话题
    target_torque_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/rl/target_torques", 10, std::bind(&LowLevelControl::target_torque_callback, this, std::placeholders::_1)); // 50 HZ
        
    joy_suber_ = this->create_subscription<sensor_msgs::msg::Joy>("/joy", 10, bind(&LowLevelControl::joy_callback, this, placeholders::_1));
    timer_ = this->create_wall_timer(std::chrono::milliseconds(5), std::bind(&LowLevelControl::state_machine, this)); // 500hz
}

void LowLevelControl::init_cmd()
{
    // 站立/趴下 时使用的 Kp/Kd
    kp = vector<double>(12, 50.0);
    kd = vector<double>(12, 1);

    for (int i = 0; i < 20; i++)
    {
        cmd_msg_.motor_cmd[i].mode = 0x01; // 伺服模式
        cmd_msg_.motor_cmd[i].q = PosStopF;
        cmd_msg_.motor_cmd[i].kp = 0;
        cmd_msg_.motor_cmd[i].dq = VelStopF;
        cmd_msg_.motor_cmd[i].kd = 0;
        cmd_msg_.motor_cmd[i].tau = 0;
    }
    pos_data_ = std_msgs::msg::Float32MultiArray();
}

void LowLevelControl::state_callback(unitree_go::msg::LowState::SharedPtr msg)
{
    for (int i = 0; i < 12; i++)
    {
        motor[i] = msg->motor_state[i];
    }
}

// !!!修改!!!: 新的回调函数
void LowLevelControl::target_torque_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    recieved_data_ = true;
    rl_target_torques_.data = msg->data; // 存储到新的 HPP 成员变量中
    if (is_standing_ and should_run_policy_)
        run_policy(); // 如果在SATA状态，立即调用
}

void LowLevelControl::joy_callback(sensor_msgs::msg::Joy::SharedPtr msg)
{
    // !!! 新增 !!!: X 键 (buttons[2]) 触发 E-Stop
    if (msg->buttons[2]) 
    {
        is_estop_ = true;
        should_stand_ = false;
        should_laydown_ = false;
        should_run_policy_ = false;
        RCLCPP_ERROR(this->get_logger(), "!!! E-STOP ACTIVATED !!!");
        return; // E-Stop 优先于一切
    }
    // ... (状态切换逻辑保持不变) ...
    if (is_laydown_ and msg->buttons[7]) 
    {
        should_stand_ = true;
        should_laydown_ = false;
        should_run_policy_ = false;
        is_estop_ = false;
        RCLCPP_INFO(this->get_logger(), "Standing Up...");
    }
    else if (msg->buttons[0]) // Button A
    {
        should_laydown_ = true;
        should_stand_ = false;
        should_run_policy_ = false;
        is_estop_ = false; 
        RCLCPP_INFO(this->get_logger(), "Resetting to Laydown...");
    }
    else if (is_standing_ and msg->buttons[4] and msg->buttons[5]) // BUtton LB and Button RB
    {
        should_laydown_ = false;
        should_stand_ = false;
        should_run_policy_ = true;
        is_estop_ = false; 
    }

    if (msg->axes[2]==-1 && msg->axes[5]==-1)
        rclcpp::shutdown();
}

void LowLevelControl::state_machine()
{
    // !!! 新增 !!!: E-Stop 具有最高优先级
    if (is_estop_)
    {
        cout << "E-STOP active! Sending passive command." << endl;
        for (int i = 0; i < 12; i++)
        {
            cmd_msg_.motor_cmd[i].mode = 0x00; // !!! 0x00 = 被动/阻尼 模式 !!!
            cmd_msg_.motor_cmd[i].q = 0;
            cmd_msg_.motor_cmd[i].kp = 0;
            cmd_msg_.motor_cmd[i].dq = 0;
            cmd_msg_.motor_cmd[i].kd = 0;
            cmd_msg_.motor_cmd[i].tau = 0;
        }
        get_crc(cmd_msg_);
        cmd_puber_->publish(cmd_msg_);
        return; 
    }
    state_obs();
    if (is_uncontrolled_ and !is_laydown_) // init pos: lay down
    {
        state_transform(laydown_angels_);
        cout << "should init" << endl;
    }
    else if (is_laydown_ and should_stand_) // from lay down state to stand up state
    {
        state_transform(standing_angels_);
        cout << "should stand up" << endl;
    }
    else if (is_standing_ and should_laydown_) // from stand up state to lay down state
    {
        state_transform(laydown_angels_);
        cout << "should lay down" << endl;
    }
    else if (is_standing_ and should_run_policy_) // from stand up state to policy state
    {
        // run_policy(); // 不在这里调用，改为在 target_torque_callback 中调用
        // cout << "should run policy" << endl;
    }
    else
    {
        // 保持 PD 控制 (趴下或站立)
        cout << "should keep" << endl;
        vector<double> target_angles;
        if (is_standing_)
            target_angles = standing_angels_;
        else
            target_angles = laydown_angels_;
        
        for (int i = 0; i < 12; i++)
        {
            cmd_msg_.motor_cmd[i].mode = 0x01; 
            cmd_msg_.motor_cmd[i].q = target_angles[i];
            cmd_msg_.motor_cmd[i].kp = kp[i]; // 使用 PD
            cmd_msg_.motor_cmd[i].dq = 0;
            cmd_msg_.motor_cmd[i].kd = kd[i]; // 使用 PD
            cmd_msg_.motor_cmd[i].tau = 0;
        }
        get_crc(cmd_msg_);
        cmd_puber_->publish(cmd_msg_);
    }
}

// !!!修改!!!: run_policy() 发送纯力矩
void LowLevelControl::run_policy()
{
    if (!recieved_data_)
    {
        cout << "Have not recieved data from policy yet" << endl;
        // (可选) 发送一个零力矩的安全命令
        // init_cmd(); 
        // cmd_puber_->publish(cmd_msg_);
        return;
    }

    for (int i = 0; i < 12; i++)
    {
        cmd_msg_.motor_cmd[i].mode = 0x01; // 伺服模式
        cmd_msg_.motor_cmd[i].q = 0;      // 位置目标设为0 (或 PosStopF)
        cmd_msg_.motor_cmd[i].kp = 0;     // Kp 设为 0
        cmd_msg_.motor_cmd[i].kd = 0;     // Kd 设为 0
        cmd_msg_.motor_cmd[i].dq = 0;     // 速度目标设为 0
        cmd_msg_.motor_cmd[i].tau = rl_target_torques_.data[i]; // !!!设置SATA计算的力矩!!!
    }
    get_crc(cmd_msg_);
    cmd_puber_->publish(cmd_msg_);
}

void LowLevelControl::state_obs()
{
    // ... (状态观测逻辑保持不变) ...
    vector<double> q(12, 0);
    vector<double> dq(12, 0);

    for (int i = 0; i < 12; i++)
    {
        q[i] = motor[i].q;
        dq[i] = motor[i].dq;
    }
    if (get_diff_norm(q, laydown_angels_) < 0.25 && get_norm(dq) < 0.15)
    {
        if (!is_laydown_)
        {
            motion_time_ = 0;
            rate_count_ = 0;
        }
        is_laydown_ = true;
        is_uncontrolled_ = false;
        is_standing_ = false;
    }
    else if (get_diff_norm(q, standing_angels_) < 0.3 && get_norm(dq) < 0.15)
    {
        if (!is_standing_)
        {
            motion_time_ = 0;
            rate_count_ = 0;
        }
        is_standing_ = true;
        is_laydown_ = false;
        is_uncontrolled_ = false;
    }
}

void LowLevelControl::state_transform(vector<double> &target_angels)
{
    // ... (状态转换逻辑(PD控制)保持不变) ...
    motion_time_++;
    if (motion_time_ >= 0 && motion_time_ < 20)
        for (int i = 0; i < 12; i++)
            q_init_[i] = motor[i].q;

    if (motion_time_ >= 20)
    {
        rate_count_++;
        double rate = rate_count_ / 400.0; // needs count to 200
        for (int i = 0; i < 12; i++)
        {
            q_des_[i] = jointLinearInterpolation(q_init_[i], target_angels[i], rate);
            cmd_msg_.motor_cmd[i].mode = 0x01; 
            cmd_msg_.motor_cmd[i].q = q_des_[i];
            cmd_msg_.motor_cmd[i].kp = kp[i]; // 使用 PD
            cmd_msg_.motor_cmd[i].dq = 0;
            cmd_msg_.motor_cmd[i].kd = kd[i]; // 使用 PD
            cmd_msg_.motor_cmd[i].tau = 0;
        }

        get_crc(cmd_msg_);
        cmd_puber_->publish(cmd_msg_);
    }
}

double LowLevelControl::jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    // ... (保持不变) ...
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos * (1 - rate) + targetPos * rate;
    return p;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);                        
    auto node = std::make_shared<LowLevelControl>(); 
    rclcpp::spin(node); 
    rclcpp::shutdown(); 
    return 0;
}