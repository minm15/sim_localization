// src/map_main.cpp

#include "sim_local/nuscenes_map.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions opts;
    opts.allow_undeclared_parameters(true);
    opts.automatically_declare_parameters_from_overrides(true);
    auto driver = std::make_shared<rclcpp::Node>("map_main", opts);

    // std::string dataset = driver->declare_parameter<std::string>("dataset", "nuscenes");
    std::string dataset = driver->get_parameter("dataset").as_string();

    if (dataset != "nuscenes") {
        RCLCPP_ERROR(driver->get_logger(),
                     "Unsupported dataset '%s', only 'nuscenes' is implemented", dataset.c_str());
        return 1;
    }

    // spin the map generator
    auto map_node = std::make_shared<sim_local::NuscenesMapNode>(opts);
    rclcpp::spin(map_node);

    rclcpp::shutdown();
    return 0;
}
