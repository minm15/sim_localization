// src/mapMain.cpp

#include "sim_local/nclt_map.hpp"
#include "sim_local/nuscenes_map.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions opts;
    opts.allow_undeclared_parameters(true);
    opts.automatically_declare_parameters_from_overrides(true);
    auto driver = std::make_shared<rclcpp::Node>("map_main", opts);

    std::string dataset;
    driver->get_parameter_or("dataset", dataset, std::string("nuscenes"));

    if (dataset == "nuscenes") {
        RCLCPP_INFO(driver->get_logger(), "Starting Nuscenes map generator...");
        auto node = std::make_shared<sim_local::NuscenesMapNode>(opts);
        rclcpp::spin(node);

    } else if (dataset == "nclt") {
        RCLCPP_INFO(driver->get_logger(), "Starting NCLT map generator...");
        auto node = std::make_shared<sim_local::NCLTMapNode>(opts);
        rclcpp::spin(node);

    } else {
        RCLCPP_ERROR(driver->get_logger(),
                     "Unsupported dataset '%s', only 'nuscenes' and 'nclt' are implemented",
                     dataset.c_str());
    }

    rclcpp::shutdown();
    return 0;
}
